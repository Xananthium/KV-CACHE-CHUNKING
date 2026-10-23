# Gemma4 / vLLM / TurboQuant adapter audit

The existing Qwen CPU bank does **not** support this backend. A read-only audit
of the deployed model configuration, captured serving metadata, and installed
vLLM source identifies a feasible extension boundary and several correctness
requirements. `gemma4_contract.py` validates proposed metadata; it neither
imports vLLM nor reads GPU allocations. All returned plans explicitly say
`runtime_execution_supported=False`.

The audit used the converted Gemma4-31B model, not the Qwen family named in its
legacy serving alias. [The source/config hashes and layer descriptions are
recorded here](results/gemma4_compatibility/audit.json). Captured deployment
metadata reports Hauhau **model** revision `9654466e82d83f5ebfe1518a369bc5900873abb1`;
this is not a vLLM source commit. The installed vLLM commit is unverified;
the audit records individual installed source-file hashes. The deployment has
`VLLM_USE_V2_MODEL_RUNNER=1`, tensor parallelism 2, a 200,000-token serving limit, primary block size 16,
`turboquant_4bit_nc`, `VLLM_TQ_UNIFIED_ATTN=1`, and MTP with two speculative
tokens. Actual per-group tensor strides, block sizes, leases, and resident pages
were **not** captured; these must be obtained from a live allocator before an
import can be authorized. Captured metrics are evidence about that deployment,
not a fresh query to the backend being profiled by another process.

## Model semantics the adapter must preserve

Gemma4 combines local and global attention and uses proportional RoPE in its
global layers. This is also described in the [official model card](https://ai.google.dev/gemma/docs/core/model_card_4).
The following exact dimensions come from the local config and implementation.
Layer indices start at zero; head counts below are **per tensor-parallel rank**.

| Layer class | Layers | Query/KV heads | Head dimension | Position/mask rule | Stored cache |
|---|---|---:|---:|---|---|
| Local, compressed | 47 | 16 / 8 | 256 | RoPE theta 10,000; window 1,024 | TQ4 SoA |
| Global, compressed | 9 | 16 / 2 | 512 | Proportional RoPE theta 1,000,000; global causal | TQ4 SoA |
| Native local boundary | 0, 1, 58 | 16 / 8 | 256 | Same local rule | BF16 K and V |
| Native global boundary | 59 | 16 / 2 | 512 | Same global rule | BF16 K and V |

Global layers are 5, 11, ..., 59. The attention scale is **1.0**, not the
reference bank's default `1/sqrt(head_dim)`. Q and K receive learned RMSNorm;
V receives RMSNorm without learned weights. `attention_k_eq_v` shares projection
weights in global layers, but stored K and V are not interchangeable: their
normalization differs and only K receives RoPE. The target has no decoder KV
sharing layers (`num_kv_shared_layers=0`).

Global proportional RoPE activates 64 of 256 angle pairs. Frequencies are
`theta ** (-2*j/512)` for `j=0..63`, followed by 192 zeros. Rotation uses two
256-element halves: inactive coordinates occur in both halves. Applying an
ordinary 128-dimensional rotary slice or dividing exponents by 128 is wrong.
The local layers rotate all 128 pairs of their 256-dimensional heads.

`final_logit_softcapping=30` applies to vocabulary logits after the decoder.
The config has no attention-logit softcap. Do not reject this configuration by
mistaking the final cap for an attention cap; conversely, the audited unified
TQ path explicitly rejects attention softcaps and ALiBi. Text-only import is
the first scope: image spans use bidirectional masks, including local-window
clamping, and require separate feature provenance and mask support.

## The packed cache ABI

The active unified path uses **SoA within each physical block**. Its tensor
shape can look identical to the older AoS path while containing incompatible
bytes. Some module-level comments still describe FP16 values; the actual
configuration and store kernel use four-bit values for this preset.

For head dimension `D`, each head/token has `D/2` key-index bytes, `D/2`
value-index bytes, and three FP16 metadata fields. Thus payload size is 262
bytes at D=256 and 518 bytes at D=512. For block size B and H local KV heads:

1. The data region occupies `B * H * D` bytes, ordered token/head/packed data.
2. The metadata region follows, organized as `[H, 3, B]` FP16 fields: key norm,
   value scale, value minimum. Padding/outer strides belong to the allocator.
3. The key scalar already includes norm correction: `||K|| / ||centroid_vector||`.
   A codec that stores raw `||K||` is incompatible even if all dimensions match.

Keys use the deterministic normalized Sylvester Hadamard matrix and a
dimension-specific Lloyd-Max codebook. Values use uniform quantization. The
seed field in the config is not the transform identity; the audited backend
uses a pure Hadamard transform. Codebook, implementation, norm convention,
packing, dtype, TP partition, layer, group, and allocation generation all belong
in the cache ABI. Native boundary layers require their own backend layout.

A zero-copy whole-block borrow is possible in principle when producer and
consumer use the same engine allocation and ABI. A token-row slice of the
apparent tensor is not a valid packed chunk: its metadata lives elsewhere in
the block. The first contract therefore requires complete immutable blocks.
Supporting arbitrary conclusion boundaries needs segment row ranges and
private append/COW handling; it must also account for the full blocks retained
by those ranges. The primary size 16 does not prove every cache group uses 16.

## Position correction is possible, but needs a kernel hook

For the same fixed decoded cache vector, the orthogonal identity still holds:

`qᵀ R(delta) k_hat = (R(-delta) q)ᵀ k_hat`.

Zero-frequency pairs make proportional RoPE a valid instance of this identity.
However, TQ attention consumes keys in a Hadamard basis. The correct query is
`H R(-delta) q`, not `R(-delta) H q`. Ten CPU contract tests include both local
and global rotary dimensions, verify the correct identity to ten decimal
places, and reject the reversed transform order. These are synthetic vector
algebra tests, not Gemma inference or CUDA-kernel parity tests.

This could avoid persistent rotated-key copies and re-quantization for a
per-segment shift. It is exact only relative to the *same already-quantized*
cache, subject to implementation rounding. It does not recover original
unquantized keys, remove donor-context effects, or equal freshly recomputed
summary KV. Each segment needs its own shifted query before the current
Hadamard step; the existing kernel transforms a query for the ordinary table.

Keeping original positions with holes also requires a kernel change. Current
unified attention derives query positions from `seq_len - query_len` and key
positions from table offsets. The model runner uses computed-token counts both
for RoPE positions and for slot mapping. Removing blocks without separating
logical positions, physical slots, and sequence lengths changes causal/local
masks or writes to the wrong slots. Old positions still count toward the
supported position range; physical compaction does not create unlimited RoPE
context.

## Concrete integration hooks

All paths below are relative to the audited `vllm/` package. The audit JSON
records source hashes and symbol line numbers, including local modifications;
upstream documentation alone is not a byte-level ABI specification.

| Concern | Existing hook | Required extension |
|---|---|---|
| Export after writes | `model_executor/layers/attention/attention.py`: `get_attention_context`, cache-update operation; TQ `do_kv_cache_update` / `_store_kv` | Publish only committed target tokens after GPU writes complete; retain owning allocator references |
| Group/layout discovery | `Attention.get_kv_cache_spec`, `v1/kv_cache_interface.py`, model-runner cache configuration | Resolve actual per-layer/group shape, stride, backend, TP rank, padding and bytes; never infer from the alias |
| Physical ownership | `v1/core/block_pool.py`: `touch`, `free_blocks`, `is_block_writable` | Bank and in-flight leases, allocation epochs, and release ordering; a Python pointer alone pins nothing |
| Request attachment | `v1/core/kv_cache_manager.py`: `get_computed_blocks`, `allocate_slots`, `free` | Separate contextual imports from APC keys and route imported blocks through scheduler accounting |
| Position/slot separation | Active V2: `v1/worker/gpu/model_runner.py`, `input_batch.py`, `block_table.py` | Supply logical RoPE positions independently of dense token indices and writable cache slots |
| Attention | `v1/attention/backends/turboquant_attn.py` and `ops/turboquant_soa/triton_turboquant_unified_attention.py` | Segment descriptors, logical causal/local masks, and pre-Hadamard query shifts for every execution path |
| Window eviction | `v1/core/single_type_kv_cache_manager.py`: skipped-block removal and sliding-window manager | Publish resident conclusion pages before eviction; reject null/missing pages and retain leases through reuse |
| Speculation | Active V2: `v1/worker/gpu/spec_decode/gemma4/speculator.py`: `_setup_gemma4_kv_sharing` | Preserve target/draft mapping, accepted-token boundaries, lookahead retention and rollback; keep speculative writes private |

The source manifest also includes legacy runner counterparts for comparison;
their presence does not establish that the deployment executes them.

Ordinary APC hashes include parent context, token IDs, and extra identity fields;
the allocator reference count protects shared blocks from eviction. These
mechanisms are described in [vLLM's prefix-caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/).
A donor conclusion moved behind a different prefix must use a separate
contextual namespace. It must never appear in exact-prefix hit statistics.
The installed code also has partial-hit/COW machinery; that is not evidence
that arbitrary contextual splicing is already supported.

The MTP proposer maps draft attention to the last target layer of each type,
which here means local layer 58 and global layer 59. These are native-cache
boundary layers. An adapter supporting only compressed layers would miss
important readers. Exporting speculative suffix bytes before acceptance can
also contaminate a shared block. A small isolated implementation can first
disable MTP, but it does not qualify for this production deployment until the
two-token MTP path has its own parity and cancellation tests.

Sliding-window state may already have been evicted. A useful initial target is
to publish a short generated summary immediately when it finishes, while its
local-layer pages remain resident. A sparse old chunk with missing local K/V
cannot be made complete by inventing zeros or relabeling null pages. Rebuilding
missing layers would be additional computation and a separate approximation.

## Implementation sequence and acceptance gates

1. Add an internal lease/export API on an isolated feature branch. Freeze a
   complete accepted prefix with resolved groups and both ranks. Resume through
   the ordinary attention path with unchanged positions and masks. Compare
   logits, greedy output, physical bytes, and four branches with private tails.
2. Add logical-position/segment descriptors to runner and attention. Validate
   unchanged-cache parity first, then deletion with holes, then contiguous
   rebasing. Cover local/global layers, native/TQ layers, decode and chunked
   prefill, and graph capture. No donor approximation may be hidden in the
   numerical-parity control.
3. Integrate MTP, preemption, offload, cancellation and deferred frees. A handle
   is engine-local and invalid after restart unless payload and ABI are safely
   restored; a recycled integer block ID must never revive an old handle.
   Account for bank pins as real cache pressure and all readers as references.
4. Run the generated-work → separate generated-summary → subsequent-question
   experiment on Gemma with construction and amortization costs. Compare full
   history, fresh identical summary text, and the two contextual placements.
   Keep omitted-fact controls. The [small Qwen generated-summary pilot](results/generated_summary/summary.json)
   supports useful explicit summaries, not hidden recovery of omitted facts.

At least one uncached suffix token is normally needed to obtain new logits;
current APC intentionally caps a complete prefix hit before its final token.
Opaque chunk handles must retain token IDs and provenance for scheduling,
sampling penalties, usage, and subsequent branching even if text is not resent
over the API. Sharing removes duplicate physical allocations; each private
tail, each new unique chunk, and attention over retained global tokens still
costs memory or computation.

Develop the full experiment on a fork/feature branch first. A narrow upstream
PR can follow for a proven lease or metadata hook. The inspected vLLM
contribution instructions require duplicate-work checks, a human review of
every change, relevant tests/evaluations, and AI-assistance disclosure. No PR or
production integration was created by this audit.

## Reproduce the CPU checks

```bash
python -m unittest test_gemma4_contract -v
python audit_gemma4.py --model-config /path/to/model/config.json \
  --vllm-root /path/to/site-packages/vllm \
  --runtime-snapshot /path/to/sanitized-runtime-snapshot.json \
  --captured-metrics /path/to/already-captured-metrics.txt \
  --output results/gemma4_compatibility
```

The collector parses files only. It does not execute installed vLLM modules,
query services, verify weight contents, or establish that changed source still
has the audited semantics. The portable test fixture contains actual model
configuration and synthetic allocation metadata; no transcripts, credentials,
weights, or GPU pointers are published.
