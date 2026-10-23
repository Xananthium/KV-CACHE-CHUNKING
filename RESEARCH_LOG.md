# KV-CACHE-CHUNKING research log

## 2026-10-23: working hypothesis and initial tests

The proposed system stores complete useful KV chunks by handle. A branch may
share a stable prefix, compute an intermediate reasoning span, and produce a
conclusion. We test whether the prefix and contextualized conclusion can be
retained while removing the intermediate span, and whether multiple later runs
can reference those retained blocks without storing duplicate copies.

This builds on paged attention, prefix caching, modular prompt caching, KV
eviction, and non-prefix cache reuse. Novelty is unestablished. An experimental
cache state is not assumed to be an exact substitute for recomputing its visible
text in a new context.

The first model is Qwen2.5-0.5B-Instruct, CPU fp32. This is an engineering screen;
results do not establish quality or speed on the production Gemma4-31B model or
its custom TurboQuant KV format. Model and script hashes are recorded per run.

Initial retrieval results, 12 questions:

| Method | Correct |
|---|---:|
| Full recomputation | 10/12 |
| Exact prefix reuse | 10/12 |
| Raw middle-chunk transplant | 3/12 |
| RoPE-rebased transplant | 10/12 |

The raw transplant's failure shows why positions must be controlled. Rebased
reuse preserved answer accuracy on these benign cases, while changing logits.
This does not imply exact semantic equivalence.

Follow-up results and branch-compaction methods are recorded in the per-run
JSON/CSV files. Negative controls are retained. In particular, a preliminary
test did not recover a secret fact appearing only in a removed donor prefix
from an otherwise neutral retained chunk. A cached conclusion must not be
assumed to contain all details of its preceding reasoning.

## Prior work to compare

- vLLM prefix caching: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- Prompt Cache: https://arxiv.org/abs/2311.04934
- CacheBlend: https://arxiv.org/abs/2405.16444
- Cache-Craft: https://arxiv.org/abs/2502.15734
- KVzip: https://arxiv.org/abs/2505.23416
- Memorizing Transformers: https://arxiv.org/abs/2203.08913
- LongMem: https://arxiv.org/abs/2306.07174
- Extended Mind Transformers: https://arxiv.org/abs/2406.02332
- LMCache: https://docs.lmcache.ai/

## Implementation boundary

Research code and a shared-block reference implementation live in this repository.
A production implementation must integrate block allocation, reference tracking,
positional handling, and segmented attention inside the inference backend. The
existing production vLLM server is not modified by these CPU experiments.

Shared storage removes duplicate physical copies. It does not make retained
blocks, attention work, or newly generated KV states free, nor does it provide
unlimited logical context.

## 2026-10-23: completed screen and owning chunk bank

The five quality runs now contain 720 scored rows over twelve retrieval records,
twelve hidden-fact records, and eight arithmetic ledgers with two layouts.
They are paired interventions, not 720 independent questions. The full report is
in README.md; every run preserves its prompts, token IDs, source snapshot, hashes,
answers, NLL, cache sizes, timings, and numerical diagnostics.

Helpful donor context with all corrected layers repaired two retrieval errors
(12/12 versus 10/12 full). Keeping only its middle eight or upper eight donor
layers reduced correctness to 5/12 and 4/12. Hidden-fact recovery from a neutral
chunk stayed 0/12, while retaining the useful donor prefix visibly gave 12/12.

In the chat-turn branch test, keeping a complete conclusion after deleting the
reasoning preserved 8/8 answers with original position holes and with rebasing,
matching freshly prefilled summary text and full history. Asking for omitted
opening balances gave 0/8 from every compacted variant versus 8/8 full history.
The opaque-conclusion case had a weak 3/8 full baseline and 0/8 compacted; it
remains reported rather than silently filtered.

The new CPU bank owns each chunk once, shares immutable handles, and keeps local
writable tails. Five views share 1,990,656 physical bytes while referencing
9,953,280 logical bytes. Tests cover owning slices, four forks, tails, sealing,
references, eviction, GQA, causal masking, per-block query rotation, and the
explicit exact-versus-contextual import distinction. Actual Qwen model inference
through segmented bank attention matches dense cached logits within 3.3e-5 and
returns the same correct answer in both position modes. A real-model tensor
fixture checks layers 0, 12, and 23 against independent dense SDPA. This is not
a measured throughput improvement or a production vLLM integration.

The requested summary-pass workflow is the next quality/cost experiment:
generate work, separately generate a sufficient durable summary, then compare
that same summary's text and contextual KV across later uses. Charge for both
generation passes and bank construction. Today's hand-authored conclusion tests
do not establish automatic-summary quality or make missing details recoverable.

Implementation direction: feature branch/fork first, then a narrow upstream PR
for any proven engine hooks. Production custom TurboQuant compatibility remains
unverified; adapter_contract.py explicitly rejects that unsupported ABI.

## 2026-10-23: generated work and generated summary pilot

The previously proposed separate summary pass is now implemented and evaluated
in `generated_summary.py`. Four independently sampled arithmetic records produce
model-generated work and two model-generated summary variants. KV comes directly
from those generation passes; no discarded work is replayed to obtain the
retained conclusion. The same generated text is used by the fresh-summary control.

Complete summaries retain both queried facts in 8/8 tests for all four methods.
With the closing-only instruction, fresh text and rebased KV answer 4/4 closing
questions, while original-position holes answer 3/4. The model includes the
supposedly omitted opening fact in one of four summaries. On the three cases
where it actually omits that fact, all compacted methods score 0/3; full history
scores 3/3. This does not demonstrate extra hidden memory from summary KV.

The pilot adds 64 scored rows (784 total), not 64 independent tasks. It ran for
153.5 seconds on the pinned 0.5B model, CPU FP32, two threads. Work and summary
generation costs are charged separately; shared-host query timings and logical
KV-size reductions are not physical allocation or production speed claims.
Next experiments should measure larger tasks and reuse amortization, and connect
generated summary chunks to the owning bank before any vLLM integration claim.

## 2026-10-23: actual Gemma4/TurboQuant compatibility audit

Read-only inspection of the converted Gemma4-31B config, captured runtime
metadata, and installed vLLM code establishes the adapter boundary. There are
50 local-window and 10 global layers with different head dimensions and rotary
rules; the captured TP2 runtime also protects layers 0, 1, 58, and 59 with BF16
caches. The other layers use TQ4 SoA packing, including key norm correction
folded at store time. Final vocabulary-logit softcap 30 is distinct from the
absent attention softcap.

The portable metadata contract rejects shape-compatible but differently packed
blocks, wrong model/config/code identities, missing shards, unfinished or
unpinned pages, unsupported partial blocks, and multimodal spans. Ten CPU tests
pass. A mathematical check confirms query-side proportional-RoPE shifts remain
valid for a fixed decoded cache, but must precede the Hadamard transform.
The tests are synthetic allocation/vector checks, not Gemma model evaluation.

Required serving work includes immutable allocator leases, all-rank/group
attachment, separate logical and slot positions, local/global masks, MTP
accepted-state handling, and safe preemption/offload/cancellation. Old local
pages can already be absent. The current runner and TQ kernels derive masks
from dense sequence offsets, so removing interior blocks alone is incorrect.
The full plan and source hashes are in `GEMMA4_ADAPTER.md` and
`results/gemma4_compatibility/`. No production package, service, or GPU was used
or changed. The Qwen bank remains the only implemented attention runtime here.
