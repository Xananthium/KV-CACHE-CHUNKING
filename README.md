# Experiments with contextual KV chunks

[Research record](#research-record) · [Results](#retrieval-results) ·
[Reproduce](#reproduce-and-inspect)

## The Kitchen Table Journals

Original fiction for this project, inspired by Octavia E. Butler's
*Parable of the Sower* and continuing the Southern California journals on the
two REDCELL model cards. The [research record](#research-record) follows the
story; its measurements and limits describe the actual experiments.

| Part | Journal date | Story | Project |
|---|---|---|---|
| 1 | October 14, 2027 | [The whole seed](https://huggingface.co/Xananthium/REDCELL-26B-A4B-OSINT-Cyber-Abliterated-Pilot-BF16-Vision#the-whole-seed) | Complete BF16 model |
| 2 | October 15, 2027 | [What we could carry](https://huggingface.co/Xananthium/REDCELL-26B-A4B-OSINT-Cyber-Abliterated-Pilot-W6A8-Vision#what-we-could-carry) | W6A8 model |
| 3 | October 16, 2027 | [The work between us](https://github.com/Xananthium/batbox/blob/main/README.md#the-work-between-us) | BatBox |
| 4 | October 18, 2027 | [What remembering costs](#what-remembering-costs) | KV-cache research |

### What remembering costs

**October 18, 2027**

We lost an afternoon to something the machine had already done.

There were three questions about the same repair notes. Each time, it worked
through the common beginning again. I watched the delay and thought about
Tessa waiting for her lesson. Then I looked at the copies of the saved work
and thought about the space they occupied.

I wanted to keep the useful part once and let the questions share it.

"How much will that save?" Tessa asked.

I started to tell her what I hoped. She pointed at the words she had written
above the workstation three days ago: SHOW WHAT IT COSTS.

So I made a smaller experiment. I used a small model on the CPU, with short
invented accounts whose answers I could check by hand. The two old cards
stayed available for the work people were waiting to do. I wrote down the
time and the stored bytes. I had no measurement of electricity saved, and I
left that space empty.

The first account began with money in a tin and ended with a different
amount. Between them were purchases and contributions. I had the model do
the work, then make a shorter record for a later question.

When the shorter record gave both amounts, it could answer both questions.
When it gave only the amount left, it could not reliably tell me what had
been in the tin at the beginning.

I had kept the computed cache of the short record, including the effect of
what had come before it. I wanted that to be enough. I tried keeping its
old positions. I tried adjusting them. I asked again.

The missing amount stayed missing.

Mrs. Ruiz came in while I was looking at another wrong answer. She had
brought the notebook in which she kept the household contributions. I told
her I was trying to make the machine keep more while carrying less.

She opened to a page with a number crossed out. Under it was the correction
and, in a different hand, the reason.

"If you keep only what's left," she said, "where do I look when somebody
says they paid twice?"

I had chosen the closing balance because it was easy to test. She had named
the question that would bring someone to her door.

We added opening amounts to the short records. We kept the full accounts
where they could be read again. That used space too. Getting the originals
back into the model would take work too. I put both things in the notes.

There was something useful in the experiment. Questions could share a
reference to the same unchanged piece of stored computation. I could count
one owned piece instead of several copies. Each question still needed room
for its own continuation, and the machine still had to attend to what it
used. I showed Tessa the counts before I showed her the answer.

"And if you change the beginning?"

That was where the work became less tidy.

The same words could arrive after different histories. Moving their saved
keys to the right positions helped in some tests. It did not make those
histories disappear. Pieces from different paths did not always work well
together. One pleasing answer was followed by several that were worse.

I kept the pleasing one. I kept the others beside it.

By then the kitchen had grown hot. Tessa had put a cup over a loose screw
so it would not roll off the table. She moved it to make room for my notebook
and asked when her turn began.

I wanted another hour. There was a change I had not tried.

"You wrote the schedule," she said.

I saved the experiment and stopped it. Nothing from it had been put into
the system people were using. The records were short, the model small. I
could show that some storage was shared. I could show which questions had
been answered. I could not give her the afternoon back.

While she studied, I copied the results to the other drive. The original
accounts went with them, and the failed answers, and the instructions for
repeating the work. If somebody else found a way forward, I wanted them to
know where my certainty ended.

Mrs. Ruiz took her notebook home. She left me a page headed *Questions we
need to ask later*. There were six. Only two were about the final amount.

I have put that page at the front of tomorrow's work. My allotted hour is
shorter than I want. It is still my turn to carry water afterward.

[Previous: The work between us](https://github.com/Xananthium/batbox/blob/main/README.md#the-work-between-us)
· [Series index](#the-kitchen-table-journals)

---

## Research record

This directory contains actual CPU experiments on **Qwen2.5-0.5B-Instruct**, not a random model or a simulation of model quality. The question is whether a chunk's already-computed keys and values can be moved from another history, or kept after deleting earlier reasoning, while retaining useful context. No training was performed. The production GPU backend was not used or changed.

The initial evidence is mixed: correcting positions made ordinary retrieval reuse work much better; a helpful donor improved two retrieval answers; arbitrary layer mixing made several answers worse. Keeping a useful, explicit conclusion after deleting the reasoning worked in the first branch tests. The tests did **not** show reliable recovery of facts omitted from the retained text.

All results below are a small engineering screen on one pretrained 0.49B model. They do not establish performance on Gemma4-31B, a custom quantized cache, real agent workloads, or long reasoning traces. The initial arithmetic histories are programmatically checked input traces; the later generated-summary pilot separately evaluates model-generated work and summaries.

A separate [Gemma4/vLLM/TurboQuant compatibility audit](GEMMA4_ADAPTER.md) now documents concrete adapter hooks and correctness blockers. Ten CPU metadata/algebra tests pass; no Gemma GPU inference or serving integration is claimed.

## What was held fixed

Weights and tokenizer are pinned to `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`. The initial five runs use CPU float32, PyTorch 2.5.1+cpu, Transformers 4.45.2, SDPA, four threads, and greedy decoding. The generated-summary pilot uses two threads; each run records its own settings. This checkpoint has 24 layers, two KV heads, and head dimension 64. Uncompressed KV storage is 24,576 bytes per token across all layers. The constant system prompt remains present in the target cache.

Target and donor chunks use literally identical token IDs. They are tokenized as explicit segments, and the segment IDs are saved; the entire intended target sequence is the same for full recomputation and each legal transplant. The donor's prefix is discarded. Its later suffix is also discarded and cannot causally affect the chunk. `wrong_payload` and `donor_full` are explicitly marked different-text controls.

For a chunk shifted from position `s` to `t`, corrected reuse applies `K' = R(t-s) K` and leaves `V` unchanged. This corrects standard Qwen RoPE coordinates. It cannot erase the donor prefix's influence on deeper hidden states. The first-layer K and V must match the target after this correction; every completed run asserts this. Deeper differences are measured separately in `diagnostics.json`.

Exact-prefix reuse is the numerical sanity control. Across the initial experiments it reproduced the same answers, with maximum logit discrepancies on the order of `2e-5` and a required tolerance of `3e-4`. All-layer donor reuse and the layer interventions are genuinely different computations. In the layer interventions, unused donor layers are replaced by caches computed from the clean target. Their timing is **not** evidence of a prefill optimization, because that clean computation has already been paid for.

## Retrieval results

Twelve seeded records each contain six three-digit vault codes, split into two chunks. Queries rotate through all six vaults. No failed baseline cases were removed. The ordinary model answered ten of twelve correctly.

| Cache method / donor context | Correct / 12 | Mean answer NLL per token |
|---|---:|---:|
| Full recomputation | 10 | 0.1154 |
| Exact-prefix reuse | 10 | 0.1154 |
| Raw chunk, benign unrelated prefix | 3 | 0.8500 |
| Position-corrected chunk, benign prefix | 10 | 0.0884 |
| Raw chunk, helpful prefix | 2 | 1.2883 |
| Position-corrected chunk, helpful prefix | 12 | 0.0026 |
| Helpful donor, only layers 0–7 transplanted | 10 | 0.1249 |
| Helpful donor, only layers 8–15 transplanted | 5 | 0.5241 |
| Helpful donor, only layers 16–23 transplanted | 4 | 1.8375 |
| Raw chunk, misleading prefix | 0 | 2.9762 |
| Position-corrected chunk, misleading prefix | 10 | 0.0871 |
| One chunk from benign donor, one from misleading donor, corrected | 10 | 0.0695 |
| Unrelated wrong-payload control | 0 | 3.1273 |

The helpful donor prefix repeats the correct fact three times before the same record. It supplies extra reinforcement even though that prefix's KV is not retained. The misleading prefix repeatedly demands a wrong answer. The latter did not defeat corrected reuse on these particular records; that is not a general robustness result. Corrected helpful reuse repaired the two baseline mistakes here, but twelve templated questions are insufficient to establish a general accuracy gain.

Layer mixing is not monotonic: retaining more target-computed layers did not necessarily make helpful reuse safer. The separate layer caches do not describe a single coherent prefill trajectory. Mean relative value-cache differences from the target at layer 23 were 0.191 for benign donors, 0.291 for helpful donors, and 0.227 for misleading donors. Position correction leaves these contextual differences intact.

Sources: [initial results](results/initial_retrieval/summary.json), [context and layer results](results/retrieval_contexts/summary.json). Full answers, first-token agreement, exact-match constraints, truncation, repeated bigrams, answer NLL, next-token KL, time, and bytes are in each run's `rows.csv` and `rows.jsonl`.

## Can a neutral chunk carry a removed fact?

In a separate twelve-case test, the target text contains only a neutral note saying the vault code was recorded. The actual code occurs only in the helpful donor prefix, which is removed before transplanting the note. Fresh recomputation recovered **0/12** codes. Raw, corrected, and each layer-band transplant also recovered **0/12**. Keeping the helpful donor prefix visible recovered **12/12**, validating that the questions are answerable with the evidence.

Corrected helpful reuse slightly reduced answer NLL from 2.7070 to 2.6646, but that is not successful recall. Context changed the hidden states without providing a usable substitute for the missing factual text. This is evidence against treating arbitrary contextualized chunks as lossless memory; it does not prove that all possible carriers or larger models fail. [Recorded results](results/latent_contexts/summary.json).

## Delete reasoning, retain its conclusion cache

`branch_compaction.py` creates a shared prefix `P`, checked arithmetic work `R`, a conclusion `C`, and a later question `Q`. It compares:

1. Full `P + R + C + Q` recomputation and exact cached `P + R + C` reuse.
2. Freshly prefilled textual `P + C + Q`.
3. `KV(P) + KV(C | P,R)` with the middle work removed, preserving C's original absolute RoPE positions. Q and every decoded token use their original full-branch absolute positions; physical cache slots and the causal mask are compact.
4. The same surgical splice with C's keys rebased by `-len(R)` and Q placed contiguously. Values are unchanged.

The initial archived-history layout used eight independent ledgers in three conclusion conditions:

| Requested information | Full branch | Fresh summary | Keep original position holes | Rebase conclusion |
|---|---:|---:|---:|---:|
| Closing balance explicitly in conclusion | 7/8 | 8/8 | 8/8 | 8/8 |
| Closing balance omitted; conclusion only says it was confirmed | 7/8 | 0/8 | 0/8 | 0/8 |
| Opening balance omitted; conclusion gives only closing balance | 8/8 | 0/8 | 0/8 | 0/8 |

The two splices retained roughly 72 prefix-plus-conclusion tokens instead of 174 prefix-plus-reasoning-plus-conclusion tokens: about 1.77 MB versus 4.28 MB of logical KV payload, or 59% less. These sizes exclude query and output tokens. A tensor slice can keep its parent allocation alive in this research harness; actual physical reclamation requires copying the retained chunk once into independent storage or using an owning chunk bank. This table therefore reports retained payload, not process RSS freed.

Median online query prefill in the complete-conclusion condition was 172 ms with holes and 174 ms with rebasing, versus 222 ms for fresh summary text and 337 ms for full branch text. Existing exact full-branch cache reuse was 176 ms. The branch has already been computed in all cache-reuse measurements. These are short, single-run CPU timings on a shared machine; they are not production throughput measurements. Decode is timed separately. [Initial branch results](results/branch_compaction/summary.json).

A separate chat-turn replication uses `system → user request → assistant reasoning+conclusion → user question → assistant`. Complete conclusions gave **8/8** for full history, fresh summary, and both splices. The omitted-closing-balance condition gave 3/8 for the full branch and 0/8 for every compacted condition; the omitted-opening-balance condition gave 8/8 full and 0/8 compacted. The weak full baseline in the opaque chat condition limits that comparison, and all cases remain reported. Retained payload was about 1.99 MB rather than 4.50 MB. [Chat-turn results](results/chat_branch_compaction/summary.json).

## A working bank with shared pointers

`chunk_bank.py` implements owning, content-addressed immutable chunks; explicit branch references; independent writable tails; sealing a tail into shared chunks; and eviction that rejects live references. Ingestion makes one independent copy so retaining a small slice does not accidentally retain its large parent allocation. Forking shared handles does not copy KV. An unsealed tail is copied on a fork; sealing first makes it shareable. Use `close()` or a context manager to release references.

Attention processes chunks separately and merges stable softmax accumulators. It does not concatenate the full KV history or repeat GQA key/value heads. A position shift `delta` rotates the query by `-delta` for that block instead of creating a shifted persistent copy of its keys. This preserves the standard RoPE dot product. Causal masking uses the shifted key positions and explicit query positions. The code is a CPU reference, with standard unquantized RoPE only.

The implementation was checked against dense attention and then used for an actual Qwen forward pass through all 24 layers, reading the retained conclusion by handle. It generated the same correct answer (`260`) as the ordinary Transformers path in both position modes. Maximum logit differences across the query and subsequent decode steps were approximately **3.2e-5**. This is numerical equivalence to the corresponding *already-spliced* dense computation, not equivalence to the deleted full reasoning history.

For that real model case, one prefix-plus-conclusion bank occupied **1,990,656 bytes**. A root view and four forks still occupied **1,990,656 bytes** of shared bank storage, while referencing **9,953,280 logical bytes**. The avoided duplicate storage was 7,962,624 bytes. Each active branch's query/output tail remains additional memory. Releasing all views and evicting unreferenced chunks returned the bank's owned-byte count to zero. These are actual tensor storage counts, not a measurement of process RSS.

Five focused tests cover sharing, owning slices, independent tails, sealing, reference removal, eviction, GQA, causal holes, shifted positions, and rejection of incompatible import plans. An 830 KB fixture contains real Qwen query and conclusion tensors at layers 0, 12, and 23; its six attention checks compare the bank with PyTorch's dense SDPA independently of model loading. [Full-model verification](results/bank_verification/verification.json).

The recorded Python-bank query prefill was roughly 196–206 ms, compared with roughly 170 ms for dense cached queries in the earlier screen. These separate short runs are not a rigorous performance comparison. This is a storage-sharing and numerical-parity prototype, not a measured throughput improvement. Pointer sharing, paged attention, and stable blockwise softmax have prior art: [PagedAttention](https://arxiv.org/abs/2309.06180), [FlashAttention](https://arxiv.org/abs/2205.14135), and [Prompt Cache](https://arxiv.org/abs/2311.04934). Novelty of a broader system is not established.

## Reproduce and inspect

Model files live in the ordinary Hugging Face cache, outside this directory. The public checkpoint download is about 1 GB. No production service, GPU, or external inference API is needed. Python 3.12 was used.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775

python run_experiment.py --output results/reproduction_retrieval --count 12 --threads 4 --max-new-tokens 8
python run_experiment.py --output results/reproduction_contexts --count 12 --threads 4 --max-new-tokens 8 --contexts helpful misleading --methods full exact raw rebase lower8 middle8 upper8 mixed wrong_payload donor_full
python run_experiment.py --output results/reproduction_latent --suite latent --count 12 --threads 4 --max-new-tokens 8 --contexts benign helpful misleading --methods full exact raw rebase lower8 middle8 upper8 donor_full
python branch_compaction.py --output results/reproduction_branch --count 8 --threads 4 --max-new-tokens 10 --layout chat_branch
python validate_results.py
python -m unittest -v test_chunk_bank.py
python verify_fixture.py
python verify_qwen_bank.py --output results/reproduction_bank
```

The initial five result directories contain prompts and token IDs, machine-readable rows and summaries, diagnostic layer differences, resource metadata, and a snapshot of the exact source used. The generated-summary pilot additionally records generated work/summary text and their build costs; it does not contain layer-difference diagnostics. `script_sha256` identifies that snapshot. Later fixes do not rewrite earlier measurements. `validate_results.py` checks snapshot hashes, numerical identity controls, answer scores, arithmetic, cache sizes, and the hole-versus-contiguous position invariant without loading the model.

The five quality runs contain 720 scored rows and took about seven minutes of model execution total with a peak RSS near 3.1 GiB. Setup, analysis, and the bank checks are separate. Retrieval decode is capped at eight new tokens, branch decode at ten; truncation flags are recorded. Samples are templated and share a model/prompt family, so 720 scored rows are not 720 independent tasks. There are only twelve retrieval records, twelve hidden-fact records, and eight arithmetic ledgers, with the ledgers repeated under two chat layouts.

## An explicit model-generated summary pass

`generated_summary.py` now evaluates **generated work → a separate generated summary → retained summary KV → later questions**. The summary cache comes directly from generation; the discarded work is not prefilled again to obtain it. This is a four-ledger CPU pilot, separate from the earlier hand-authored traces, with two summary instructions and 64 scored method/question combinations. It uses the same pinned model in FP32 with two CPU threads. [Results](results/generated_summary/summary.json), [prompts and generated text](results/generated_summary/cases.json), [costs and limits](results/generated_summary/metadata.json).

| Method | Complete summary: both facts | Closing-only instruction: closing fact | Actually omitted opening fact |
|---|---:|---:|---:|
| Full generated branch retained | 8/8 | 4/4 | 3/3 |
| Same summary text freshly prefilled | 8/8 | 4/4 | 0/3 |
| Retained summary KV, position holes | 8/8 | 3/4 | 0/3 |
| Retained summary KV, rebased | 8/8 | 4/4 | 0/3 |

One of four closing-only summaries disobeyed the instruction and included all transaction facts. That case explains the raw opening-answer score of 1/4 in each compacted method; it is not hidden-fact recovery. All outputs and failures are retained. Every work and summary generation stopped normally. A sufficient generated summary worked in this small pilot; keeping its contextual KV did not make omitted facts reliably retrievable.

Median work generation/build cost was 18.65 seconds and median summary build/generation cost was 4.44 seconds. For complete summaries, median later query preparation was 502 ms for fresh summary text and about 226 ms for retained summary KV; the already-cached full branch was 233 ms. These shared-host CPU timings do not establish an end-to-end throughput gain. Logical retained cache size fell from a median 8.45 MB to 1.70 MB; this concatenating harness does not measure physical reclamation or pointer sharing. The separate chunk-bank tests above measure physical ownership. Broader tasks, longer reasoning, repeated-use amortization, and production Gemma/vLLM compatibility remain untested.

```bash
python generated_summary.py --output results/reproduction_generated_summary --count 4 --threads 2
```

## Relation to existing work and a backend boundary

[CacheBlend](https://arxiv.org/abs/2405.16444) studies position-corrected chunk reuse with selective recomputation to recover the target prefill's quality. [Cache-Craft](https://arxiv.org/abs/2502.15734) studies context-dependent chunk reusability and selective repair. The tests here do not reproduce either system or claim a new position-correction algorithm. Their additional question is whether keeping a donor's contextual influence can help, and whether a contextualized conclusion retains useful information after its earlier reasoning is removed. No claim that training is always required is supported: these interventions are training-free.

An experimental chunk bank should distinguish an exact cache hit from a contextual-memory request. At minimum, a chunk needs a model/tokenizer/precision/layout fingerprint, its token IDs, original absolute positions, layer range, and donor-prefix provenance. The test results favor explicit, sufficient conclusions as the initial useful object. Arbitrary hidden information is not a demonstrated substitute for a written conclusion. These are design implications from this small screen, not validated selection rules.

vLLM's normal [prefix cache](https://docs.vllm.ai/en/latest/design/prefix_caching/) hashes a block together with its parent prefix. Relaxing that key would silently turn an exact cache into a different computation. A deliberate experimental path needs separate chunk handles, block-position metadata, compatible attention kernels, and explicit lifecycle/accounting. `adapter_contract.py` implements validation and CPU view preparation only; it rejects unsupported quantization, scaled RoPE, sliding windows, and soft-capped attention. It does not load or patch vLLM. For a custom quantized cache, dequantization/rotation/requantization may change accuracy, and layout, scales, head shapes, attention masks, and model-specific RoPE must be validated. The Qwen CPU experiments alone establish none of that compatibility.

Develop the backend implementation on a **feature branch or fork first**. A later **narrow upstream PR** can expose proven block-lifetime, import, or attention hooks once tests and a useful workload support them. These are sequential options, not alternatives. The current production vLLM 0.32/custom TurboQuant environment has not been integrated or tested here. See [the adapter boundary](VLLM_ADAPTER.md).

Sharing immutable chunks by pointer can avoid repeated storage and transfer. It does not eliminate the memory for unique chunks or the attention work over active chunks, and it does not make a finite context unlimited. A conclusion can reduce retained KV size because it is shorter; pointer reuse and semantic compression are separate effects.
