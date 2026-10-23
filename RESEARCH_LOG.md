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
