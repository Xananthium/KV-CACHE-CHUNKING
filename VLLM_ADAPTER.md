# Backend adapter boundary

The CPU `adapter_contract.py` and `chunk_bank.py` are working reference pieces. They are not a loaded vLLM plugin. The production vLLM 0.32/custom TurboQuant setup remains untouched and unverified by this repository.

An experiment on a feature branch should use a separate namespace for contextual chunks. Preserve ordinary prefix-cache keys and their exact parent-prefix requirement. A contextual chunk import must identify weights, tokenizer, layer, head layout, dtype/codec, original positions, token IDs, and donor-prefix provenance. Do not count a contextual import as an exact prefix hit.

The initial backend API can have five operations:

1. Publish an owned layer block and its immutable metadata; return a content handle.
2. Attach an ordered handle list and per-block position shifts to a branch. Pin every physical block before exposing it to a batch.
3. Fork the branch's block table. Shared blocks stay immutable; decode appends to private writable blocks.
4. Remove or replace selected block references only when no in-flight kernel can read the old table. Reference counts include scheduled/in-flight work.
5. Release branch references and evict only unpinned blocks.

The scheduler must separately account for physical shared bytes, per-branch logical tokens, private tails, and attention work. Query absolute positions may differ from compact physical slot indices after deletion. Both must reach the kernel. The CPU reference's query rotation per shifted block avoids stored copies of rotated keys, but a production kernel must implement the same dot product under its actual RoPE and codec conventions.

Start with unquantized standard-RoPE attention and one model whose full logits can be compared. Then validate any model-specific softcaps, local/sliding attention, rotary scaling, positional limits, head dimensions, and padding behavior. A custom TurboQuant block cannot be treated as a float tensor: packing, scales, quantization transforms, allocator ownership, and quantized dot products require an explicit ABI. This reference deliberately refuses that configuration. No result here demonstrates compatibility with the production Gemma4-31B model or cache format.

The acceptance ladder is: dense-versus-segmented attention on real model tensors; all-layer logits and greedy decode parity for an unchanged logical cache; physical storage sharing across branches; safe cancellation/eviction under concurrent batches; then separate quality experiments for approximate contextual reuse and summary compaction. Measure construction, import, prefill, decode, bytes, and throughput separately. The current CPU prototype clears the first three on a small case, plus single-process lifecycle tests; it does not clear concurrent serving or quality generalization.

Use a feature branch/fork for the complete experimental path. Once a small hook has clear semantics and maintained tests, propose that piece as an upstream PR. The broader semantic policy for contextual memory may remain outside the core engine. No upstream acceptance or production readiness is assumed.
