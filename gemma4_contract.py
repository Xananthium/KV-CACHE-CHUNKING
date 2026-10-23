"""CPU-only metadata boundary for a proposed Gemma4/TurboQuant adapter.

No vLLM imports, tensor loading, pointer dereferences, or runtime integration.
A valid descriptor is necessary, not sufficient, for a correct import. The
producer must attest the actual allocations and cache provenance. Every plan
returned here explicitly remains unsupported by the serving runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re


@dataclass(frozen=True)
class LayerABI:
    layer: int
    attention_type: str
    query_heads: int
    kv_heads: int
    head_dim: int
    rope_type: str
    rope_theta: float
    rotary_pairs: int
    scale: float
    sliding_window: int | None
    attention_softcap: float | None
    codec: str
    storage_dtype: str
    packing: str
    bytes_per_head_token: int
    key_norm_semantics: str


def derive_layers(config, *, tensor_parallel_size=2,
                  skip_layers=(0, 1, 58, 59)):
    """Describe the audited text-model ABI; do not guess allocator layouts."""
    c = config.get("text_config", config)
    if c.get("model_type") != "gemma4_text":
        raise ValueError("Expected a Gemma4 text configuration")
    if c.get("dtype", config.get("dtype")) != "bfloat16":
        raise NotImplementedError("This audit describes BF16 model computation")
    if c.get("num_kv_shared_layers", 0):
        raise NotImplementedError("Shared decoder layers need explicit producer mappings")
    kinds = c["layer_types"]
    if len(kinds) != c["num_hidden_layers"]:
        raise ValueError("Layer type count does not match the model")
    if tensor_parallel_size <= 0 or c["num_attention_heads"] % tensor_parallel_size:
        raise ValueError("Invalid tensor-parallel query-head partition")
    if any(i < 0 or i >= len(kinds) for i in skip_layers):
        raise ValueError("Unknown native-cache boundary layer")
    result = []
    for i, kind in enumerate(kinds):
        if kind not in ("sliding_attention", "full_attention"):
            raise NotImplementedError("Unsupported layer type")
        local = kind == "sliding_attention"
        dim = c["head_dim"] if local else c["global_head_dim"]
        heads = c["num_key_value_heads"] if local else c["num_global_key_value_heads"]
        if heads % tensor_parallel_size:
            raise NotImplementedError("Replicated KV-head partitions need a separate ABI")
        rope = c["rope_parameters"][kind]
        if dim <= 0 or dim & (dim - 1) or rope["rope_theta"] <= 0:
            raise ValueError("Invalid Hadamard dimension or rotary base")
        expected_rope = "default" if local else "proportional"
        if rope["rope_type"] != expected_rope:
            raise NotImplementedError("Unexpected Gemma4 rotary implementation")
        pairs = int(dim * rope.get("partial_rotary_factor", 1.0)) // 2
        if not 0 < pairs <= dim // 2 or dim % 2:
            raise ValueError("Invalid rotary dimensions")
        if local and pairs != dim // 2:
            raise NotImplementedError("Partial local RoPE was not audited")
        native = i in skip_layers
        result.append(LayerABI(
            i, kind, c["num_attention_heads"] // tensor_parallel_size,
            heads // tensor_parallel_size, dim, expected_rope,
            float(rope["rope_theta"]), pairs, 1.0,
            c["sliding_window"] if local else None,
            c.get("attn_logit_softcapping"),
            "native_bf16" if native else "turboquant_4bit_nc",
            "bfloat16" if native else "uint8",
            "backend_native_kv" if native else "tq_soa_data_then_head_field_token",
            4 * dim if native else dim + 6,
            "none" if native else "fp16_norm_div_centroid_norm_at_store",
        ))
    return tuple(result)


def rope_inverse_frequencies(layer: LayerABI):
    """Half-split angles, including proportional RoPE's identity pairs.

    Global exponents use the FULL head dimension as denominator. Its active
    pairs occupy the front of each half, not a leading contiguous rotary slice.
    """
    active = tuple(layer.rope_theta ** (-2 * j / layer.head_dim)
                   for j in range(layer.rotary_pairs))
    return active + (0.0,) * (layer.head_dim // 2 - layer.rotary_pairs)


def rotate_vector(values, delta, layer: LayerABI):
    """Double-precision mathematical oracle, not a backend implementation."""
    if len(values) != layer.head_dim:
        raise ValueError("Vector dimension mismatch")
    half = layer.head_dim // 2
    first, second = [], []
    for j, freq in enumerate(rope_inverse_frequencies(layer)):
        cosine, sine = math.cos(delta * freq), math.sin(delta * freq)
        first.append(values[j] * cosine - values[j + half] * sine)
        second.append(values[j + half] * cosine + values[j] * sine)
    return tuple(first + second)


@dataclass(frozen=True)
class ProducerIdentity:
    # Digests must describe actual weights/tokenizer/codec implementations.
    # A model alias or a hash of config.json alone is not a weights digest.
    weights_sha256: str
    model_config_sha256: str
    tokenizer_sha256: str
    implementation_sha256: str
    engine_instance: str
    tensor_parallel_size: int
    rank: int


@dataclass(frozen=True)
class LayerChunk:
    identity: ProducerIdentity
    abi: LayerABI
    token_ids_sha256: str
    donor_prefix_sha256: str
    source_start: int
    num_tokens: int
    kv_group_id: int
    block_size: int
    block_ids: tuple[int, ...]
    allocation_epochs: tuple[int, ...]
    block_stride_bytes: int
    resident: bool
    producer_write_complete: bool
    accepted_tokens_only: bool
    immutable: bool
    owner_lease_held: bool

    def __post_init__(self):
        object.__setattr__(self, "block_ids", tuple(self.block_ids))
        object.__setattr__(self, "allocation_epochs", tuple(self.allocation_epochs))


@dataclass(frozen=True)
class ImportPlan:
    mode: str
    position_delta: int
    logical_tokens: int
    required_extensions: tuple[str, ...]
    metadata_valid: bool = True
    runtime_execution_supported: bool = False


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def _natural(value, *, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


def validate_bundle(chunks, layers, expected_identity, *, target_start,
                    query_position, target_prefix_sha256, mode,
                    serving_position_limit=200000, speculative_tokens=2,
                    modality="text"):
    """Check one all-layer, all-rank conclusion span for a future import.

    Initial scope uses whole immutable physical blocks. Partial-block views,
    multimodal spans, and missing layers are rejected rather than treating
    uncomputed or evicted K/V as valid. Returned hooks are a work list, not
    capabilities present in the existing engine.
    """
    if mode not in ("exact_prefix", "contextual_chunk"):
        raise ValueError("Reuse semantics must be explicit")
    if modality != "text":
        raise NotImplementedError("Vision bidirectional spans need separate masks and provenance")
    if not _sha(target_prefix_sha256):
        raise ValueError("Target prefix requires a digest")
    if not expected_identity.engine_instance:
        raise ValueError("Borrowed allocations require a live engine identity")
    if any(not _sha(v) for v in (expected_identity.weights_sha256,
                                expected_identity.model_config_sha256,
                                expected_identity.tokenizer_sha256,
                                expected_identity.implementation_sha256)):
        raise ValueError("Weights, config, tokenizer, and implementation require SHA256 identities")
    if (not _natural(target_start) or not _natural(query_position)
            or not _natural(serving_position_limit, positive=True)
            or query_position >= serving_position_limit):
        raise ValueError("Logical positions exceed the serving contract")
    if not _natural(speculative_tokens):
        raise ValueError("Invalid speculative token count")
    chunks = tuple(chunks)
    required = {(layer.layer, rank) for layer in layers
                for rank in range(expected_identity.tensor_parallel_size)}
    if len(chunks) != len(required) or {(c.abi.layer, c.identity.rank) for c in chunks} != required:
        raise ValueError("Need every layer and tensor-parallel shard exactly once")
    span = {(c.source_start, c.num_tokens, c.token_ids_sha256, c.donor_prefix_sha256)
            for c in chunks}
    if len(span) != 1:
        raise ValueError("Layer shards disagree on token span or donor provenance")
    source_start, count, token_hash, prefix_hash = span.pop()
    if (not _natural(source_start) or not _natural(count, positive=True)
            or not _sha(token_hash) or not _sha(prefix_hash)):
        raise ValueError("Invalid source token span or provenance")
    if source_start + count > serving_position_limit or target_start + count > query_position:
        raise ValueError("Source span or causal target placement is out of bounds")
    by_layer = {layer.layer: layer for layer in layers}
    for c in chunks:
        if c.identity != replace(expected_identity, rank=c.identity.rank):
            raise ValueError("Different weights/tokenizer/code/engine/parallel partition")
        if c.abi != by_layer[c.abi.layer]:
            raise ValueError("Layer codec, layout, attention, or rotary ABI mismatch")
        if c.abi.attention_softcap:
            raise NotImplementedError("Audited unified TQ does not implement attention softcap")
        if not _natural(c.kv_group_id) or not _natural(c.block_size, positive=True):
            raise ValueError("Physical cache group/block size must come from the allocator")
        if source_start % c.block_size or target_start % c.block_size or count % c.block_size:
            raise NotImplementedError("Partial blocks require read ranges and private append/COW handling")
        if len(c.block_ids) != count // c.block_size or len(c.allocation_epochs) != len(c.block_ids):
            raise ValueError("Physical blocks do not cover the described token span")
        if len(set(c.block_ids)) != len(c.block_ids) or any(not _natural(b) for b in c.block_ids):
            raise ValueError("Repeated or invalid physical block IDs")
        if any(not _natural(epoch) for epoch in c.allocation_epochs):
            raise ValueError("Allocation generations are required for stale-handle detection")
        minimum = c.block_size * c.abi.kv_heads * c.abi.bytes_per_head_token
        if (not _natural(c.block_stride_bytes, positive=True)
                or c.block_stride_bytes < minimum or c.block_stride_bytes % 2):
            raise ValueError("Block stride cannot hold the packed payload and metadata")
        if not all(flag is True for flag in (
                c.resident, c.producer_write_complete, c.accepted_tokens_only,
                c.immutable, c.owner_lease_held)):
            raise ValueError("Missing, unfinished, writable, or unpinned blocks cannot be borrowed")
    delta = target_start - source_start
    if mode == "exact_prefix" and (delta or target_prefix_sha256 != prefix_hash):
        raise ValueError("Changed position or prefix is contextual reuse, not an exact cache hit")
    hooks = ["allocator-owned immutable leases with allocation generations",
             "publish/attach/release synchronization on every rank and cache group",
             "private decode tails and in-flight reader references"]
    if mode == "contextual_chunk":
        hooks += ["contextual namespace separate from prefix-cache hashes",
                  "logical positions separate from token/slot/block-table offsets",
                  "causal and per-layer sliding masks over logical positions",
                  "request token IDs, penalties, and usage after history surgery"]
    if delta:
        hooks.append("segment-specific query RoPE shift BEFORE Hadamard transform")
    if speculative_tokens:
        hooks.append("MTP target/draft mapping, accepted-boundary snapshot, and rollback")
    return ImportPlan(mode, delta, count, tuple(hooks))
