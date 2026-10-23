"""Validated import-plan boundary for a future inference-backend adapter.

This module does not import, patch, launch, or claim integration with vLLM.
Its reference implementation can prepare a CPU ChunkBank view. A production
adapter must implement block ownership, scheduler accounting and attention.
"""
from dataclasses import dataclass

from chunk_bank import ChunkBank, ChunkHandle, ChunkUse, ModelFingerprint


@dataclass(frozen=True)
class CacheABI:
    fingerprint: ModelFingerprint
    quantization: str = "none"
    rope_scaling: str = "none"
    sliding_window: int | None = None
    logits_soft_cap: float | None = None


@dataclass(frozen=True)
class Placement:
    handle: ChunkHandle
    target_start: int
    target_prefix_digest: str


def prepare_reference_view(bank: ChunkBank, abi: CacheABI, placements, *, mode: str):
    """Explicitly distinguish exact reuse from contextual transplantation.

    Prefix digests are caller-supplied provenance, not a proof that an imported
    tensor was calculated correctly. Exactness also requires a trusted producer.
    """
    if abi.fingerprint != bank.fingerprint:
        raise ValueError("Model/tokenizer/dtype/layout fingerprint mismatch")
    if abi.quantization != "none" or abi.rope_scaling != "none":
        raise NotImplementedError("Quantized caches and scaled RoPE are not supported by this reference")
    if abi.sliding_window is not None or abi.logits_soft_cap is not None:
        raise NotImplementedError("Windowed or soft-capped attention needs a model-specific adapter")
    if mode not in ["exact_prefix", "contextual_chunk"]:
        raise ValueError("Choose exact_prefix or contextual_chunk explicitly")
    uses, ends = [], {}
    for placement in placements:
        meta = bank.metadata(placement.handle)
        if meta.positions != tuple(range(meta.positions[0], meta.positions[0] + len(meta.positions))):
            raise ValueError("This adapter contract accepts contiguous chunks only")
        if placement.target_start < ends.get(meta.layer, 0):
            raise ValueError("Chunk positions must be ordered and nonoverlapping within each layer")
        delta = placement.target_start - meta.positions[0]
        if mode == "exact_prefix" and (delta or placement.target_prefix_digest != meta.prefix_digest):
            raise ValueError("Changed prefix or position is a contextual request, not an exact cache hit")
        uses.append(ChunkUse(placement.handle, delta))
        ends[meta.layer] = placement.target_start + len(meta.positions)
    return bank.view(uses)
