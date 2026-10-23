"""Owning, content-addressed KV chunks with shared views and segmented attention.

CPU research reference: standard, already-applied RoPE; no quantized layouts,
sliding windows, softcaps, distributed execution, or vLLM runtime integration.
Bank tensors are private and immutable through the public API. Python itself does
not enforce read-only Tensor storage; callers must not reach into private fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import threading
from typing import Iterable

import torch


@dataclass(frozen=True)
class ModelFingerprint:
    model: str
    revision: str
    tokenizer: str
    dtype: str = "torch.float32"
    layout: str = "B,Hkv,T,D"
    rope: str = "standard-half-split"


@dataclass(frozen=True)
class ChunkMetadata:
    layer: int
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    prefix_digest: str
    provenance: str = ""

    def __post_init__(self):
        # Protect metadata too when a caller supplies lists despite the hints.
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "positions", tuple(self.positions))


@dataclass(frozen=True)
class ChunkHandle:
    digest: str


@dataclass(frozen=True)
class ChunkUse:
    handle: ChunkHandle
    position_delta: int = 0


@dataclass
class _Entry:
    key: torch.Tensor
    value: torch.Tensor
    metadata: ChunkMetadata
    references: int = 0

    @property
    def nbytes(self):
        # Each K and V is an independent owning clone, including input views.
        return self.key.untyped_storage().nbytes() + self.value.untyped_storage().nbytes()


class ChunkInUseError(RuntimeError):
    pass


def _owned_tensor(tensor):
    if tensor.device.type != "cpu":
        raise ValueError("This reference bank supports CPU tensors only")
    if tensor.ndim != 4 or tensor.shape[2] == 0 or not tensor.is_floating_point():
        raise ValueError("Expected a nonempty floating B,Hkv,T,D tensor")
    return tensor.detach().clone(memory_format=torch.contiguous_format)


def rotate(tensor, delta, inv_freq):
    """Standard Qwen/Llama half-split RoPE, used for Q or a dense test oracle."""
    if tensor.shape[-1] != 2 * inv_freq.numel():
        raise ValueError("RoPE frequencies do not match head dimension")
    a, b = tensor.chunk(2, dim=-1)
    angle = torch.cat((inv_freq * delta, inv_freq * delta)).to(tensor)
    return tensor * angle.cos() + torch.cat((-b, a), dim=-1) * angle.sin()


class ChunkBank:
    """Own one physical allocation per content handle; views pin their handles."""
    def __init__(self, fingerprint: ModelFingerprint):
        if fingerprint.layout != "B,Hkv,T,D" or fingerprint.rope != "standard-half-split":
            raise ValueError("Unsupported cache layout or RoPE convention")
        self.fingerprint = fingerprint
        self._entries: dict[ChunkHandle, _Entry] = {}
        self._lock = threading.RLock()

    def put(self, key, value, metadata: ChunkMetadata) -> ChunkHandle:
        if key.shape != value.shape or key.dtype != value.dtype:
            raise ValueError("K/V shape and dtype must agree")
        if str(key.dtype) != self.fingerprint.dtype:
            raise ValueError("Cache dtype differs from model fingerprint")
        if key.shape[2] != len(metadata.token_ids) or len(metadata.positions) != len(metadata.token_ids):
            raise ValueError("Token IDs/positions must describe every cached token")
        if metadata.layer < 0 or any(p < 0 for p in metadata.positions):
            raise ValueError("Layer and stored positions must be nonnegative")
        # Clone once at ingestion: a tiny slice must not retain a huge parent.
        k, v = _owned_tensor(key), _owned_tensor(value)
        header = {"fingerprint": asdict(self.fingerprint), "metadata": asdict(metadata),
                  "shape": list(k.shape)}
        digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
        for item in (k, v):
            digest.update(item.view(torch.uint8).numpy().tobytes())
        handle = ChunkHandle(digest.hexdigest())
        with self._lock:
            if handle not in self._entries:
                self._entries[handle] = _Entry(k, v, metadata)
        return handle

    def metadata(self, handle):
        with self._lock:
            return self._entries[handle].metadata

    def storage_identity(self, handle):
        """Addresses and sizes only, for inspecting sharing without exposing tensors."""
        with self._lock:
            entry = self._entries[handle]
            return (entry.key.data_ptr(), entry.value.data_ptr(), entry.nbytes)

    def stats(self):
        with self._lock:
            entries = list(self._entries.values())
            physical = sum(e.nbytes for e in entries)
            referenced = sum(e.nbytes for e in entries if e.references)
            logical = sum(e.nbytes * e.references for e in entries)
            return {"chunks": len(entries), "unique_physical_bytes": physical,
                    "referenced_unique_bytes": referenced, "logical_reference_bytes": logical,
                    "avoided_duplicate_bytes": logical - referenced,
                    "handle_references": sum(e.references for e in entries)}

    def evict(self, handle):
        with self._lock:
            if self._entries[handle].references:
                raise ChunkInUseError("A live branch still references this chunk")
            del self._entries[handle]

    def view(self, uses: Iterable[ChunkUse] = ()):
        return BranchView(self, uses)

    def _acquire(self, handle):
        with self._lock:
            self._entries[handle].references += 1

    def _release(self, handle):
        with self._lock:
            entry = self._entries[handle]
            if entry.references <= 0:
                raise RuntimeError("Reference accounting underflow")
            entry.references -= 1


class BranchView:
    """Shared immutable handles plus branch-local, append-only writable KV tails.

    A fork shares bank handles and copies any still-unsealed tail blocks. Seal a
    large tail before forking to share it as well. Branch methods are intentionally
    single-owner; bank allocation/reference operations are lock protected.
    """
    def __init__(self, bank, uses=()):
        self.bank = bank
        self._uses: list[ChunkUse] = []
        self._tails: dict[int, list[_Entry]] = {}
        self._closed = False
        try:
            for use in uses:
                self.add(use)
        except Exception:
            self.close()
            raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Branch is closed")

    @property
    def uses(self):
        return tuple(self._uses)

    @property
    def tail_bytes(self):
        return sum(e.nbytes for entries in self._tails.values() for e in entries)

    def add(self, use):
        self._check_open()
        meta = self.bank.metadata(use.handle)
        if any(p + use.position_delta < 0 for p in meta.positions):
            raise ValueError("Shifted key positions cannot be negative")
        self.bank._acquire(use.handle)
        self._uses.append(use)

    def remove(self, handle):
        self._check_open()
        kept, removed = [], 0
        for use in self._uses:
            if use.handle == handle:
                self.bank._release(handle)
                removed += 1
            else:
                kept.append(use)
        self._uses = kept
        return removed

    def fork(self):
        self._check_open()
        other = BranchView(self.bank, self._uses)
        for layer, entries in self._tails.items():
            other._tails[layer] = [_Entry(_owned_tensor(e.key), _owned_tensor(e.value), e.metadata)
                                    for e in entries]
        return other

    def append_tail(self, key, value, metadata):
        self._check_open()
        if key.shape != value.shape or str(key.dtype) != self.bank.fingerprint.dtype or key.dtype != value.dtype:
            raise ValueError("Tail K/V does not match the bank fingerprint")
        if len(metadata.positions) != key.shape[2] or len(metadata.token_ids) != key.shape[2]:
            raise ValueError("Tail metadata length mismatch")
        if metadata.layer < 0 or any(p < 0 for p in metadata.positions):
            raise ValueError("Layer and tail positions must be nonnegative")
        self._tails.setdefault(metadata.layer, []).append(
            _Entry(_owned_tensor(key), _owned_tensor(value), metadata))

    def seal_tail(self):
        self._check_open()
        for entries in self._tails.values():
            for entry in entries:
                handle = self.bank.put(entry.key, entry.value, entry.metadata)
                self.add(ChunkUse(handle))
        self._tails.clear()

    def _blocks(self, layer):
        self._check_open()
        for use in self._uses:
            entry = self.bank._entries[use.handle]
            if entry.metadata.layer == layer:
                yield entry, use.position_delta
        for entry in self._tails.get(layer, []):
            yield entry, 0

    @torch.inference_mode()
    def attend(self, query, *, layer, query_positions, inv_freq=None, causal=True, scale=None):
        """Exactly merge per-chunk softmax sums without concatenating/repeating KV.

        Q is already RoPE-rotated at query_positions and has B,Hq,Q,D shape.
        GQA groups use einsum directly; KV heads are never repeat_interleaved.
        A block shift delta is represented by R(-delta)Q for that block, avoiding
        new persistent copies of rotated K. Positions also drive causal masking.
        """
        self._check_open()
        if query.ndim != 4 or query.device.type != "cpu":
            raise ValueError("Expected CPU query with shape B,Hq,Q,D")
        blocks = list(self._blocks(layer))
        if not blocks:
            return torch.zeros_like(query)
        batch, q_heads, q_length, dim = query.shape
        kv_heads = blocks[0][0].key.shape[1]
        if q_heads % kv_heads:
            raise ValueError("Query head count must be a multiple of KV heads")
        groups = q_heads // kv_heads
        work_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
        q = query.to(work_dtype).reshape(batch, kv_heads, groups, q_length, dim)
        positions = torch.as_tensor(query_positions, device=query.device)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch, -1)
        if positions.shape != (batch, q_length):
            raise ValueError("query_positions must be Q or B,Q")
        maximum = torch.full(q.shape[:-1], -torch.inf, dtype=work_dtype, device=query.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(q)
        scale = dim ** -0.5 if scale is None else scale
        for entry, delta in blocks:
            k, v = entry.key, entry.value
            if k.shape[:2] != (batch, kv_heads) or k.shape[-1] != dim:
                raise ValueError("Incompatible chunk head/batch dimensions")
            if delta and inv_freq is None:
                raise ValueError("Shifted chunks require explicit RoPE frequencies")
            block_q = rotate(q, -delta, inv_freq) if delta else q
            scores = torch.einsum("bhgqd,bhtd->bhgqt", block_q, k.to(work_dtype)) * scale
            if causal:
                key_positions = torch.tensor(entry.metadata.positions, device=query.device) + delta
                allowed = key_positions.view(1, 1, 1, 1, -1) <= positions[:, None, None, :, None]
                scores.masked_fill_(~allowed, -torch.inf)
            block_maximum = scores.amax(-1)
            new_maximum = torch.maximum(maximum, block_maximum)
            safe_maximum = torch.where(torch.isfinite(new_maximum), new_maximum, 0)
            old_scale = torch.where(torch.isfinite(maximum), torch.exp(maximum - safe_maximum), 0)
            weights = torch.exp(scores - safe_maximum.unsqueeze(-1))
            numerator = numerator * old_scale.unsqueeze(-1) + torch.einsum(
                "bhgqt,bhtd->bhgqd", weights, v.to(work_dtype))
            denominator = denominator * old_scale + weights.sum(-1)
            maximum = new_maximum
        result = numerator / denominator.clamp_min(torch.finfo(work_dtype).tiny).unsqueeze(-1)
        return result.reshape_as(query).to(query.dtype)

    def close(self):
        if self._closed:
            return
        for use in self._uses:
            self.bank._release(use.handle)
        self._uses.clear()
        self._tails.clear()
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
