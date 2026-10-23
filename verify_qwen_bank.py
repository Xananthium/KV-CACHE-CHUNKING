#!/usr/bin/env python3
"""Actual pretrained Qwen inference through the segmented cache-bank reference.

This narrow forward adapter exists to verify the prototype using real weights,
query projections, conclusion KV, logits, and greedy outputs. It is not a vLLM
plugin or general Transformers replacement. No training or production API calls.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from branch_compaction import make_cases
from chunk_bank import ChunkBank, ChunkMetadata, ChunkUse, ModelFingerprint
from run_experiment import MODEL_ID, REVISION, Runner, cat_cache, prefix, rebase_cache, slice_cache, tensor


@torch.inference_mode()
def bank_forward(runner, view, token_ids, position_start, fixtures=None):
    """Same Qwen decoder layers; attention reads shared blocks and local tails."""
    model = runner.model.model
    hidden = model.embed_tokens(tensor(token_ids))
    positions = torch.arange(position_start, position_start + len(token_ids)).view(1, -1)
    cos, sin = model.rotary_emb(hidden, positions)
    batch, length, _ = hidden.shape
    for layer_idx, layer in enumerate(model.layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attention = layer.self_attn
        q = attention.q_proj(normed).view(batch, length, attention.num_heads, attention.head_dim).transpose(1, 2)
        k = attention.k_proj(normed).view(batch, length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
        v = attention.v_proj(normed).view(batch, length, attention.num_key_value_heads, attention.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        view.append_tail(k, v, ChunkMetadata(layer_idx, tuple(token_ids), tuple(positions[0].tolist()),
                                             "private-query-tail", "Qwen verification"))
        out = view.attend(q, layer=layer_idx, query_positions=positions, inv_freq=runner.inv_freq)
        if fixtures is not None and layer_idx in [0, 12, 23]:
            fixtures[f"layer_{layer_idx}_query"] = q.clone()
            fixtures[f"layer_{layer_idx}_query_key"] = k.clone()
            fixtures[f"layer_{layer_idx}_query_value"] = v.clone()
            fixtures[f"layer_{layer_idx}_attention_output"] = out.clone()
        attention_output = attention.o_proj(out.transpose(1, 2).contiguous().view(batch, length, -1))
        hidden = residual + attention_output
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return runner.model.lm_head(model.norm(hidden)[:, -1, :])[0].float()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/bank_verification"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    runner = Runner(args)
    case = make_cases(runner.tokenizer, 1, args.seed, "chat_branch")[0]
    p, r, c, q = [case["tokens"][key] for key in "prcq"]
    full, _ = runner.forward(p + r + c)
    pc = prefix(full, len(p))
    cc = slice_cache(full, len(p) + len(r), len(p) + len(r) + len(c))
    bank = ChunkBank(ModelFingerprint(MODEL_ID, REVISION, REVISION))
    p_handles, c_handles = [], []
    pdigest = hashlib.sha256(json.dumps([]).encode()).hexdigest()
    rdigest = hashlib.sha256(json.dumps(p + r).encode()).hexdigest()
    for layer, ((pk, pv), (ck, cv)) in enumerate(zip(pc, cc)):
        p_handles.append(bank.put(pk, pv, ChunkMetadata(layer, tuple(p), tuple(range(len(p))), pdigest, "shared system/request prefix")))
        c_handles.append(bank.put(ck, cv, ChunkMetadata(layer, tuple(c), tuple(range(len(p) + len(r), len(p) + len(r) + len(c))), rdigest, "contextualized conclusion")))
    root = bank.view([ChunkUse(h) for h in p_handles + c_handles])
    forks = [root.fork() for _ in range(4)]
    sharing = bank.stats()
    assert sharing["logical_reference_bytes"] == 5 * sharing["unique_physical_bytes"]
    for view in forks:
        view.close()
    root.close()
    results, fixtures = [], {"case": case, "inv_freq": runner.inv_freq.clone(), "model_revision": REVISION}
    for mode in ["holes", "rebase"]:
        delta = 0 if mode == "holes" else -len(r)
        uses = [ChunkUse(h) for h in p_handles] + [ChunkUse(h, delta) for h in c_handles]
        dense = cat_cache(pc, cc if mode == "holes" else rebase_cache(cc, delta, runner.inv_freq))
        position = len(p) + len(c) + (len(r) if mode == "holes" else 0)
        dense_cache, dense_logits = runner.forward(q, dense, position_start=position)
        with bank.view(uses) as view:
            t = time.perf_counter()
            model_logits = bank_forward(runner, view, q, position,
                                        fixtures if mode == "holes" else None)
            bank_prefill_ms = 1000 * (time.perf_counter() - t)
            max_error = float((model_logits - dense_logits).abs().max())
            assert max_error < 5e-4, (mode, max_error)
            ids_out, token_errors = [], [max_error]
            position += len(q)
            for _ in range(args.max_new_tokens):
                chosen = int(dense_logits.argmax())
                assert chosen == int(model_logits.argmax()), (mode, ids_out)
                if chosen in runner.eos:
                    break
                ids_out.append(chosen)
                dense_cache, dense_logits = runner.forward([chosen], dense_cache, position_start=position)
                model_logits = bank_forward(runner, view, [chosen], position)
                error = float((model_logits - dense_logits).abs().max())
                token_errors.append(error)
                assert error < 5e-4, (mode, error)
                position += 1
            answer = runner.tokenizer.decode(ids_out, skip_special_tokens=True)
            assert answer.strip() == case["gold"], (mode, answer, case["gold"])
            results.append({"mode": mode, "answer": answer, "gold": case["gold"],
                            "max_logit_error": max(token_errors), "per_step_logit_error": token_errors,
                            "bank_prefill_ms": bank_prefill_ms, "private_tail_bytes": view.tail_bytes,
                            "bank_unique_bytes": bank.stats()["unique_physical_bytes"]})
    # Small actual-model attention fixture for checking the bank without model loading.
    for layer in [0, 12, 23]:
        fixtures[f"layer_{layer}_prefix_key"] = pc[layer][0].clone()
        fixtures[f"layer_{layer}_prefix_value"] = pc[layer][1].clone()
        fixtures[f"layer_{layer}_conclusion_key"] = cc[layer][0].clone()
        fixtures[f"layer_{layer}_conclusion_value"] = cc[layer][1].clone()
    torch.save(fixtures, args.output / "qwen_attention_fixture.pt")
    stats_before_evict = bank.stats()
    assert stats_before_evict["handle_references"] == 0
    for handle in p_handles + c_handles:
        bank.evict(handle)
    assert bank.stats()["unique_physical_bytes"] == 0
    report = {"model": MODEL_ID, "revision": REVISION, "case_id": case["case_id"],
              "dtype": "float32", "threads": args.threads, "sharing_five_views": sharing,
              "results": results, "cleanup": bank.stats(),
              "elapsed_seconds": time.perf_counter() - start,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "bank_sha256": hashlib.sha256(Path(__file__).with_name("chunk_bank.py").read_bytes()).hexdigest()}
    for name in ["verify_qwen_bank.py", "chunk_bank.py", "run_experiment.py", "branch_compaction.py"]:
        (args.output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
