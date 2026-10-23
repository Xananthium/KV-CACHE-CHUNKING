#!/usr/bin/env python3
"""Compare actual Qwen attention tensors with an independent dense SDPA oracle."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from chunk_bank import ChunkBank, ChunkMetadata, ChunkUse, ModelFingerprint, rotate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", nargs="?", type=Path,
                        default=Path("results/bank_verification/qwen_attention_fixture.pt"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = torch.load(args.fixture, map_location="cpu", weights_only=True)
    case, inv = data["case"], data["inv_freq"]
    p, r, c, q_ids = [case["tokens"][x] for x in "prcq"]
    results = []
    for layer in [0, 12, 23]:
        for mode in ["holes", "rebase"]:
            bank = ChunkBank(ModelFingerprint("Qwen", data["model_revision"], data["model_revision"]))
            delta = 0 if mode == "holes" else -len(r)
            pk, pv = data[f"layer_{layer}_prefix_key"], data[f"layer_{layer}_prefix_value"]
            ck, cv = data[f"layer_{layer}_conclusion_key"], data[f"layer_{layer}_conclusion_value"]
            q = rotate(data[f"layer_{layer}_query"], delta, inv)
            qk = rotate(data[f"layer_{layer}_query_key"], delta, inv)
            qv = data[f"layer_{layer}_query_value"]
            q_start = len(p) + len(r) + len(c) + delta
            q_positions = torch.arange(q_start, q_start + len(q_ids))
            ph = bank.put(pk, pv, ChunkMetadata(layer, tuple(p), tuple(range(len(p))), "fixture"))
            ch = bank.put(ck, cv, ChunkMetadata(layer, tuple(c), tuple(range(len(p) + len(r), len(p) + len(r) + len(c))), "fixture"))
            with bank.view([ChunkUse(ph), ChunkUse(ch, delta)]) as view:
                view.append_tail(qk, qv, ChunkMetadata(layer, tuple(q_ids), tuple(q_positions.tolist()), "fixture"))
                actual = view.attend(q, layer=layer, query_positions=q_positions, inv_freq=inv)
                # Only the oracle concatenates KV and repeats GQA heads.
                key = torch.cat([pk, rotate(ck, delta, inv), qk], dim=2).repeat_interleave(7, dim=1)
                value = torch.cat([pv, cv, qv], dim=2).repeat_interleave(7, dim=1)
                key_positions = torch.tensor(list(range(len(p))) +
                                             list(range(len(p) + len(r) + delta, q_start)) +
                                             q_positions.tolist())
                mask = key_positions[None, :] <= q_positions[:, None]
                expected = F.scaled_dot_product_attention(q, key, value, attn_mask=mask)
                error = float((actual - expected).abs().max())
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
                results.append({"layer": layer, "mode": mode, "max_attention_error": error})
    print(json.dumps({"actual_qwen_tensor_checks": results, "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
