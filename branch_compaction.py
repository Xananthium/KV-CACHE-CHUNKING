#!/usr/bin/env python3
"""Delete a middle reasoning segment without rebuilding the later conclusion KV.

Compare complete branch, freshly prefilled textual conclusion, and surgical KV
splicing. Original-position holes use compact physical cache slots/causal indices
and explicit original absolute RoPE positions; the query and decode retain those
absolute positions. This is a controlled CPU experiment, not a serving backend.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import resource
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers

from run_experiment import (MODEL_ID, REVISION, Runner, cache_bytes, cat_cache,
                            ids, layer_errors, prefix, rebase_cache, slice_cache)


def make_cases(tokenizer, count, seed):
    rng = random.Random(seed)
    cases = []
    names = ["Cedar", "Birch", "Maple", "Willow", "Oak", "Pine"]
    for index in range(count):
        name = names[index % len(names)]
        opening = rng.randint(110, 700)
        deposit = rng.randint(11, 99)
        withdrawal = rng.randint(11, 99)
        intermediate, closing = opening + deposit, opening + deposit - withdrawal
        shared = (
            "<|im_start|>system\nYou are a careful archive assistant. "
            "Answer the final question using the available record. "
            "Reply with the requested number only, with no explanation.<|im_end|>\n"
            f"<|im_start|>user\nWe are checking the {name} account. "
            "The work below is an archived reasoning branch.\n"
        )
        reasoning = (
            "<reasoning>\n"
            f"The {name} account's opening balance is {opening} credits. "
            f"A deposit adds {deposit} credits: {opening} + {deposit} = {intermediate}. "
            f"A withdrawal removes {withdrawal} credits: {intermediate} - {withdrawal} = {closing}. "
            f"Therefore the closing balance is {closing} credits. "
            f"Recheck: {closing} + {withdrawal} - {deposit} = {opening}. "
            "All arithmetic is consistent.\n</reasoning>\n"
        )
        for kind in ["complete", "opaque", "lossy_detail"]:
            if kind == "opaque":
                conclusion = (f"<conclusion>\nThe {name} account has been reconciled. "
                              "Its closing balance is confirmed.\n</conclusion>\n")
            else:
                conclusion = (f"<conclusion>\nThe {name} account's closing balance is "
                              f"{closing} credits.\n</conclusion>\n")
            if kind == "lossy_detail":
                gold = str(opening)
                question = f"What was the opening balance of the {name} account, before the deposit and withdrawal?"
            else:
                gold = str(closing)
                question = f"What is the closing balance of the {name} account?"
            query = (f"\nQuestion: {question}\nReply with the number only."
                     "<|im_end|>\n<|im_start|>assistant\n")
            cases.append({"case_id": f"branch_{index:02d}_{kind}", "kind": kind,
                          "shared_prefix": shared, "reasoning": reasoning,
                          "conclusion": conclusion, "query": query, "gold": gold,
                          "opening": opening, "deposit": deposit, "withdrawal": withdrawal,
                          "closing": closing,
                          "tokens": {"p": ids(tokenizer, shared), "r": ids(tokenizer, reasoning),
                                     "c": ids(tokenizer, conclusion), "q": ids(tokenizer, query)}})
    return cases


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["kind"], row["method"])].append(row)
    summary = []
    for (kind, method), group in groups.items():
        item = {"kind": kind, "method": method, "n": len(group),
                "exact_answers": sum(x["exact_answer"] for x in group),
                "exact_accuracy": statistics.mean(x["exact_answer"] for x in group),
                "contains_gold_rate": statistics.mean(x["contains_gold"] for x in group),
                "token_limit_rate": statistics.mean(x["hit_token_limit"] for x in group)}
        for field in ["gold_nll_per_token", "next_token_kl_from_full", "online_prefill_ms",
                      "assembly_ms", "decode_ms", "cache_bytes", "removed_tokens",
                      "retained_tokens", "repeated_bigram_fraction"]:
            item[field + "_mean"] = statistics.mean(x[field] for x in group)
            item[field + "_median"] = statistics.median(x[field] for x in group)
        summary.append(item)
    return summary


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8, help="Independent arithmetic records; each has three conclusion conditions")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runner = Runner(args)
    cases = make_cases(runner.tokenizer, args.count, args.seed)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (args.output / "branch_compaction.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "run_experiment.py").write_bytes(Path(__file__).with_name("run_experiment.py").read_bytes())
    metadata = {"model_id": MODEL_ID, "revision": REVISION, "torch": torch.__version__,
                "transformers": transformers.__version__, "python": platform.python_version(),
                "device": "cpu", "dtype": "float32", "threads": args.threads, "seed": args.seed,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "reasoning_source": "Deterministic, programmatically verified arithmetic traces; not model-generated reasoning.",
                "position_semantics": "Physical cache positions are compact in both splice modes. splice_holes retains original C key rotations, explicitly places Q and every decoded token at their original branch absolute RoPE positions. splice_rebase rotates C keys by -len(R), leaves values untouched, and uses contiguous Q/decode positions.",
                "timing_semantics": "Branch construction is offline. splice online prefill includes tensor assembly plus query; fresh_summary computes P+C+Q from scratch. exact_full_cache uses pre-existing P+R+C. No decode-time speedup guarantee."}
    runner.forward(ids(runner.tokenizer, "Warmup: hello."))
    rows, diagnostics = [], []
    with (args.output / "rows.jsonl").open("w") as log:
        for case in cases:
            p, r, c, q = [case["tokens"][key] for key in "prcq"]
            full_tokens = p + r + c + q
            t = time.perf_counter()
            full_cache, full_logits = runner.forward(full_tokens)
            full_ms = 1000 * (time.perf_counter() - t)
            branch_prefix = prefix(full_cache, len(p) + len(r) + len(c))
            shared_cache = prefix(full_cache, len(p))
            conclusion_cache = slice_cache(full_cache, len(p) + len(r), len(p) + len(r) + len(c))
            # Re-prefilled summary is an independent control: same words, different hidden states.
            t = time.perf_counter()
            fresh_cache, fresh_logits = runner.forward(p + c + q)
            fresh_ms = 1000 * (time.perf_counter() - t)
            fresh_conclusion = slice_cache(fresh_cache, len(p), len(p) + len(c))
            shifted_conclusion = rebase_cache(conclusion_cache, -len(r), runner.inv_freq)
            errors = layer_errors(fresh_conclusion, shifted_conclusion)
            assert errors[0]["key_max_absolute"] < 2e-4, errors[0]
            assert errors[0]["value_max_absolute"] < 2e-5, errors[0]
            diagnostics.append({"case_id": case["case_id"], "layer_errors": errors})
            for method in ["full_branch", "exact_full_cache", "fresh_summary", "splice_holes", "splice_rebase"]:
                assembly_ms, next_position = 0.0, None
                if method == "full_branch":
                    cache, logits, prefill_ms = full_cache, full_logits, full_ms
                    retained, removed = len(p) + len(r) + len(c), 0
                    retained_cache = branch_prefix
                elif method == "fresh_summary":
                    cache, logits, prefill_ms = fresh_cache, fresh_logits, fresh_ms
                    retained, removed = len(p) + len(c), len(r)
                    retained_cache = prefix(fresh_cache, retained)
                else:
                    t = time.perf_counter()
                    if method == "exact_full_cache":
                        retained_cache = branch_prefix
                        retained, removed = len(p) + len(r) + len(c), 0
                    elif method == "splice_holes":
                        retained_cache = cat_cache(shared_cache, conclusion_cache)
                        retained, removed = len(p) + len(c), len(r)
                        next_position = len(full_tokens)
                    else:
                        corrected = rebase_cache(conclusion_cache, -len(r), runner.inv_freq)
                        retained_cache = cat_cache(shared_cache, corrected)
                        retained, removed = len(p) + len(c), len(r)
                    assembly_ms = 1000 * (time.perf_counter() - t)
                    position_start = len(p) + len(r) + len(c) if method == "splice_holes" else None
                    cache, logits = runner.forward(q, retained_cache, position_start=position_start)
                    prefill_ms = 1000 * (time.perf_counter() - t)
                    if method == "exact_full_cache":
                        assert float((logits - full_logits).abs().max()) < 3e-4
                result = runner.score_and_decode(cache, logits, case["gold"], next_position=next_position)
                lp, base_lp = logits.log_softmax(-1), full_logits.log_softmax(-1)
                row = {"case_id": case["case_id"], "kind": case["kind"], "method": method,
                       "gold": case["gold"], "prefix_tokens": len(p), "reasoning_tokens": len(r),
                       "conclusion_tokens": len(c), "query_tokens": len(q),
                       "removed_tokens": removed, "retained_tokens": retained,
                       "cache_bytes": cache_bytes(retained_cache),
                       "branch_prefill_ms": full_ms,
                       "online_prefill_ms": prefill_ms, "assembly_ms": assembly_ms,
                       "next_token_kl_from_full": max(0.0, float((base_lp.exp() * (base_lp - lp)).sum())),
                       "max_abs_logit_diff": float((full_logits - logits).abs().max()),
                       "next_absolute_position": next_position if next_position is not None else retained + len(q),
                       "physical_cache_tokens_after_query": cache[0][0].shape[2],
                       **result}
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps({k: row[k] for k in ["case_id", "method", "gold", "answer", "exact_answer", "gold_nll_per_token"]}), flush=True)
            (args.output / "summary.json").write_text(json.dumps(summarize(rows), indent=2) + "\n")
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    with (args.output / "rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata.update({"n_scored_rows": len(rows), "elapsed_seconds": time.perf_counter() - started,
                     "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"summary": summarize(rows), "metadata": metadata}), flush=True)


if __name__ == "__main__":
    main()
