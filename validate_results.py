#!/usr/bin/env python3
"""Audit recorded results and position/cache invariants without loading a model."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def validate_run(path):
    metadata = json.loads((path / "metadata.json").read_text())
    rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
    if (path / "generated_summary.py").exists():
        assert len(rows) == metadata["rows"], path
        assert hashlib.sha256((path / "generated_summary.py").read_bytes()).hexdigest() == metadata["script_sha256"]
        cases = {(c["case_id"], c["kind"]): c for c in json.loads((path / "cases.json").read_text())}
        seen = set()
        for row in rows:
            key = (row["case_id"], row["summary_kind"], row["question_kind"], row["method"])
            assert key not in seen, key
            seen.add(key)
            case = cases[key[:2]]
            assert case["opening"] + case["deposit"] - case["withdrawal"] == case["closing"]
            assert row["gold"] == str(case[row["question_kind"]]), key
            assert row["exact_answer"] == (row["answer"].strip() == row["gold"]), key
            for field in ["gold_nll_per_token", "online_assembly_and_prefill_ms", "decode_ms", "work_build_ms", "summary_build_ms"]:
                assert math.isfinite(row[field]) and row[field] >= 0, (key, field)
            tokens = len(case["tokens"]["p"]) + len(case["tokens"]["c"])
            if row["method"] == "full_generated_branch":
                tokens += len(case["tokens"]["r"])
            assert row["logical_retained_cache_bytes"] == 24576 * tokens, key
        return {"run": path.name, "rows": len(rows), "status": "PASS"}
    assert len(rows) == metadata["n_scored_rows"], path
    branch = "kind" in rows[0]
    source = path / ("branch_compaction.py" if branch else "run_experiment.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["script_sha256"], path
    seen = set()
    baseline = {}
    for row in rows:
        key = (row["case_id"], row.get("context", ""), row["method"])
        assert key not in seen, key
        seen.add(key)
        assert row["exact_answer"] == (row["answer"].strip() == row["gold"]), key
        for field in ["gold_nll_per_token", "online_prefill_ms", "decode_ms", "next_token_kl_from_full"]:
            assert math.isfinite(row[field]) and row[field] >= 0, (key, field)
        if row["method"] in ["full", "full_branch"]:
            baseline[row["case_id"]] = row
        if row["method"] in ["exact", "exact_full_cache"]:
            assert row["max_abs_logit_diff"] < 3e-4, key
            assert row["answer"] == baseline[row["case_id"]]["answer"], key
        if branch:
            assert row["cache_bytes"] == 24576 * row["retained_tokens"], key
            expected = row["retained_tokens"] + row["query_tokens"]
            assert row["physical_cache_tokens_after_query"] == expected, key
            if row["method"] == "splice_holes":
                expected += row["removed_tokens"]
            assert row["next_absolute_position"] == expected, key
    diagnostics = json.loads((path / "diagnostics.json").read_text())
    for item in diagnostics:
        errors = item["layer_errors" if branch else "layer_errors_after_rope_rebase"]
        assert errors[0]["key_max_absolute"] < 2e-4
        assert errors[0]["value_max_absolute"] < 2e-5
    if branch:
        for case in json.loads((path / "cases.json").read_text()):
            assert case["opening"] + case["deposit"] - case["withdrawal"] == case["closing"]
    return {"run": path.name, "rows": len(rows), "status": "PASS"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    paths = sorted(p.parent for p in args.path.glob("*/metadata.json"))
    assert paths, args.path
    results = [validate_run(path) for path in paths]
    print(json.dumps({"runs": results, "total_rows": sum(x["rows"] for x in results)}, indent=2))


if __name__ == "__main__":
    main()
