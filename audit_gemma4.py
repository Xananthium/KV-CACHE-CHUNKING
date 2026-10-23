"""Collect a portable Gemma4 adapter audit without importing vLLM or using CUDA."""
import argparse
import ast
from dataclasses import asdict
import datetime
import hashlib
import json
from pathlib import Path
import re

from gemma4_contract import derive_layers


SOURCES = (
    "model_executor/models/gemma4.py",
    "model_executor/models/gemma4_mtp.py",
    "model_executor/layers/rotary_embedding/gemma4_rope.py",
    "model_executor/layers/attention/attention.py",
    "model_executor/layers/quantization/turboquant/config.py",
    "model_executor/layers/quantization/turboquant/centroids.py",
    "engine/arg_utils.py",
    "v1/kv_cache_interface.py",
    "v1/attention/backends/turboquant_attn.py",
    "v1/attention/ops/turboquant_soa/triton_turboquant_store.py",
    "v1/attention/ops/turboquant_soa/triton_turboquant_unified_attention.py",
    "v1/core/block_pool.py",
    "v1/core/kv_cache_manager.py",
    "v1/core/single_type_kv_cache_manager.py",
    "v1/core/kv_cache_utils.py",
    "v1/worker/block_table.py",
    "v1/worker/gpu_model_runner.py",
    "v1/spec_decode/gemma4.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--runtime-snapshot", type=Path, required=True)
    parser.add_argument("--captured-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_raw = args.model_config.read_bytes()
    config = json.loads(config_raw)
    tokenizer = args.model_config.with_name("tokenizer.json")
    runtime_raw = args.runtime_snapshot.read_bytes()
    runtime = json.loads(runtime_raw)
    argv = runtime["args"]
    option = lambda key, fallback: argv[argv.index(key) + 1] if key in argv else fallback
    metric_text = args.captured_metrics.read_text()
    cache_metric = next(line for line in metric_text.splitlines()
                        if line.startswith("vllm:cache_config_info{"))
    labels = {key: json.loads('"' + value + '"') for key, value in
              re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', cache_metric)}
    skip = tuple(int(v) for v in ast.literal_eval(labels["kv_cache_dtype_skip_layers"]))
    tp = int(option("--tensor-parallel-size", "1"))
    layers = derive_layers(config, tensor_parallel_size=tp, skip_layers=skip)
    speculative = json.loads(option("--speculative-config", "{}"))
    source_records = {}
    for relative in SOURCES:
        raw = (args.vllm_root / relative).read_bytes()
        tree = ast.parse(raw)
        symbols = [{"name": n.name, "line": n.lineno} for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        source_records[relative] = {"sha256": sha(raw), "symbols": symbols}
    report = {
        "status": "static_cpu_audit_not_backend_support",
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_sha256": sha(config_raw),
        "tokenizer_sha256": sha(tokenizer.read_bytes()) if tokenizer.exists() else None,
        "runtime_snapshot_sha256": sha(runtime_raw),
        "runtime_snapshot_utc": runtime.get("utc"),
        "captured_metrics_sha256": sha(metric_text.encode()),
        "runtime_source_revision_reported": runtime.get("source_revision"),
        "runtime_manifest_sha256_reported": runtime.get("runtime_python_manifest_sha256"),
        "tensor_parallel_size": tp,
        "configured_context": int(option("--max-model-len", "0")),
        "model_max_positions": config["text_config"]["max_position_embeddings"],
        "primary_block_size_captured": int(labels["block_size"]),
        "physical_per_group_geometry_captured": False,
        "boundary_native_layers": list(skip),
        "cache_dtype": labels["cache_dtype"],
        "unified_tq": runtime["environment"].get("VLLM_TQ_UNIFIED_ATTN") == "1",
        "swa_bounded_replay_captured": labels.get("swa_bounded_replay"),
        "speculative_method": speculative.get("method"),
        "speculative_tokens": speculative.get("num_speculative_tokens", 0),
        "final_vocabulary_logit_softcap": config["text_config"].get("final_logit_softcapping"),
        "attention_softcap": config["text_config"].get("attn_logit_softcapping"),
        "all_layer_shapes_are_per_rank": True,
        "layers": [asdict(layer) for layer in layers],
        "source_files": source_records,
        "weights_digest_verified": False,
        "live_cache_allocations_observed": False,
        "gpu_calls": 0,
        "production_packages_modified": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    # No prompts, credentials, environment dump, or raw deployment argv.
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    fixture = {"text_config": config["text_config"]}
    (args.output / "model_attention_config.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "tensor_parallel_size",
        "boundary_native_layers", "primary_block_size_captured", "unified_tq", "speculative_tokens")}))


if __name__ == "__main__":
    main()
