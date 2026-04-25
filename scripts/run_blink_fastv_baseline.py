import argparse
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MANTIS_ROOT = Path(os.environ.get("MANTIS_ROOT", "/mnt/jarvis/migration/Mantis")).resolve()
MANTIS_RUNNER = MANTIS_ROOT / "scripts" / "autoprune" / "run_blink_multiview.py"

sys.path.insert(0, str(MANTIS_ROOT))
sys.path.insert(0, str(MANTIS_ROOT / "mantis"))
sys.path.insert(0, str(REPO_ROOT))

from methods.baseline_pruning.mantis import (  # noqa: E402
    build_mantis_baseline_config_from_env,
    maybe_apply_mantis_baseline_pruning,
)
from mllm_tools import MLLM_Models  # noqa: E402


def _alias_env_if_missing(dst: str, src: str) -> None:
    if os.environ.get(dst) is None and os.environ.get(src) is not None:
        os.environ[dst] = os.environ[src]


def _normalize_baseline_prune_env_aliases() -> None:
    # Older launch snippets used MANTIS_BASELINE_PRUNING_* while the config
    # reader intentionally uses MANTIS_BASELINE_PRUNE_*. Accept both so a typo
    # cannot silently disable pruning during debug or sweeps.
    aliases = {
        "METHOD": "METHOD",
        "ENABLED": "ENABLED",
        "TARGET_TOKEN_NUM": "TARGET_TOKEN_NUM",
        "DEBUG": "DEBUG",
        "DISABLE_KV_CACHE": "DISABLE_KV_CACHE",
        "FASTV_PRUNING_LAYER": "LAYER",
    }
    for dst_suffix, legacy_suffix in aliases.items():
        _alias_env_if_missing(
            f"MANTIS_BASELINE_PRUNE_{dst_suffix}",
            f"MANTIS_BASELINE_PRUNING_{legacy_suffix}",
        )


def _describe_model_object(obj) -> str:
    return f"{type(obj).__module__}.{type(obj).__name__}"


def _candidate_patch_targets(model):
    seen = set()

    def add(obj, label):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        yield label, obj

    yield from add(getattr(model, "model", None), "wrapper.model")
    yield from add(model, "wrapper")
    base_getter = getattr(getattr(model, "model", None), "get_base_model", None)
    if callable(base_getter):
        try:
            yield from add(base_getter(), "wrapper.model.get_base_model()")
        except Exception:
            pass


def _apply_baseline_pruning_or_raise(model, baseline_config):
    if not baseline_config.enabled:
        raise RuntimeError(
            "FastV/Mantis baseline pruning is disabled. Set "
            "MANTIS_BASELINE_PRUNE_ENABLED=1 "
            "(legacy MANTIS_BASELINE_PRUNING_ENABLED=1 is also accepted)."
        )

    attempts = []
    for label, target in _candidate_patch_targets(model):
        patch = maybe_apply_mantis_baseline_pruning(target, baseline_config)
        layer_count = len(getattr(patch, "layers", []) or []) if patch is not None else 0
        attempts.append(f"{label}:{_describe_model_object(target)} layers={layer_count}")
        if patch is not None and layer_count > 0:
            model.autoprune_patch = patch
            model.last_autoprune_stats = None
            if hasattr(target, "_autoprune_patch"):
                target._autoprune_patch = patch
            return patch, label, target, attempts

    raise RuntimeError(
        "Failed to attach Mantis baseline pruning patch to a callable model target. "
        "Attempts: " + "; ".join(attempts)
    )


def _load_mantis_runner():
    spec = importlib.util.spec_from_file_location("mantis_blink_runner", MANTIS_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Mantis BLINK runner from {MANTIS_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _avg_stat(rows, key):
    values = [
        row["autoprune_stats"][key]
        for row in rows
        if isinstance(row.get("autoprune_stats"), dict) and row["autoprune_stats"].get(key) is not None
    ]
    return float(np.mean(values)) if values else None


def main():
    runner = _load_mantis_runner()
    _normalize_baseline_prune_env_aliases()

    parser = argparse.ArgumentParser(description="BLINK multi-view runner for Mantis baseline pruning methods.")
    parser.add_argument("--model-name", default="mantis-8b-siglip-llama3-ablation1")
    parser.add_argument("--task-name", default="Multi-view_Reasoning")
    parser.add_argument("--split", default="val")
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--sub-sample-size", type=int, default=133)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-mode", choices=["accuracy", "latency"], default="accuracy")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--answer-backend", choices=["generate", "loss"], default=None)
    parser.add_argument(
        "--loss-choice-format",
        choices=["plain_text", "letter", "option_line", "answer_letter", "answer_text"],
        default=None,
    )
    args = parser.parse_args()

    random.seed(args.seed)
    chosen_max_new_tokens = runner.resolve_generation_budget(args.eval_mode, args.max_new_tokens)
    answer_backend = runner.resolve_answer_backend(args.eval_mode, args.answer_backend)
    loss_choice_format = runner.resolve_loss_choice_format(args.loss_choice_format)
    os.environ["MANTIS_GENERATION_MAX_NEW_TOKENS"] = str(chosen_max_new_tokens)

    dataset = runner.load_dataset_entries(args.task_name, args.split, args.sub_sample_size, args.seed)
    model_ctor = MLLM_Models(args.model_name)
    model = model_ctor()

    baseline_config = build_mantis_baseline_config_from_env()
    baseline_patch, patch_label, patch_target, patch_attempts = _apply_baseline_pruning_or_raise(
        model,
        baseline_config,
    )
    print(
        "[blink-fastv] baseline pruning attached: "
        f"method={baseline_config.method}, target={baseline_config.target_token_num}, "
        f"fastv_layer={baseline_config.fastv_pruning_layer}, "
        f"target={patch_label}:{_describe_model_object(patch_target)}, "
        f"layers={len(baseline_patch.layers)}"
    )

    rows = []
    for entry in dataset:
        inputs = runner.build_inputs(entry, args.eval_mode, answer_backend)
        t0 = time.perf_counter()
        choice_scores = None
        if answer_backend == "loss":
            rendered_choices, choice_labels, parsed_texts = runner.build_loss_choices(entry, loss_choice_format)
            if not hasattr(model, "score_choices"):
                raise RuntimeError(f"Model {args.model_name} does not support loss-based choice scoring.")
            choice_scores = model.score_choices(inputs, rendered_choices, normalize_by_length=True)
            best_choice = choice_scores[0]["choice"]
            best_idx = rendered_choices.index(best_choice)
            raw_answer = best_choice
            prediction = choice_labels[best_idx]
            correct = prediction == entry["answer"]
            parsed_answer = parsed_texts[best_idx]
        else:
            raw_answer = model(inputs)
            prediction, correct, parsed_answer = runner.evaluate_multichoice(
                raw_answer,
                entry["answer"],
                entry["choices"],
            )
        latency = time.perf_counter() - t0
        rows.append(
            {
                "idx": entry["idx"],
                "task_name": args.task_name,
                "split": args.split,
                "num_views": sum(1 for item in inputs if item["type"] == "image"),
                "latency_seconds": float(latency),
                "answer": entry["answer"],
                "prediction": prediction,
                "parsed_answer": parsed_answer,
                "correct": bool(correct),
                "raw_answer": raw_answer,
                "choice_scores": choice_scores,
                "autoprune_stats": getattr(model, "last_autoprune_stats", None)
                or getattr(baseline_patch, "runtime_stats", None),
                "multiview_token_budget": getattr(model, "multiview_token_budget", None),
            }
        )
        if len(rows) % 10 == 0:
            print(f"[blink-fastv] progress {len(rows)}/{len(dataset)}", flush=True)

    summary = {
        "model_name": args.model_name,
        "task_name": args.task_name,
        "split": args.split,
        "eval_mode": args.eval_mode,
        "answer_backend": answer_backend,
        "loss_choice_format": loss_choice_format if answer_backend == "loss" else None,
        "generation_max_new_tokens": chosen_max_new_tokens,
        "sample_count": len(rows),
        "avg_latency_seconds": float(np.mean([row["latency_seconds"] for row in rows])) if rows else None,
        "avg_accuracy": float(np.mean([1.0 if row["correct"] else 0.0 for row in rows])) if rows else None,
        "avg_num_views": float(np.mean([row["num_views"] for row in rows])) if rows else None,
        "avg_prefill_seq_len": _avg_stat(rows, "prefill_seq_len"),
        "avg_last_tokens_before": _avg_stat(rows, "last_tokens_before"),
        "avg_last_tokens_after": _avg_stat(rows, "last_tokens_after"),
        "avg_method_overhead_seconds": _avg_stat(rows, "method_overhead_seconds"),
        "avg_flops_reduction": _avg_stat(rows, "flops_reduction"),
        "baseline_pruning_config": baseline_patch.config.__dict__ if baseline_patch is not None else None,
        "baseline_pruning_patch_target": patch_label,
        "baseline_pruning_patch_target_type": _describe_model_object(patch_target),
        "baseline_pruning_patch_attempts": patch_attempts,
    }

    if rows:
        prediction_counter = Counter(row["prediction"] for row in rows)
        answer_counter = Counter(row["answer"] for row in rows)
        top_prediction, top_prediction_count = prediction_counter.most_common(1)[0]
        summary["prediction_distribution"] = dict(prediction_counter)
        summary["answer_distribution"] = dict(answer_counter)
        summary["top_prediction"] = top_prediction
        summary["top_prediction_ratio"] = float(top_prediction_count / len(rows))
        summary["degenerate_prediction_warning"] = bool(top_prediction_count / len(rows) >= 0.95)
        summary["raw_answer_preview"] = [row["raw_answer"] for row in rows[:5]]

    payload = {"summary": summary, "rows": rows}
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
