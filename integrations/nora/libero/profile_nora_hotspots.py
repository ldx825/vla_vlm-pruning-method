import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.profiler import ProfilerActivity, profile

from autoprune_nora import NoraAutoPruneConfig
from nora_utils import Nora
from qwen_vl_utils import process_vision_info


DEFAULT_HOTSPOT_OUT = (
    "/cache/ma-user/nora_eval_logs/hotspot_report.txt"
    if os.path.isdir("/cache/ma-user")
    else "./experiments/logs/hotspot_report.txt"
)


@dataclass
class RunConfig:
    name: str
    enable_autoprune: bool
    disable_progressive_prune: bool = False
    static_drop_tokens_at_start: int = 0


def build_inputs(model: Nora, image_size: int, prompt: str) -> Dict[str, torch.Tensor]:
    img = Image.fromarray(np.random.randint(0, 255, size=(image_size, image_size, 3), dtype=np.uint8))

    primary = {"type": "image", "image": img, "resized_height": image_size, "resized_width": image_size}
    wrist = {"type": "image", "image": img, "resized_height": image_size, "resized_width": image_size}
    messages = [{"role": "user", "content": [primary, wrist, {"type": "text", "text": prompt}]}]

    text = model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = model.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(model.device) for k, v in inputs.items()}


def make_model(args: argparse.Namespace, cfg: RunConfig) -> Nora:
    ap_cfg = NoraAutoPruneConfig(
        enabled=cfg.enable_autoprune,
        visual_tokens=args.autoprune_visual_tokens,
        start_layer=args.autoprune_start_layer,
        num_views=2,
        b_min=args.autoprune_b_min,
        b_max=args.autoprune_b_max,
        monitor_only=False,
        disable_token_prune=False,
        disable_progressive_prune=cfg.disable_progressive_prune,
        static_drop_tokens_at_start=cfg.static_drop_tokens_at_start,
    )

    model = Nora(
        model_path=args.model_path,
        lora_adapter_path=args.lora_adapter_path,
        image_resize_size=(args.image_size, args.image_size),
        force_attn_implementation="eager",
        force_output_attentions=False,
        autoprune_config=ap_cfg,
        enable_stage_profiling=False,
    )
    return model


def summarize_profiles(profiles: List[Dict[str, float]]) -> Dict[str, float]:
    if not profiles:
        return {}
    keys = [
        "inference_total_seconds",
        "preprocess_seconds",
        "generate_seconds",
        "decode_seconds",
        "vit_seconds",
        "prefill_seconds",
        "decode_model_seconds",
        "patch_total_seconds",
        "force_attn_seconds",
        "topk_seconds",
        "gather_seconds",
        "last_tokens_before",
        "last_tokens_after",
        "pruned_layers",
    ]
    out: Dict[str, float] = {}
    n = float(len(profiles))
    for key in keys:
        out[key] = sum(float(p.get(key, 0.0) or 0.0) for p in profiles) / n
    return out


def run_hotspot(args: argparse.Namespace, cfg: RunConfig) -> Dict[str, str]:
    model = make_model(args, cfg)
    inputs = build_inputs(model, args.image_size, args.prompt)

    for _ in range(args.warmup_steps):
        if model.autoprune_patch is not None:
            model.autoprune_patch.reset_runtime_state()
        _ = model.model.generate(**inputs)
        if model.device.startswith("cuda"):
            torch.cuda.synchronize(device=model.device)

    profiles: List[Dict[str, float]] = []
    wall_times_ms: List[float] = []

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(args.profile_steps):
            if model.autoprune_patch is not None:
                model.autoprune_patch.reset_runtime_state()
            t0 = time.perf_counter()
            _ = model.model.generate(**inputs)
            if model.device.startswith("cuda"):
                torch.cuda.synchronize(device=model.device)
            wall_times_ms.append((time.perf_counter() - t0) * 1000.0)

            if cfg.enable_autoprune and model.autoprune_patch is not None:
                stats = dict(model.autoprune_patch.runtime_stats or {})
                profiles.append({
                    "patch_total_seconds": float(stats.get("patch_total_seconds", 0.0) or 0.0),
                    "force_attn_seconds": float(stats.get("force_attn_seconds", 0.0) or 0.0),
                    "topk_seconds": float(stats.get("topk_seconds", 0.0) or 0.0),
                    "gather_seconds": float(stats.get("gather_seconds", 0.0) or 0.0),
                    "last_tokens_before": float(stats.get("last_tokens_before", 0.0) or 0.0),
                    "last_tokens_after": float(stats.get("last_tokens_after", 0.0) or 0.0),
                    "pruned_layers": float(stats.get("pruned_layers", 0.0) or 0.0),
                })

    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=args.topk_ops)

    summary = summarize_profiles(profiles)
    result = {
        "name": cfg.name,
        "avg_generate_wall_ms": f"{sum(wall_times_ms) / max(1, len(wall_times_ms)):.2f}",
        "min_generate_wall_ms": f"{min(wall_times_ms):.2f}",
        "max_generate_wall_ms": f"{max(wall_times_ms):.2f}",
        "profiler_table": table,
        "summary": "\n".join([f"{k}: {v}" for k, v in summary.items()]) if summary else "(no runtime_stats)",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Hotspot profiler for NORA/AutoPrune")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_adapter_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--prompt", type=str, default="pick up the black bowl between the plate and the ramekin and place it on the plate")
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--profile_steps", type=int, default=8)
    parser.add_argument("--topk_ops", type=int, default=30)
    parser.add_argument("--autoprune_visual_tokens", type=int, default=512)
    parser.add_argument("--autoprune_start_layer", type=int, default=2)
    parser.add_argument("--autoprune_b_min", type=int, default=96)
    parser.add_argument("--autoprune_b_max", type=int, default=156)
    parser.add_argument("--out", type=str, default=DEFAULT_HOTSPOT_OUT)
    args = parser.parse_args()

    run_cfgs = [
        RunConfig(name="baseline", enable_autoprune=False),
        RunConfig(name="autoprune_dynamic", enable_autoprune=True, disable_progressive_prune=False, static_drop_tokens_at_start=0),
        RunConfig(name="autoprune_static_drop500", enable_autoprune=True, disable_progressive_prune=True, static_drop_tokens_at_start=500),
    ]

    lines: List[str] = []
    lines.append("=== NORA Hotspot Profiling Report ===")
    lines.append(f"model_path: {args.model_path}")
    lines.append(f"lora_adapter_path: {args.lora_adapter_path}")
    lines.append(f"image_size: {args.image_size}")
    lines.append(f"profile_steps: {args.profile_steps}")
    lines.append("")

    for cfg in run_cfgs:
        lines.append(f"=== [{cfg.name}] ===")
        result = run_hotspot(args, cfg)
        lines.append(f"avg_generate_wall_ms: {result['avg_generate_wall_ms']}")
        lines.append(f"min_generate_wall_ms: {result['min_generate_wall_ms']}")
        lines.append(f"max_generate_wall_ms: {result['max_generate_wall_ms']}")
        lines.append("runtime_stats_summary:")
        lines.append(result["summary"])
        lines.append("top_cuda_hotspots:")
        lines.append(result["profiler_table"])
        lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved hotspot report to: {args.out}")


if __name__ == "__main__":
    main()
