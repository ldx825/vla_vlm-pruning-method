"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import wandb
from libero.libero import benchmark

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
LIBERO_DEPS_SITE = os.environ.get("LIBERO_DEPS_SITE")
if LIBERO_DEPS_SITE and LIBERO_DEPS_SITE not in sys.path:
    # Append, do not prepend: NORA needs its own newer transformers/qwen stack.
    sys.path.append(LIBERO_DEPS_SITE)

DEFAULT_LOCAL_LOG_DIR = "/cache/ma-user/nora_eval_logs" if os.path.isdir("/cache/ma-user") else "./experiments/logs"

from libero_utils import get_libero_dummy_action, get_libero_env, get_libero_images, get_libero_video_frame, quat2axisangle, save_rollout_video
from robot_utils import DATE_TIME, get_image_resize_size, invert_gripper_action, normalize_gripper_action, set_seed_everywhere
from nora_utils import Nora
from autoprune_nora import NoraAutoPruneConfig
try:
    from compare_pruning_patch_nora import NoraComparePruningConfig
except Exception:
    @dataclass
    class NoraComparePruningConfig:
        enabled: bool = False
        method: str = "off"
        visual_tokens: int = 256
        num_views: int = 2
        fastv_pruning_layer: int = 2
        sparsevlm_pruning_start_layer: int = 2
        sparsevlm_lambda: float = 1.0
        sparsevlm_recycle_top_ratio: float = 0.5
        sparsevlm_cluster_ratio: float = 0.25
        sparsevlm_rank_tolerance: float = 1e-4
        sparsevlm_knn_k: int = 8
        detailed_timing: bool = True
        debug: bool = False
from nora_baseline_pruning import NoraBaselinePruningConfig


# Keep latency accounting identical to OpenVLA eval script: per inference call,
# skip the first measured call as warmup.
GLOBAL_INFERENCE_TIME = 0.0
GLOBAL_INFERENCE_COUNT = 0


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_latency_ms(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }

    vals = sorted(vals)

    def percentile(p):
        idx = max(0, min(len(vals) - 1, int(np.ceil(p * len(vals))) - 1))
        return float(vals[idx])

    return {
        "count": len(vals),
        "mean_ms": float(np.mean(vals)),
        "median_ms": float(np.median(vals)),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "min_ms": float(vals[0]),
        "max_ms": float(vals[-1]),
    }


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_resize_resolution(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return (size, size)
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    raise argparse.ArgumentTypeError(
        f"Expected image resize resolution in the form '224' or '224,224', got {value!r}"
    )


def parse_int_tuple(value) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return tuple()
    return tuple(int(part) for part in parts)


def apply_autoprune_preset(cfg):
    preset = str(getattr(cfg, "autoprune_preset", "off") or "off").strip().lower()
    if preset in {"", "off", "none"}:
        return cfg

    cfg.enable_autoprune = True
    budget_mode = str(getattr(cfg, "autoprune_budget_mode", "absolute") or "absolute").strip().lower()
    budget_semantics = str(getattr(cfg, "autoprune_budget_semantics", "final_cap") or "final_cap").strip().lower()
    using_absolute_area = budget_mode == "absolute" and budget_semantics == "area"
    using_default_min_keep = int(getattr(cfg, "autoprune_min_keep_per_view", 24) or 24) == 24
    using_default_start_layer = int(getattr(cfg, "autoprune_start_layer", 2) or 2) == 2

    if preset == "monitor_448":
        cfg.autoprune_visual_tokens = 512
        if using_default_start_layer:
            cfg.autoprune_start_layer = 6
        cfg.autoprune_monitor_only = True
        cfg.autoprune_disable_token_prune = True
        cfg.autoprune_disable_progressive_prune = True
        cfg.autoprune_allocation_temp = 0.8
        if using_default_min_keep:
            cfg.autoprune_min_keep_per_view = 128
        cfg.force_eager_attention = True
        return cfg

    if preset in {"single_shot_448", "conservative_448"}:
        cfg.autoprune_visual_tokens = 512
        if using_default_start_layer:
            cfg.autoprune_start_layer = 6
        cfg.autoprune_disable_progressive_prune = True
        cfg.autoprune_allocation_temp = 0.8
        if using_default_min_keep:
            cfg.autoprune_min_keep_per_view = 128
        if not using_absolute_area:
            cfg.autoprune_fixed_total_budget = 416 if cfg.autoprune_fixed_total_budget <= 0 else cfg.autoprune_fixed_total_budget
        cfg.force_eager_attention = True
        return cfg

    if preset in {"flash_single_shot_448", "flash_conservative_448"}:
        cfg.autoprune_visual_tokens = 512
        if using_default_start_layer:
            cfg.autoprune_start_layer = 6
        cfg.autoprune_disable_progressive_prune = True
        cfg.autoprune_allocation_temp = 0.8
        if using_default_min_keep:
            cfg.autoprune_min_keep_per_view = 128
        cfg.autoprune_prefer_flash_attention = True
        cfg.autoprune_manual_attention_probe = True
        if not using_absolute_area:
            cfg.autoprune_fixed_total_budget = 416 if cfg.autoprune_fixed_total_budget <= 0 else cfg.autoprune_fixed_total_budget
        return cfg

    if preset in {"flash_progressive_448", "flash_progressive_conservative_448", "flash_448"}:
        cfg.autoprune_visual_tokens = 512
        if using_default_start_layer:
            cfg.autoprune_start_layer = 6
        cfg.autoprune_disable_progressive_prune = False
        cfg.autoprune_allocation_temp = 0.8
        if using_default_min_keep:
            cfg.autoprune_min_keep_per_view = 24
        cfg.autoprune_prefer_flash_attention = True
        cfg.autoprune_manual_attention_probe = True
        if not using_absolute_area:
            cfg.autoprune_fixed_total_budget = 416 if cfg.autoprune_fixed_total_budget <= 0 else cfg.autoprune_fixed_total_budget
        return cfg

    raise ValueError(f"Unknown AutoPrune preset: {cfg.autoprune_preset!r}")


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "nora"                       # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Reserved for compatibility
    model_path: Union[str, Path] = "declare-lab/nora-long-finetuned-libero-object"
    lora_adapter_path: Optional[Union[str, Path]] = None
    full_model_state_path: Optional[Union[str, Path]] = None
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    action_horizon: int = 5                          # Use 1 for base NORA, 5 for NORA-LONG-style checkpoints

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"         # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                 # Number of rollouts per task
    trial_offset: int = 0                         # Starting trial index within the task init-state list
    num_tasks_to_run: int = 1                      # Run only first N tasks for quick experiments; <=0 means all
    image_resize_resolution: tuple[int, int] = (448, 448)
    env_camera_resolution: int = 448
    use_wrist_image: bool = True
    dual_view_mode: str = "separate"
    use_anyres: bool = False
    force_eager_attention: bool = False
    force_output_attentions: bool = False
    enable_stage_profiling: bool = True
    stop_at_expected_action_length: bool = False

    #################################################################################################################
    # AutoPrune (inference only)
    #################################################################################################################
    enable_autoprune: bool = False
    autoprune_preset: str = "off"
    autoprune_visual_tokens: int = 512
    autoprune_start_layer: int = 2
    autoprune_num_views: int = 2
    autoprune_allocation_temp: float = 0.4
    autoprune_x0: float = 12.0
    autoprune_k_shape: float = 0.3
    autoprune_b_min: int = 96
    autoprune_b_max: int = 156
    autoprune_fixed_total_budget: int = 0
    autoprune_target_token_num: int = 0
    autoprune_mu: float = 4.0
    autoprune_tau: float = 0.5
    autoprune_min_keep_per_view: int = 24
    autoprune_budget_mode: str = "absolute"
    autoprune_absolute_anchor_mode: str = "elbow"
    autoprune_absolute_budget_scale: float = 1.0
    autoprune_absolute_auto_bounds: bool = True
    autoprune_enable_adaptive_layer_policy: bool = True
    autoprune_enable_oneshot_fallback: bool = True
    autoprune_adaptive_layer_choices: tuple[int, int, int] = (1, 3, 6)
    autoprune_adaptive_layer_mu: float = 4.0
    autoprune_adaptive_layer_tau: float = 0.5
    autoprune_oneshot_fallback_max_layers: int = 3
    autoprune_oneshot_fallback_budget_margin: float = 0.15
    autoprune_oneshot_fallback_frontload_ratio: float = 0.5
    autoprune_monitor_only: bool = False
    autoprune_disable_token_prune: bool = False
    autoprune_disable_progressive_prune: bool = False
    autoprune_static_drop_tokens_at_start: int = 0
    autoprune_prefer_flash_attention: bool = False
    autoprune_manual_attention_probe: bool = False
    autoprune_manual_attention_probe_stride: int = 1
    autoprune_manual_attention_probe_max_layers: int = 0
    autoprune_prune_only_on_anchor_layers: bool = False
    autoprune_piecewise_linear_keep_schedule: bool = False
    autoprune_piecewise_linear_hard_final_cap: bool = True
    autoprune_budget_semantics: str = "final_cap"
    autoprune_budget_aware_x0_shift: float = 0.0
    autoprune_compaction_delta_threshold_tokens: int = 0
    autoprune_physical_compaction_max_shots: int = 0
    autoprune_detailed_timing: bool = True
    autoprune_debug: bool = False

    #################################################################################################################
    # Compare pruning baselines (FastV / SparseVLM)
    #################################################################################################################
    enable_compare_pruning: bool = False
    compare_pruning_method: str = "off"
    compare_visual_tokens: int = 256
    compare_num_views: int = 2
    compare_fastv_pruning_layer: int = 2
    compare_sparsevlm_pruning_start_layer: int = 2
    compare_sparsevlm_lambda: float = 1.0
    compare_sparsevlm_recycle_top_ratio: float = 0.5
    compare_sparsevlm_cluster_ratio: float = 0.25
    compare_sparsevlm_rank_tolerance: float = 1e-4
    compare_sparsevlm_knn_k: int = 8
    compare_detailed_timing: bool = True
    compare_debug: bool = False

    #################################################################################################################
    # Paper baselines (FasterVLM / VisionZip / PyramidDrop)
    #################################################################################################################
    enable_baseline_pruning: bool = False
    baseline_pruning_method: str = "fastervlm"
    baseline_target_token_num: int = 128
    baseline_visual_tokens: int = 512
    baseline_start_layer: int = 2
    baseline_max_pruning_layers: int = 6
    baseline_layer_indices: tuple[int, ...] = tuple()
    baseline_visionzip_dominant_ratio: float = 0.7
    baseline_debug: bool = False

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = DEFAULT_LOCAL_LOG_DIR        # Local directory for eval logs
    latency_csv_filename: Optional[str] = None        # Optional CSV filename for per-step latency diagnostics

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)
    
    
    # fmt: on


def run_eval_libero(cfg: GenerateConfig) -> None:
    global GLOBAL_INFERENCE_TIME, GLOBAL_INFERENCE_COUNT
    GLOBAL_INFERENCE_TIME = 0.0
    GLOBAL_INFERENCE_COUNT = 0

    if cfg.pretrained_checkpoint and "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    cfg = apply_autoprune_preset(cfg)
    enabled_pruners = int(bool(cfg.enable_autoprune)) + int(bool(cfg.enable_compare_pruning)) + int(bool(cfg.enable_baseline_pruning))
    if enabled_pruners > 1:
        raise ValueError("Enable only one pruning path among AutoPrune, compare pruning, and paper baseline pruning.")
    budget_mode = str(getattr(cfg, "autoprune_budget_mode", "absolute") or "absolute").strip().lower()
    budget_semantics = str(getattr(cfg, "autoprune_budget_semantics", "final_cap") or "final_cap").strip().lower()
    using_absolute_area = budget_mode == "absolute" and budget_semantics == "area"

    if cfg.autoprune_fixed_total_budget > 0 and not using_absolute_area:
        cfg.autoprune_b_min = cfg.autoprune_fixed_total_budget
        cfg.autoprune_b_max = cfg.autoprune_fixed_total_budget
    if budget_semantics == "final_cap" and cfg.autoprune_target_token_num <= 0 and cfg.autoprune_fixed_total_budget > 0:
        cfg.autoprune_target_token_num = cfg.autoprune_fixed_total_budget
    if budget_semantics == "area":
        cfg.autoprune_piecewise_linear_hard_final_cap = False
        if cfg.autoprune_target_token_num <= 0:
            cfg.autoprune_target_token_num = 0

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    
    model = Nora(
        model_path=str(cfg.model_path),
        image_resize_size=get_image_resize_size(cfg),
        lora_adapter_path=str(cfg.lora_adapter_path) if cfg.lora_adapter_path else None,
        full_model_state_path=str(cfg.full_model_state_path) if cfg.full_model_state_path else None,
        dual_view_mode=cfg.dual_view_mode,
        use_anyres=cfg.use_anyres,
        force_attn_implementation="eager" if cfg.force_eager_attention else None,
        force_output_attentions=cfg.force_output_attentions,
        enable_stage_profiling=cfg.enable_stage_profiling,
        stop_at_expected_action_length=cfg.stop_at_expected_action_length,
        autoprune_config=NoraAutoPruneConfig(
            enabled=cfg.enable_autoprune,
            target_token_num=cfg.autoprune_target_token_num,
            visual_tokens=cfg.autoprune_visual_tokens,
            start_layer=cfg.autoprune_start_layer,
            num_views=cfg.autoprune_num_views,
            allocation_temp=cfg.autoprune_allocation_temp,
            x0=cfg.autoprune_x0,
            k_shape=cfg.autoprune_k_shape,
            b_min=cfg.autoprune_b_min,
            b_max=cfg.autoprune_b_max,
            mu=cfg.autoprune_mu,
            tau=cfg.autoprune_tau,
            min_keep_per_view=cfg.autoprune_min_keep_per_view,
            budget_mode=cfg.autoprune_budget_mode,
            absolute_anchor_mode=cfg.autoprune_absolute_anchor_mode,
            absolute_budget_scale=cfg.autoprune_absolute_budget_scale,
            absolute_auto_bounds=cfg.autoprune_absolute_auto_bounds,
            enable_adaptive_layer_policy=cfg.autoprune_enable_adaptive_layer_policy,
            enable_oneshot_fallback=cfg.autoprune_enable_oneshot_fallback,
            adaptive_layer_choices=cfg.autoprune_adaptive_layer_choices,
            adaptive_layer_mu=cfg.autoprune_adaptive_layer_mu,
            adaptive_layer_tau=cfg.autoprune_adaptive_layer_tau,
            oneshot_fallback_max_layers=cfg.autoprune_oneshot_fallback_max_layers,
            oneshot_fallback_budget_margin=cfg.autoprune_oneshot_fallback_budget_margin,
            oneshot_fallback_frontload_ratio=cfg.autoprune_oneshot_fallback_frontload_ratio,
            monitor_only=cfg.autoprune_monitor_only,
            disable_token_prune=cfg.autoprune_disable_token_prune,
            disable_progressive_prune=cfg.autoprune_disable_progressive_prune,
            static_drop_tokens_at_start=cfg.autoprune_static_drop_tokens_at_start,
            prefer_flash_attention=cfg.autoprune_prefer_flash_attention,
            manual_attention_probe=cfg.autoprune_manual_attention_probe,
            manual_attention_probe_stride=cfg.autoprune_manual_attention_probe_stride,
            manual_attention_probe_max_layers=cfg.autoprune_manual_attention_probe_max_layers,
            prune_only_on_anchor_layers=cfg.autoprune_prune_only_on_anchor_layers,
            piecewise_linear_keep_schedule=cfg.autoprune_piecewise_linear_keep_schedule,
            piecewise_linear_hard_final_cap=cfg.autoprune_piecewise_linear_hard_final_cap,
            budget_semantics=cfg.autoprune_budget_semantics,
            budget_aware_x0_shift=cfg.autoprune_budget_aware_x0_shift,
            compaction_delta_threshold_tokens=cfg.autoprune_compaction_delta_threshold_tokens,
            physical_compaction_max_shots=cfg.autoprune_physical_compaction_max_shots,
            detailed_timing=cfg.autoprune_detailed_timing,
            debug=cfg.autoprune_debug,
        ),
        compare_pruning_config=NoraComparePruningConfig(
            enabled=cfg.enable_compare_pruning,
            method=cfg.compare_pruning_method,
            visual_tokens=cfg.compare_visual_tokens,
            num_views=cfg.compare_num_views,
            fastv_pruning_layer=cfg.compare_fastv_pruning_layer,
            sparsevlm_pruning_start_layer=cfg.compare_sparsevlm_pruning_start_layer,
            sparsevlm_lambda=cfg.compare_sparsevlm_lambda,
            sparsevlm_recycle_top_ratio=cfg.compare_sparsevlm_recycle_top_ratio,
            sparsevlm_cluster_ratio=cfg.compare_sparsevlm_cluster_ratio,
            sparsevlm_rank_tolerance=cfg.compare_sparsevlm_rank_tolerance,
            sparsevlm_knn_k=cfg.compare_sparsevlm_knn_k,
            detailed_timing=cfg.compare_detailed_timing,
            debug=cfg.compare_debug,
        ),
        baseline_pruning_config=NoraBaselinePruningConfig(
            enabled=cfg.enable_baseline_pruning,
            method=cfg.baseline_pruning_method,
            target_token_num=cfg.baseline_target_token_num,
            visual_tokens=cfg.baseline_visual_tokens,
            start_layer=cfg.baseline_start_layer,
            max_pruning_layers=cfg.baseline_max_pruning_layers,
            layer_indices=cfg.baseline_layer_indices,
            visionzip_dominant_ratio=cfg.baseline_visionzip_dominant_ratio,
            debug=cfg.baseline_debug,
        ),
    )
    model.fast_tokenizer.time_horizon = cfg.action_horizon

    # [OpenVLA] Check that the model contains the action un-normalization key
    #if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        ##if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            #cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
       # assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    #processor = None
    #if cfg.model_family == "openvla":
       # processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-qwen-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    results_json_path = os.path.join(cfg.local_log_dir, run_id + ".json")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    if cfg.env_camera_resolution < max(resize_size):
        print(
            "Warning: env_camera_resolution is lower than image_resize_resolution. "
            "This upscales low-res observations and can hurt both policy quality and rollout video fidelity."
        )

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = []
    timing_totals = {
        "obs_preprocess_seconds": 0.0,
        "env_step_seconds": 0.0,
        "inference_total_seconds": 0.0,
        "inference_preprocess_seconds": 0.0,
        "inference_generate_seconds": 0.0,
        "inference_decode_seconds": 0.0,
        "inference_vit_seconds": 0.0,
        "inference_prefill_seconds": 0.0,
        "inference_decode_model_seconds": 0.0,
        "patch_total_seconds": 0.0,
        "force_attn_seconds": 0.0,
        "force_attn_self_attn_seconds": 0.0,
        "force_attn_mlp_seconds": 0.0,
        "force_attn_residual_norm_seconds": 0.0,
        "regular_forward_seconds": 0.0,
        "regular_self_attn_seconds": 0.0,
        "regular_mlp_seconds": 0.0,
        "regular_residual_norm_seconds": 0.0,
        "manual_probe_seconds": 0.0,
        "metric_seconds": 0.0,
        "rank_seconds": 0.0,
        "budget_schedule_seconds": 0.0,
        "topk_seconds": 0.0,
        "gather_seconds": 0.0,
        "compaction_bookkeeping_seconds": 0.0,
        "last_tokens_before": 0.0,
        "last_tokens_after": 0.0,
        "schedule_len": 0.0,
        "pruned_layers": 0.0,
        "layout_ready": 0.0,
        "layout_query_tokens": 0.0,
        "layout_view0_tokens": 0.0,
        "layout_view1_tokens": 0.0,
        "manual_attention_probe_layers": 0.0,
        "manual_probe_qk_cache_hits": 0.0,
        "manual_probe_qk_cache_misses": 0.0,
        "manual_probe_reused_layers": 0.0,
        "planned_compaction_layers": 0.0,
        "generated_token_count": 0.0,
        "action_token_count": 0.0,
        "decode_to_action_ratio": 0.0,
        "expected_coeff_count": 0.0,
        "layer0_total_calls": 0.0,
        "layer0_short_seq_calls": 0.0,
        "layer0_long_seq_calls": 0.0,
        "layer0_past_none_calls": 0.0,
        "layer0_past_present_calls": 0.0,
        "flops_reduction": 0.0,
        "decoder_layers_recorded": 0.0,
        "decoder_layers_missing": 0.0,
        "final_visual_tokens": 0.0,
        "raw_total_budget": 0.0,
        "active_pruning_layers": 0.0,
    }
    per_step_latency_rows = []
    task_count = num_tasks_in_suite if cfg.num_tasks_to_run <= 0 else min(num_tasks_in_suite, cfg.num_tasks_to_run)
    for task_id in tqdm.tqdm(range(task_count)):
        env = None
        task_description = ""

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_camera_resolution)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_latency_ms = []
        task_result = {
            "task_id": int(task_id),
            "episodes": [],
        }
        for local_episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            episode_idx = cfg.trial_offset + local_episode_idx
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            task_result["task_description"] = task_description

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 500  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 500  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 500  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 500  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 500  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1} (trial index {episode_idx})...")
            log_file.write(f"Starting episode {task_episodes+1} (trial index {episode_idx})...\n")
            episode_inference_ms = []
            episode_profile_rows = []
            episode_done = False
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        env_t0 = time.perf_counter()
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        timing_totals["env_step_seconds"] += (time.perf_counter() - env_t0)
                        replay_images.append(get_libero_video_frame(obs))
                        t += 1
                        continue

                    # Get preprocessed images
                    obs_prep_t0 = time.perf_counter()
                    images = get_libero_images(obs, resize_size, use_anyres=cfg.use_anyres)

                    # Save the current simulator frame before executing the predicted action chunk.
                    replay_images.append(get_libero_video_frame(obs))

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "image_primary": images["image_primary"],
                        "image_wrist": images["image_wrist"] if cfg.use_wrist_image else None,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    timing_totals["obs_preprocess_seconds"] += (time.perf_counter() - obs_prep_t0)

                    # Query model to get action
                    t_start = time.perf_counter()
                    action = model.inference(
                        observation["image_primary"],
                        observation["image_wrist"],
                        task_description,
                        cfg.task_suite_name,
                    )
                    t_end = time.perf_counter()
                    # Same behavior as OpenVLA script: skip warmup by excluding the
                    # first measured call in an episode (t == num_steps_wait).
                    if t > cfg.num_steps_wait:
                        query_ms = (t_end - t_start) * 1000.0
                        GLOBAL_INFERENCE_TIME += (t_end - t_start)
                        GLOBAL_INFERENCE_COUNT += 1
                        task_latency_ms.append(query_ms)
                        episode_inference_ms.append(query_ms)
                        profile = getattr(model, "last_inference_profile", None)
                        if profile:
                            timing_totals["inference_total_seconds"] += float(profile.get("inference_total_seconds", 0.0))
                            timing_totals["inference_preprocess_seconds"] += float(profile.get("preprocess_seconds", 0.0))
                            timing_totals["inference_generate_seconds"] += float(profile.get("generate_seconds", 0.0))
                            timing_totals["inference_decode_seconds"] += float(profile.get("decode_seconds", 0.0))
                            timing_totals["inference_vit_seconds"] += float(profile.get("vit_seconds", 0.0))
                            timing_totals["inference_prefill_seconds"] += float(profile.get("prefill_seconds", 0.0))
                            timing_totals["inference_decode_model_seconds"] += float(profile.get("decode_model_seconds", 0.0))
                            timing_totals["patch_total_seconds"] += float(profile.get("patch_total_seconds", 0.0))
                            timing_totals["force_attn_seconds"] += float(profile.get("force_attn_seconds", 0.0))
                            timing_totals["force_attn_self_attn_seconds"] += float(profile.get("force_attn_self_attn_seconds", 0.0))
                            timing_totals["force_attn_mlp_seconds"] += float(profile.get("force_attn_mlp_seconds", 0.0))
                            timing_totals["force_attn_residual_norm_seconds"] += float(profile.get("force_attn_residual_norm_seconds", 0.0))
                            timing_totals["regular_forward_seconds"] += float(profile.get("regular_forward_seconds", 0.0))
                            timing_totals["regular_self_attn_seconds"] += float(profile.get("regular_self_attn_seconds", 0.0))
                            timing_totals["regular_mlp_seconds"] += float(profile.get("regular_mlp_seconds", 0.0))
                            timing_totals["regular_residual_norm_seconds"] += float(profile.get("regular_residual_norm_seconds", 0.0))
                            timing_totals["manual_probe_seconds"] += float(profile.get("manual_probe_seconds", 0.0))
                            timing_totals["metric_seconds"] += float(profile.get("metric_seconds", 0.0))
                            timing_totals["rank_seconds"] += float(profile.get("rank_seconds", 0.0))
                            timing_totals["budget_schedule_seconds"] += float(profile.get("budget_schedule_seconds", 0.0))
                            timing_totals["topk_seconds"] += float(profile.get("topk_seconds", 0.0))
                            timing_totals["gather_seconds"] += float(profile.get("gather_seconds", 0.0))
                            timing_totals["compaction_bookkeeping_seconds"] += float(profile.get("compaction_bookkeeping_seconds", 0.0))
                            timing_totals["last_tokens_before"] += float(profile.get("last_tokens_before", 0.0))
                            timing_totals["last_tokens_after"] += float(profile.get("last_tokens_after", 0.0))
                            timing_totals["schedule_len"] += float(profile.get("schedule_len", 0.0))
                            timing_totals["pruned_layers"] += float(profile.get("pruned_layers", 0.0))
                            timing_totals["layout_ready"] += float(profile.get("layout_ready", 0.0))
                            timing_totals["layout_query_tokens"] += float(profile.get("layout_query_tokens", 0.0))
                            timing_totals["layout_view0_tokens"] += float(profile.get("layout_view0_tokens", 0.0))
                            timing_totals["layout_view1_tokens"] += float(profile.get("layout_view1_tokens", 0.0))
                            timing_totals["manual_attention_probe_layers"] += float(profile.get("manual_attention_probe_layers", 0.0))
                            timing_totals["manual_probe_qk_cache_hits"] += float(profile.get("manual_probe_qk_cache_hits", 0.0))
                            timing_totals["manual_probe_qk_cache_misses"] += float(profile.get("manual_probe_qk_cache_misses", 0.0))
                            timing_totals["manual_probe_reused_layers"] += float(profile.get("manual_probe_reused_layers", 0.0))
                            timing_totals["planned_compaction_layers"] += float(profile.get("planned_compaction_layers", 0.0))
                            timing_totals["generated_token_count"] += float(profile.get("generated_token_count", 0.0))
                            timing_totals["action_token_count"] += float(profile.get("action_token_count", 0.0))
                            timing_totals["decode_to_action_ratio"] += float(profile.get("decode_to_action_ratio", 0.0))
                            timing_totals["expected_coeff_count"] += float(profile.get("expected_coeff_count", 0.0))
                            timing_totals["layer0_total_calls"] += float(profile.get("layer0_total_calls", 0.0))
                            timing_totals["layer0_short_seq_calls"] += float(profile.get("layer0_short_seq_calls", 0.0))
                            timing_totals["layer0_long_seq_calls"] += float(profile.get("layer0_long_seq_calls", 0.0))
                            timing_totals["layer0_past_none_calls"] += float(profile.get("layer0_past_none_calls", 0.0))
                            timing_totals["layer0_past_present_calls"] += float(profile.get("layer0_past_present_calls", 0.0))
                            timing_totals["flops_reduction"] += float(profile.get("flops_reduction", 0.0))
                            timing_totals["decoder_layers_recorded"] += float(profile.get("decoder_layers_recorded", 0.0))
                            timing_totals["decoder_layers_missing"] += float(profile.get("decoder_layers_missing", 0.0))
                            timing_totals["final_visual_tokens"] += float(profile.get("final_visual_tokens", 0.0))
                            timing_totals["raw_total_budget"] += float(profile.get("raw_total_budget", 0.0))
                            timing_totals["active_pruning_layers"] += float(profile.get("active_pruning_layers", 0.0))
                            per_step_latency_rows.append({
                                "task_id": int(task_id),
                                "episode_idx": int(episode_idx),
                                "sim_step": int(t),
                                "query_latency_ms": float(query_ms),
                                "inference_total_seconds": float(profile.get("inference_total_seconds", 0.0)),
                                "preprocess_seconds": float(profile.get("preprocess_seconds", 0.0)),
                                "generate_seconds": float(profile.get("generate_seconds", 0.0)),
                                "decode_seconds": float(profile.get("decode_seconds", 0.0)),
                                "vit_seconds": float(profile.get("vit_seconds", 0.0)),
                                "prefill_seconds": float(profile.get("prefill_seconds", 0.0)),
                                "decode_model_seconds": float(profile.get("decode_model_seconds", 0.0)),
                                "patch_total_seconds": float(profile.get("patch_total_seconds", 0.0)),
                                "force_attn_seconds": float(profile.get("force_attn_seconds", 0.0)),
                                "force_attn_self_attn_seconds": float(profile.get("force_attn_self_attn_seconds", 0.0)),
                                "force_attn_mlp_seconds": float(profile.get("force_attn_mlp_seconds", 0.0)),
                                "force_attn_residual_norm_seconds": float(profile.get("force_attn_residual_norm_seconds", 0.0)),
                                "regular_forward_seconds": float(profile.get("regular_forward_seconds", 0.0)),
                                "regular_self_attn_seconds": float(profile.get("regular_self_attn_seconds", 0.0)),
                                "regular_mlp_seconds": float(profile.get("regular_mlp_seconds", 0.0)),
                                "regular_residual_norm_seconds": float(profile.get("regular_residual_norm_seconds", 0.0)),
                                "manual_probe_seconds": float(profile.get("manual_probe_seconds", 0.0)),
                                "metric_seconds": float(profile.get("metric_seconds", 0.0)),
                                "rank_seconds": float(profile.get("rank_seconds", 0.0)),
                                "budget_schedule_seconds": float(profile.get("budget_schedule_seconds", 0.0)),
                                "topk_seconds": float(profile.get("topk_seconds", 0.0)),
                                "gather_seconds": float(profile.get("gather_seconds", 0.0)),
                                "compaction_bookkeeping_seconds": float(profile.get("compaction_bookkeeping_seconds", 0.0)),
                                "last_tokens_before": float(profile.get("last_tokens_before", 0.0)),
                                "last_tokens_after": float(profile.get("last_tokens_after", 0.0)),
                                "schedule_len": float(profile.get("schedule_len", 0.0)),
                                "pruned_layers": float(profile.get("pruned_layers", 0.0)),
                                "layout_ready": float(profile.get("layout_ready", 0.0)),
                                "layout_query_tokens": float(profile.get("layout_query_tokens", 0.0)),
                                "layout_view0_tokens": float(profile.get("layout_view0_tokens", 0.0)),
                                "layout_view1_tokens": float(profile.get("layout_view1_tokens", 0.0)),
                                "manual_attention_probe_layers": float(profile.get("manual_attention_probe_layers", 0.0)),
                                "manual_probe_qk_cache_hits": float(profile.get("manual_probe_qk_cache_hits", 0.0)),
                                "manual_probe_qk_cache_misses": float(profile.get("manual_probe_qk_cache_misses", 0.0)),
                                "manual_probe_reused_layers": float(profile.get("manual_probe_reused_layers", 0.0)),
                                "planned_compaction_layers": float(profile.get("planned_compaction_layers", 0.0)),
                                "generated_token_count": float(profile.get("generated_token_count", 0.0)),
                                "action_token_count": float(profile.get("action_token_count", 0.0)),
                                "decode_to_action_ratio": float(profile.get("decode_to_action_ratio", 0.0)),
                                "expected_coeff_count": float(profile.get("expected_coeff_count", 0.0)),
                                "flops_reduction": float(profile.get("flops_reduction", 0.0)),
                                "decoder_layers_recorded": float(profile.get("decoder_layers_recorded", 0.0)),
                                "decoder_layers_missing": float(profile.get("decoder_layers_missing", 0.0)),
                                "final_visual_tokens": float(profile.get("final_visual_tokens", 0.0)),
                                "raw_total_budget": float(profile.get("raw_total_budget", 0.0)),
                                "active_pruning_layers": float(profile.get("active_pruning_layers", 0.0)),
                            })
                            episode_profile_rows.append(
                                {
                                    "sim_step": int(t),
                                    "query_latency_ms": float(query_ms),
                                    "profile": sanitize_for_json(profile),
                                }
                            )
                        else:
                            episode_profile_rows.append(
                                {
                                    "sim_step": int(t),
                                    "query_latency_ms": float(query_ms),
                                    "profile": None,
                                }
                            )
                    #print(action.shape)
                    

                    #Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    #print(action)
                    action = normalize_gripper_action(action, binarize=True)
                    
                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    #print("Action",action)
                    if cfg.model_family in {"openvla", "nora"}:
                        action = invert_gripper_action(action)
                        
                    
                    # Execute action in environment
                    action[..., -1] = np.where(action[..., -1] >= 0.0, 1.0, action[..., -1])

                    action_sequence = np.asarray(action)
                    if action_sequence.ndim == 1:
                        action_sequence = action_sequence[None, :]

                    for action_step in action_sequence:
                        env_t0 = time.perf_counter()
                        obs, reward, done, info = env.step(action_step.tolist())
                        timing_totals["env_step_seconds"] += (time.perf_counter() - env_t0)
                        replay_images.append(get_libero_video_frame(obs))
                        t += 1
                        if done:
                            task_successes += 1
                            total_successes += 1
                            episode_done = True
                            break

                    if done:
                        episode_done = True
                        break

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()
            task_result["episodes"].append(
                {
                    "episode_index": int(episode_idx),
                    "success": bool(episode_done),
                    "num_policy_queries": len(episode_inference_ms),
                    "query_latency_ms": sanitize_for_json(episode_inference_ms),
                    "latency_summary_ms": summarize_latency_ms(episode_inference_ms),
                    "query_profiles": sanitize_for_json(episode_profile_rows),
                }
            )

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        task_result["task_success_rate"] = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        task_result["num_episodes"] = int(task_episodes)
        task_result["latency_summary_ms"] = summarize_latency_ms(task_latency_ms)
        log_file.write(
            "Current task latency: "
            f"query_mean={task_result['latency_summary_ms']['mean_ms']:.2f} ms, "
            f"query_p90={task_result['latency_summary_ms']['p90_ms']:.2f} ms\n"
        )
        log_file.flush()
        task_results.append(task_result)
        if cfg.use_wandb:
            wandb.log(
            {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                warning_message = f"Warning: failed to close LIBERO env cleanly: {close_exc}"
                print(warning_message)
                log_file.write(warning_message + "\n")
                log_file.flush()
            finally:
                env = None
                gc.collect()

    if GLOBAL_INFERENCE_COUNT > 0:
        avg_lat_ms = (GLOBAL_INFERENCE_TIME / GLOBAL_INFERENCE_COUNT) * 1000.0
        print(f"Average Inference Latency: {avg_lat_ms:.2f} ms/step (Count: {GLOBAL_INFERENCE_COUNT})")
        log_file.write(f"Average Inference Latency: {avg_lat_ms:.2f} ms/step\n")
        avg_obs_prep_ms = (timing_totals["obs_preprocess_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_env_ms = (timing_totals["env_step_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_inf_total_ms = (timing_totals["inference_total_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_inf_prep_ms = (timing_totals["inference_preprocess_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_inf_gen_ms = (timing_totals["inference_generate_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_inf_decode_ms = (timing_totals["inference_decode_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_vit_ms = (timing_totals["inference_vit_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_prefill_ms = (timing_totals["inference_prefill_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_decode_model_ms = (timing_totals["inference_decode_model_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_stage_sum_ms = avg_vit_ms + avg_prefill_ms + avg_decode_model_ms
        avg_stage_residual_ms = avg_inf_gen_ms - avg_stage_sum_ms
        avg_patch_ms = (timing_totals["patch_total_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_force_attn_ms = (timing_totals["force_attn_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_force_attn_self_attn_ms = (timing_totals["force_attn_self_attn_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_force_attn_mlp_ms = (timing_totals["force_attn_mlp_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_force_attn_residual_norm_ms = (timing_totals["force_attn_residual_norm_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_regular_forward_ms = (timing_totals["regular_forward_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_regular_self_attn_ms = (timing_totals["regular_self_attn_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_regular_mlp_ms = (timing_totals["regular_mlp_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_regular_residual_norm_ms = (timing_totals["regular_residual_norm_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_manual_probe_ms = (timing_totals["manual_probe_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_metric_ms = (timing_totals["metric_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_rank_ms = (timing_totals["rank_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_budget_schedule_ms = (timing_totals["budget_schedule_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_topk_ms = (timing_totals["topk_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_gather_ms = (timing_totals["gather_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_compaction_bookkeeping_ms = (timing_totals["compaction_bookkeeping_seconds"] / GLOBAL_INFERENCE_COUNT) * 1000.0
        avg_last_tokens_before = timing_totals["last_tokens_before"] / GLOBAL_INFERENCE_COUNT
        avg_last_tokens_after = timing_totals["last_tokens_after"] / GLOBAL_INFERENCE_COUNT
        avg_schedule_len = timing_totals["schedule_len"] / GLOBAL_INFERENCE_COUNT
        avg_pruned_layers = timing_totals["pruned_layers"] / GLOBAL_INFERENCE_COUNT
        avg_layout_ready = timing_totals["layout_ready"] / GLOBAL_INFERENCE_COUNT
        avg_layout_query_tokens = timing_totals["layout_query_tokens"] / GLOBAL_INFERENCE_COUNT
        avg_layout_view0_tokens = timing_totals["layout_view0_tokens"] / GLOBAL_INFERENCE_COUNT
        avg_layout_view1_tokens = timing_totals["layout_view1_tokens"] / GLOBAL_INFERENCE_COUNT
        avg_manual_attention_probe_layers = timing_totals["manual_attention_probe_layers"] / GLOBAL_INFERENCE_COUNT
        avg_manual_probe_qk_cache_hits = timing_totals["manual_probe_qk_cache_hits"] / GLOBAL_INFERENCE_COUNT
        avg_manual_probe_qk_cache_misses = timing_totals["manual_probe_qk_cache_misses"] / GLOBAL_INFERENCE_COUNT
        avg_manual_probe_reused_layers = timing_totals["manual_probe_reused_layers"] / GLOBAL_INFERENCE_COUNT
        avg_planned_compaction_layers = timing_totals["planned_compaction_layers"] / GLOBAL_INFERENCE_COUNT
        avg_generated_token_count = timing_totals["generated_token_count"] / GLOBAL_INFERENCE_COUNT
        avg_action_token_count = timing_totals["action_token_count"] / GLOBAL_INFERENCE_COUNT
        avg_decode_to_action_ratio = timing_totals["decode_to_action_ratio"] / GLOBAL_INFERENCE_COUNT
        avg_expected_coeff_count = timing_totals["expected_coeff_count"] / GLOBAL_INFERENCE_COUNT
        avg_layer0_total_calls = timing_totals["layer0_total_calls"] / GLOBAL_INFERENCE_COUNT
        avg_layer0_short_seq_calls = timing_totals["layer0_short_seq_calls"] / GLOBAL_INFERENCE_COUNT
        avg_layer0_long_seq_calls = timing_totals["layer0_long_seq_calls"] / GLOBAL_INFERENCE_COUNT
        avg_layer0_past_none_calls = timing_totals["layer0_past_none_calls"] / GLOBAL_INFERENCE_COUNT
        avg_layer0_past_present_calls = timing_totals["layer0_past_present_calls"] / GLOBAL_INFERENCE_COUNT
        avg_flops_reduction = timing_totals["flops_reduction"] / GLOBAL_INFERENCE_COUNT
        avg_decoder_layers_recorded = timing_totals["decoder_layers_recorded"] / GLOBAL_INFERENCE_COUNT
        avg_decoder_layers_missing = timing_totals["decoder_layers_missing"] / GLOBAL_INFERENCE_COUNT
        avg_final_visual_tokens = timing_totals["final_visual_tokens"] / GLOBAL_INFERENCE_COUNT
        avg_raw_total_budget = timing_totals["raw_total_budget"] / GLOBAL_INFERENCE_COUNT
        avg_active_pruning_layers = timing_totals["active_pruning_layers"] / GLOBAL_INFERENCE_COUNT

        print("Latency Breakdown (avg ms per measured policy step):")
        print(f"- Obs preprocess (outside model): {avg_obs_prep_ms:.2f}")
        print(f"- Env step (all env.step calls / measured step): {avg_env_ms:.2f}")
        print(f"- Inference total (Nora.inference): {avg_inf_total_ms:.2f}")
        print(f"  - Inference preprocess (chat template + processor): {avg_inf_prep_ms:.2f}")
        print(f"  - Inference generate (model.generate): {avg_inf_gen_ms:.2f}")
        print(f"  - Inference decode (action decode + unnorm): {avg_inf_decode_ms:.2f}")
        print("  - Generate stage split (model internals):")
        print(f"    - T_ViT: {avg_vit_ms:.2f}")
        print(f"    - T_Prefill (LLM prefill only): {avg_prefill_ms:.2f}")
        print(f"    - T_Decode (LLM decode loop): {avg_decode_model_ms:.2f}")
        print(f"    - Stage sum (ViT+Prefill+Decode): {avg_stage_sum_ms:.2f}")
        print(f"    - Generate residual (generate - stage sum): {avg_stage_residual_ms:.2f}")
        print(f"- Patch total (all patched layers): {avg_patch_ms:.2f}")
        print(f"  - Regular layer forward time: {avg_regular_forward_ms:.2f}")
        print(f"    - self_attn kernel+proj time: {avg_regular_self_attn_ms:.2f}")
        print(f"    - mlp time: {avg_regular_mlp_ms:.2f}")
        print(f"    - residual/norm/other time: {avg_regular_residual_norm_ms:.2f}")
        print(f"  - Forced attentions forward time: {avg_force_attn_ms:.2f}")
        print(f"    - self_attn kernel+proj time: {avg_force_attn_self_attn_ms:.2f}")
        print(f"    - mlp time: {avg_force_attn_mlp_ms:.2f}")
        print(f"    - residual/norm/other time: {avg_force_attn_residual_norm_ms:.2f}")
        print(f"  - Manual probe time: {avg_manual_probe_ms:.2f}")
        print(f"  - MI/KL/NMI time: {avg_metric_ms:.2f}")
        print(f"  - Effective-rank time: {avg_rank_ms:.2f}")
        print(f"  - Budget/schedule time: {avg_budget_schedule_ms:.2f}")
        print(f"  - TopK time: {avg_topk_ms:.2f}")
        print(f"  - Gather time: {avg_gather_ms:.2f}")
        print(f"  - Compaction bookkeeping time: {avg_compaction_bookkeeping_ms:.2f}")
        print(f"  - Avg visual tokens before prune: {avg_last_tokens_before:.2f}")
        print(f"  - Avg visual tokens after prune: {avg_last_tokens_after:.2f}")
        print(f"  - Avg final visual tokens: {avg_final_visual_tokens:.2f}")
        print(f"  - Avg raw total budget: {avg_raw_total_budget:.2f}")
        print(f"  - Avg schedule length: {avg_schedule_len:.2f}")
        print(f"  - Avg pruned layers: {avg_pruned_layers:.2f}")
        print(f"  - Avg active pruning layers: {avg_active_pruning_layers:.2f}")
        print(f"  - Avg FLOPs reduction: {avg_flops_reduction:.4f}")
        print(f"  - Avg decoder layers recorded: {avg_decoder_layers_recorded:.2f}")
        print(f"  - Avg decoder layers missing: {avg_decoder_layers_missing:.2f}")
        print("  - Layout diagnostics:")
        print(f"    - layout ready: {avg_layout_ready:.2f}")
        print(f"    - query tokens: {avg_layout_query_tokens:.2f}")
        print(f"    - view0 image tokens: {avg_layout_view0_tokens:.2f}")
        print(f"    - view1 image tokens: {avg_layout_view1_tokens:.2f}")
        print(f"    - manual attention probe layers: {avg_manual_attention_probe_layers:.2f}")
        print(f"    - manual probe q/k cache hits: {avg_manual_probe_qk_cache_hits:.2f}")
        print(f"    - manual probe q/k cache misses: {avg_manual_probe_qk_cache_misses:.2f}")
        print(f"    - manual probe reused layers: {avg_manual_probe_reused_layers:.2f}")
        print(f"    - planned compaction layers: {avg_planned_compaction_layers:.2f}")
        print("  - Decode token diagnostics:")
        print(f"    - generated token count: {avg_generated_token_count:.2f}")
        print(f"    - action token count: {avg_action_token_count:.2f}")
        print(f"    - decode/action ratio: {avg_decode_to_action_ratio:.2f}")
        print(f"    - expected coeff count: {avg_expected_coeff_count:.2f}")
        print("- AutoPrune layer0 diagnostics (avg counts per measured step):")
        print(f"  - layer0 total calls: {avg_layer0_total_calls:.2f}")
        print(f"  - layer0 short-seq calls (<=96): {avg_layer0_short_seq_calls:.2f}")
        print(f"  - layer0 long-seq calls (>96): {avg_layer0_long_seq_calls:.2f}")
        print(f"  - layer0 past_key_value is None: {avg_layer0_past_none_calls:.2f}")
        print(f"  - layer0 past_key_value present: {avg_layer0_past_present_calls:.2f}")

        log_file.write("Latency Breakdown (avg ms per measured policy step):\n")
        log_file.write(f"- Obs preprocess (outside model): {avg_obs_prep_ms:.2f}\n")
        log_file.write(f"- Env step (all env.step calls / measured step): {avg_env_ms:.2f}\n")
        log_file.write(f"- Inference total (Nora.inference): {avg_inf_total_ms:.2f}\n")
        log_file.write(f"  - Inference preprocess (chat template + processor): {avg_inf_prep_ms:.2f}\n")
        log_file.write(f"  - Inference generate (model.generate): {avg_inf_gen_ms:.2f}\n")
        log_file.write(f"  - Inference decode (action decode + unnorm): {avg_inf_decode_ms:.2f}\n")
        log_file.write("  - Generate stage split (model internals):\n")
        log_file.write(f"    - T_ViT: {avg_vit_ms:.2f}\n")
        log_file.write(f"    - T_Prefill (LLM prefill only): {avg_prefill_ms:.2f}\n")
        log_file.write(f"    - T_Decode (LLM decode loop): {avg_decode_model_ms:.2f}\n")
        log_file.write(f"    - Stage sum (ViT+Prefill+Decode): {avg_stage_sum_ms:.2f}\n")
        log_file.write(f"    - Generate residual (generate - stage sum): {avg_stage_residual_ms:.2f}\n")
        log_file.write(f"- Patch total (all patched layers): {avg_patch_ms:.2f}\n")
        log_file.write(f"  - Regular layer forward time: {avg_regular_forward_ms:.2f}\n")
        log_file.write(f"    - self_attn kernel+proj time: {avg_regular_self_attn_ms:.2f}\n")
        log_file.write(f"    - mlp time: {avg_regular_mlp_ms:.2f}\n")
        log_file.write(f"    - residual/norm/other time: {avg_regular_residual_norm_ms:.2f}\n")
        log_file.write(f"  - Forced attentions forward time: {avg_force_attn_ms:.2f}\n")
        log_file.write(f"    - self_attn kernel+proj time: {avg_force_attn_self_attn_ms:.2f}\n")
        log_file.write(f"    - mlp time: {avg_force_attn_mlp_ms:.2f}\n")
        log_file.write(f"    - residual/norm/other time: {avg_force_attn_residual_norm_ms:.2f}\n")
        log_file.write(f"  - Manual probe time: {avg_manual_probe_ms:.2f}\n")
        log_file.write(f"  - MI/KL/NMI time: {avg_metric_ms:.2f}\n")
        log_file.write(f"  - Effective-rank time: {avg_rank_ms:.2f}\n")
        log_file.write(f"  - Budget/schedule time: {avg_budget_schedule_ms:.2f}\n")
        log_file.write(f"  - TopK time: {avg_topk_ms:.2f}\n")
        log_file.write(f"  - Gather time: {avg_gather_ms:.2f}\n")
        log_file.write(f"  - Compaction bookkeeping time: {avg_compaction_bookkeeping_ms:.2f}\n")
        log_file.write(f"  - Avg visual tokens before prune: {avg_last_tokens_before:.2f}\n")
        log_file.write(f"  - Avg visual tokens after prune: {avg_last_tokens_after:.2f}\n")
        log_file.write(f"  - Avg final visual tokens: {avg_final_visual_tokens:.2f}\n")
        log_file.write(f"  - Avg raw total budget: {avg_raw_total_budget:.2f}\n")
        log_file.write(f"  - Avg schedule length: {avg_schedule_len:.2f}\n")
        log_file.write(f"  - Avg pruned layers: {avg_pruned_layers:.2f}\n")
        log_file.write(f"  - Avg active pruning layers: {avg_active_pruning_layers:.2f}\n")
        log_file.write(f"  - Avg FLOPs reduction: {avg_flops_reduction:.4f}\n")
        log_file.write(f"  - Avg decoder layers recorded: {avg_decoder_layers_recorded:.2f}\n")
        log_file.write(f"  - Avg decoder layers missing: {avg_decoder_layers_missing:.2f}\n")
        log_file.write("  - Layout diagnostics:\n")
        log_file.write(f"    - layout ready: {avg_layout_ready:.2f}\n")
        log_file.write(f"    - query tokens: {avg_layout_query_tokens:.2f}\n")
        log_file.write(f"    - view0 image tokens: {avg_layout_view0_tokens:.2f}\n")
        log_file.write(f"    - view1 image tokens: {avg_layout_view1_tokens:.2f}\n")
        log_file.write(f"    - manual attention probe layers: {avg_manual_attention_probe_layers:.2f}\n")
        log_file.write(f"    - manual probe q/k cache hits: {avg_manual_probe_qk_cache_hits:.2f}\n")
        log_file.write(f"    - manual probe q/k cache misses: {avg_manual_probe_qk_cache_misses:.2f}\n")
        log_file.write(f"    - manual probe reused layers: {avg_manual_probe_reused_layers:.2f}\n")
        log_file.write(f"    - planned compaction layers: {avg_planned_compaction_layers:.2f}\n")
        log_file.write("  - Decode token diagnostics:\n")
        log_file.write(f"    - generated token count: {avg_generated_token_count:.2f}\n")
        log_file.write(f"    - action token count: {avg_action_token_count:.2f}\n")
        log_file.write(f"    - decode/action ratio: {avg_decode_to_action_ratio:.2f}\n")
        log_file.write(f"    - expected coeff count: {avg_expected_coeff_count:.2f}\n")
        log_file.write("- AutoPrune layer0 diagnostics (avg counts per measured step):\n")
        log_file.write(f"  - layer0 total calls: {avg_layer0_total_calls:.2f}\n")
        log_file.write(f"  - layer0 short-seq calls (<=96): {avg_layer0_short_seq_calls:.2f}\n")
        log_file.write(f"  - layer0 long-seq calls (>96): {avg_layer0_long_seq_calls:.2f}\n")
        log_file.write(f"  - layer0 past_key_value is None: {avg_layer0_past_none_calls:.2f}\n")
        log_file.write(f"  - layer0 past_key_value present: {avg_layer0_past_present_calls:.2f}\n")
        log_file.flush()

    overall_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    latency_summary_ms = summarize_latency_ms([row.get("query_latency_ms", 0.0) for row in per_step_latency_rows])
    log_file.write(f"Overall success rate: {overall_success_rate:.4f} ({overall_success_rate * 100:.1f}%)\n")
    log_file.write(
        "Latency summary: "
        f"query_mean={latency_summary_ms['mean_ms']:.2f} ms, "
        f"query_p90={latency_summary_ms['p90_ms']:.2f} ms, "
        f"query_p95={latency_summary_ms['p95_ms']:.2f} ms\n"
    )
    log_file.flush()

    # Save local log file
    log_file.close()

    if cfg.latency_csv_filename:
        csv_path = os.path.join(cfg.local_log_dir, cfg.latency_csv_filename)
        if per_step_latency_rows:
            fieldnames = list(per_step_latency_rows[0].keys())
            with open(csv_path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_step_latency_rows)
            print(f"Saved per-step latency CSV to: {csv_path}")
        else:
            print("No per-step latency rows to write.")

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": overall_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    summary_payload = {
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "overall_success_rate": float(overall_success_rate),
    }

    results_payload = {
        "run_id": run_id,
        "config": sanitize_for_json(asdict(cfg)),
        # Keep both keys for downstream compatibility. Some older consumers
        # expect "metrics", while the newer scripts read "summary".
        "summary": sanitize_for_json(summary_payload),
        "metrics": sanitize_for_json(summary_payload),
        "latency_summary_ms": sanitize_for_json(latency_summary_ms),
        "timing_totals_seconds": sanitize_for_json(timing_totals),
        "global_inference_count": int(GLOBAL_INFERENCE_COUNT),
        "tasks": sanitize_for_json(task_results),
    }
    with open(results_json_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved structured results JSON to: {results_json_path}")


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    run_eval_libero(cfg)


def parse_args() -> GenerateConfig:
    defaults = GenerateConfig()
    parser = argparse.ArgumentParser(description="Run NORA evaluation in LIBERO.")
    parser.add_argument("--model_family", type=str, default=defaults.model_family)
    parser.add_argument("--pretrained_checkpoint", type=str, default=str(defaults.pretrained_checkpoint))
    parser.add_argument("--model_path", type=str, default=str(defaults.model_path))
    parser.add_argument("--lora_adapter_path", type=str, default=defaults.lora_adapter_path)
    parser.add_argument("--full_model_state_path", type=str, default=defaults.full_model_state_path)
    parser.add_argument("--load_in_8bit", type=parse_bool, default=defaults.load_in_8bit)
    parser.add_argument("--load_in_4bit", type=parse_bool, default=defaults.load_in_4bit)
    parser.add_argument("--action_horizon", type=int, default=defaults.action_horizon)
    parser.add_argument("--center_crop", type=parse_bool, default=defaults.center_crop)
    parser.add_argument("--task_suite_name", type=str, default=defaults.task_suite_name)
    parser.add_argument("--num_steps_wait", type=int, default=defaults.num_steps_wait)
    parser.add_argument("--num_trials_per_task", type=int, default=defaults.num_trials_per_task)
    parser.add_argument("--trial_offset", type=int, default=defaults.trial_offset)
    parser.add_argument("--num_tasks_to_run", type=int, default=defaults.num_tasks_to_run)
    parser.add_argument("--image_resize_resolution", type=parse_resize_resolution, default=defaults.image_resize_resolution)
    parser.add_argument("--env_camera_resolution", type=int, default=defaults.env_camera_resolution)
    parser.add_argument("--use_wrist_image", type=parse_bool, default=defaults.use_wrist_image)
    parser.add_argument("--dual_view_mode", type=str, default=defaults.dual_view_mode)
    parser.add_argument("--use_anyres", type=parse_bool, default=defaults.use_anyres)
    parser.add_argument("--force_eager_attention", type=parse_bool, default=defaults.force_eager_attention)
    parser.add_argument("--force_output_attentions", type=parse_bool, default=defaults.force_output_attentions)
    parser.add_argument("--enable_stage_profiling", type=parse_bool, default=defaults.enable_stage_profiling)
    parser.add_argument("--stop_at_expected_action_length", type=parse_bool, default=defaults.stop_at_expected_action_length)
    parser.add_argument("--enable_autoprune", type=parse_bool, default=defaults.enable_autoprune)
    parser.add_argument("--autoprune_preset", type=str, default=defaults.autoprune_preset)
    parser.add_argument("--autoprune_visual_tokens", type=int, default=defaults.autoprune_visual_tokens)
    parser.add_argument("--autoprune_start_layer", type=int, default=defaults.autoprune_start_layer)
    parser.add_argument("--autoprune_num_views", type=int, default=defaults.autoprune_num_views)
    parser.add_argument("--autoprune_allocation_temp", type=float, default=defaults.autoprune_allocation_temp)
    parser.add_argument("--autoprune_x0", type=float, default=defaults.autoprune_x0)
    parser.add_argument("--autoprune_k_shape", type=float, default=defaults.autoprune_k_shape)
    parser.add_argument("--autoprune_b_min", type=int, default=defaults.autoprune_b_min)
    parser.add_argument("--autoprune_b_max", type=int, default=defaults.autoprune_b_max)
    parser.add_argument("--autoprune_fixed_total_budget", type=int, default=defaults.autoprune_fixed_total_budget)
    parser.add_argument("--autoprune_target_token_num", type=int, default=defaults.autoprune_target_token_num)
    parser.add_argument("--autoprune_mu", type=float, default=defaults.autoprune_mu)
    parser.add_argument("--autoprune_tau", type=float, default=defaults.autoprune_tau)
    parser.add_argument("--autoprune_min_keep_per_view", type=int, default=defaults.autoprune_min_keep_per_view)
    parser.add_argument("--autoprune_budget_mode", type=str, default=defaults.autoprune_budget_mode)
    parser.add_argument("--autoprune_absolute_anchor_mode", type=str, default=defaults.autoprune_absolute_anchor_mode)
    parser.add_argument("--autoprune_absolute_budget_scale", type=float, default=defaults.autoprune_absolute_budget_scale)
    parser.add_argument("--autoprune_absolute_auto_bounds", type=parse_bool, default=defaults.autoprune_absolute_auto_bounds)
    parser.add_argument("--autoprune_enable_adaptive_layer_policy", type=parse_bool, default=defaults.autoprune_enable_adaptive_layer_policy)
    parser.add_argument("--autoprune_enable_oneshot_fallback", type=parse_bool, default=defaults.autoprune_enable_oneshot_fallback)
    parser.add_argument("--autoprune_adaptive_layer_choices", type=parse_int_tuple, default=defaults.autoprune_adaptive_layer_choices)
    parser.add_argument("--autoprune_adaptive_layer_mu", type=float, default=defaults.autoprune_adaptive_layer_mu)
    parser.add_argument("--autoprune_adaptive_layer_tau", type=float, default=defaults.autoprune_adaptive_layer_tau)
    parser.add_argument("--autoprune_oneshot_fallback_max_layers", type=int, default=defaults.autoprune_oneshot_fallback_max_layers)
    parser.add_argument("--autoprune_oneshot_fallback_budget_margin", type=float, default=defaults.autoprune_oneshot_fallback_budget_margin)
    parser.add_argument("--autoprune_oneshot_fallback_frontload_ratio", type=float, default=defaults.autoprune_oneshot_fallback_frontload_ratio)
    parser.add_argument("--autoprune_monitor_only", type=parse_bool, default=defaults.autoprune_monitor_only)
    parser.add_argument("--autoprune_disable_token_prune", type=parse_bool, default=defaults.autoprune_disable_token_prune)
    parser.add_argument("--autoprune_disable_progressive_prune", type=parse_bool, default=defaults.autoprune_disable_progressive_prune)
    parser.add_argument("--autoprune_static_drop_tokens_at_start", type=int, default=defaults.autoprune_static_drop_tokens_at_start)
    parser.add_argument("--autoprune_prefer_flash_attention", type=parse_bool, default=defaults.autoprune_prefer_flash_attention)
    parser.add_argument("--autoprune_manual_attention_probe", type=parse_bool, default=defaults.autoprune_manual_attention_probe)
    parser.add_argument("--autoprune_manual_attention_probe_stride", type=int, default=defaults.autoprune_manual_attention_probe_stride)
    parser.add_argument("--autoprune_manual_attention_probe_max_layers", type=int, default=defaults.autoprune_manual_attention_probe_max_layers)
    parser.add_argument("--autoprune_prune_only_on_anchor_layers", type=parse_bool, default=defaults.autoprune_prune_only_on_anchor_layers)
    parser.add_argument("--autoprune_piecewise_linear_keep_schedule", type=parse_bool, default=defaults.autoprune_piecewise_linear_keep_schedule)
    parser.add_argument("--autoprune_piecewise_linear_hard_final_cap", type=parse_bool, default=defaults.autoprune_piecewise_linear_hard_final_cap)
    parser.add_argument("--autoprune_budget_semantics", type=str, default=defaults.autoprune_budget_semantics)
    parser.add_argument("--autoprune_budget_aware_x0_shift", type=float, default=defaults.autoprune_budget_aware_x0_shift)
    parser.add_argument("--autoprune_compaction_delta_threshold_tokens", type=int, default=defaults.autoprune_compaction_delta_threshold_tokens)
    parser.add_argument("--autoprune_physical_compaction_max_shots", type=int, default=defaults.autoprune_physical_compaction_max_shots)
    parser.add_argument("--autoprune_detailed_timing", type=parse_bool, default=defaults.autoprune_detailed_timing)
    parser.add_argument("--autoprune_debug", type=parse_bool, default=defaults.autoprune_debug)
    parser.add_argument("--enable_compare_pruning", type=parse_bool, default=defaults.enable_compare_pruning)
    parser.add_argument("--compare_pruning_method", type=str, default=defaults.compare_pruning_method)
    parser.add_argument("--compare_visual_tokens", type=int, default=defaults.compare_visual_tokens)
    parser.add_argument("--compare_num_views", type=int, default=defaults.compare_num_views)
    parser.add_argument("--compare_fastv_pruning_layer", type=int, default=defaults.compare_fastv_pruning_layer)
    parser.add_argument("--compare_sparsevlm_pruning_start_layer", type=int, default=defaults.compare_sparsevlm_pruning_start_layer)
    parser.add_argument("--compare_sparsevlm_lambda", type=float, default=defaults.compare_sparsevlm_lambda)
    parser.add_argument("--compare_sparsevlm_recycle_top_ratio", type=float, default=defaults.compare_sparsevlm_recycle_top_ratio)
    parser.add_argument("--compare_sparsevlm_cluster_ratio", type=float, default=defaults.compare_sparsevlm_cluster_ratio)
    parser.add_argument("--compare_sparsevlm_rank_tolerance", type=float, default=defaults.compare_sparsevlm_rank_tolerance)
    parser.add_argument("--compare_sparsevlm_knn_k", type=int, default=defaults.compare_sparsevlm_knn_k)
    parser.add_argument("--compare_detailed_timing", type=parse_bool, default=defaults.compare_detailed_timing)
    parser.add_argument("--compare_debug", type=parse_bool, default=defaults.compare_debug)
    parser.add_argument("--enable_baseline_pruning", type=parse_bool, default=defaults.enable_baseline_pruning)
    parser.add_argument("--baseline_pruning_method", type=str, default=defaults.baseline_pruning_method)
    parser.add_argument("--baseline_target_token_num", type=int, default=defaults.baseline_target_token_num)
    parser.add_argument("--baseline_visual_tokens", type=int, default=defaults.baseline_visual_tokens)
    parser.add_argument("--baseline_start_layer", type=int, default=defaults.baseline_start_layer)
    parser.add_argument("--baseline_max_pruning_layers", type=int, default=defaults.baseline_max_pruning_layers)
    parser.add_argument("--baseline_layer_indices", type=parse_int_tuple, default=defaults.baseline_layer_indices)
    parser.add_argument("--baseline_visionzip_dominant_ratio", type=float, default=defaults.baseline_visionzip_dominant_ratio)
    parser.add_argument("--baseline_debug", type=parse_bool, default=defaults.baseline_debug)
    parser.add_argument("--run_id_note", type=str, default=defaults.run_id_note)
    parser.add_argument("--local_log_dir", type=str, default=defaults.local_log_dir)
    parser.add_argument("--latency_csv_filename", type=str, default=defaults.latency_csv_filename)
    parser.add_argument("--use_wandb", type=parse_bool, default=defaults.use_wandb)
    parser.add_argument("--wandb_project", type=str, default=defaults.wandb_project)
    parser.add_argument("--wandb_entity", type=str, default=defaults.wandb_entity)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()
    return GenerateConfig(**vars(args))


if __name__ == "__main__":
    run_eval_libero(parse_args())
