"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import json
import logging
import os
import sys
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
##autoprune
sys.path.append(os.getcwd())
from autoprune_patch_improved_1 import AutoPrunePatchImproved as AutoPrunePatchLegacy
from vlapruner_oft_patch import VLAPrunerOFTConfig, VLAPrunerOFTPatch
from vlaadp_oft_patch import VLAADPOFTConfig, VLAADPOFTPatch
try:
    from autoprune_patch_improved import AutoPrunePatchImproved as AutoPrunePatchModern
except Exception:
    AutoPrunePatchModern = AutoPrunePatchLegacy
try:
    from compare_pruning_patch import ComparePruningConfig, ComparePruningPatch
except Exception:
    ComparePruningConfig = None
    ComparePruningPatch = None
import wandb
import time

# Global Latency Tracker
LATENCY_TRACKER = []
MODEL_LATENCY_TRACKER = []
DECODER_LATENCY_TRACKER = []
PRUNING_LAYER_LATENCY_TRACKER = []
AUTOPRUNE_STATS_TRACKER = []

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # AutoPrune parameters (improved patch)
    #################################################################################################################
    use_auto_prune: bool = False                     # Whether to enable AutoPrune token pruning
    target_token_num: int = 64                       # Target number of tokens to keep after pruning
    visual_tokens: int = 512                         # Number of visual tokens (usually 256 * num_images_in_input)
    x0: float = 14.9                                 # Horizontal shift parameter for logistic curve
    k0: float = 0.4                                  # Base slope parameter
    gamma: float = 0.2                               # Sensitivity parameter
    pruning_start_layer: int = 2                     # Layer to start applying pruning
    text_sample_ratio: float = 0.25                  # Ratio of text tokens used to estimate MI
    safe_ratio: float = 0.2                          # Ratio of tokens kept as safety window
    min_keep_per_view: int = 24                      # Minimum kept tokens per view
    budget_mode: str = "absolute"                    # Budget mode: absolute / legacy variants
    absolute_anchor_mode: str = "none"               # Anchor mode for absolute budget: none / elbow
    absolute_budget_scale: float = 1.0               # Single coarse scaling knob across models
    absolute_auto_bounds: bool = True                # Whether to use data-dependent auto bounds
    enable_adaptive_layer_policy: bool = False       # Compress dense schedule to fewer execution layers
    adaptive_layer_choices: tuple[int, ...] = (1, 3, 6)  # Candidate execution-layer counts
    adaptive_layer_mu: float = 4.0                   # Logistic center for layer-count selection
    adaptive_layer_tau: float = 0.5                  # Logistic slope for layer-count selection
    enable_oneshot_fallback: bool = False            # Collapse to one-shot when schedule front-loads enough
    oneshot_fallback_max_layers: int = 3             # Only allow fallback when compressed schedule is short
    oneshot_fallback_budget_margin: float = 0.15     # Near-target margin for one-shot fallback
    oneshot_fallback_frontload_ratio: float = 0.5    # Minimum front-loaded drop ratio for one-shot fallback
    autoprune_patch_variant: str = "legacy_budget"   # modern / legacy_budget

    #################################################################################################################
    # Compare pruning baselines
    #################################################################################################################
    use_compare_pruning: bool = False                # Whether to enable imported pruning baselines
    compare_pruning_method: str = "visionzip"        # visionzip / fastervlm / pyramiddrop / fastv / sparsevlm
    compare_visual_token_num: int = 128              # Final kept visual token budget for one-shot baselines
    compare_important_ratio: float = 0.0             # <=0 means use method-specific default ratio
    compare_layer_list: tuple[int, ...] = (8, 16, 24)  # PyramidDrop pruning layers (1-based)
    compare_image_token_ratio_list: tuple[float, ...] = (0.75, 0.5, 0.25)  # PyramidDrop retained ratios
    compare_fastv_pruning_layer: int = 2             # FastV pruning layer K (1-based, prune after this layer)
    compare_sparsevlm_pruning_start_layer: int = 1   # SparseVLM pruning start layer (1-based)
    compare_sparsevlm_lambda: float = 1.0            # SparseVLM Eq.(7) deletion scale lambda
    compare_sparsevlm_recycle_top_ratio: float = 0.5 # SparseVLM recycle top-tau ratio from deleted tokens
    compare_sparsevlm_cluster_ratio: float = 0.25    # SparseVLM theta: fraction of recycled pool used as centers
    compare_sparsevlm_rank_tolerance: float = 1e-4   # Numerical threshold for rank(P)
    compare_sparsevlm_knn_k: int = 8                 # k-NN size for density-peak recycling
    compare_enable_attention_probe: bool = True      # Debug knob: disable vision attention probes for compare methods
    compare_patch_scope: str = "full"                # full / vision_only / regression_only / passthrough

    #################################################################################################################
    # VLA-Pruner on OpenVLA-OFT (unified visual token pool, no per-view allocation)
    #################################################################################################################
    use_vlapruner_oft: bool = False                  # Enable VLA-Pruner adapted to OpenVLA-OFT
    vlapruner_keep_visual_tokens: int = 224          # Target average final visual tokens, not pruning ratio
    vlapruner_pruning_layer: int = 3                 # Paper K; 1-based layer, prune after this layer
    vlapruner_temporal_window: int = 3               # Paper temporal window w
    vlapruner_temporal_gamma: float = 0.8            # Paper temporal decay gamma
    vlapruner_warmup_steps: int = 3                  # Warm-start steps before pruning
    vlapruner_action_attention_layer_start: str = "half"  # "half" or 0-based layer index

    #################################################################################################################
    # VLA-ADP on OpenVLA-OFT (pre-LLM QK pruning, per-view 4:6 main/wrist allocation)
    #################################################################################################################
    use_vlaadp_oft: bool = False                    # Enable VLA-ADP adapted to OpenVLA-OFT
    vlaadp_keep_visual_tokens: int = 224            # Total visual tokens kept across all views
    vlaadp_qk_layer: int = 0                        # Paper default: use layer-0 Q/K before LLM forward
    vlaadp_main_view_weight: float = 0.4            # Paper multi-view split: main camera share
    vlaadp_wrist_view_weight: float = 0.6           # Paper multi-view split: wrist camera share
    vlaadp_use_dynamic_visual_strategy: bool = True # Use previous action trajectory to gate pruning
    vlaadp_decision_method: str = "adjacent"        # adjacent / avg
    vlaadp_adjacent_variant: str = "extrema"        # Kept for compatibility with ADP config wording
    vlaadp_adjacent_extrema_window: int = 3         # Compare current window delta with previous extrema
    vlaadp_adjacent_lookback: int = 2               # Compatibility knob; extrema_window-1 is used by default
    vlaadp_adjacent_last_state: bool = True         # Paper Eq. 18: middle band keeps previous state
    vlaadp_initial_state: int = 0                   # 0 = full visual, 1 = prune
    vlaadp_delta_method: str = "net"                # net / sum
    vlaadp_min_delta_pos: float = 0.0
    vlaadp_min_delta_rot: float = 0.0
    vlaadp_hysteresis_up: float = 0.0
    vlaadp_hysteresis_down: float = 0.0
    vlaadp_tol_equal: float = 0.0
    vlaadp_cold_start_windows: int = 2              # First two action windows keep full visual tokens
    vlaadp_limit_consecutive_pruned_enabled: bool = True
    vlaadp_limit_max_consecutive_pruned: int = 3    # Force full vision after 3 consecutive pruned windows

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
    task_id: Optional[int] = None                    # If specified, only evaluate this task ID

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    # Validate AutoPrune parameters
    if cfg.use_auto_prune:
        assert cfg.visual_tokens > 0, "visual_tokens must be positive!"
        assert cfg.pruning_start_layer >= 0, "pruning_start_layer must be non-negative!"
        if cfg.budget_mode == "legacy":
            assert cfg.target_token_num > 0, "target_token_num must be positive in legacy budget mode!"
    if cfg.use_compare_pruning:
        assert ComparePruningConfig is not None and ComparePruningPatch is not None, "compare_pruning_patch.py is not available!"
        assert cfg.compare_visual_token_num > 0, "compare_visual_token_num must be positive!"
        if cfg.compare_pruning_method == "fastv":
            assert cfg.compare_fastv_pruning_layer > 0, "compare_fastv_pruning_layer must be positive for FastV!"
        if cfg.compare_pruning_method == "sparsevlm":
            assert cfg.compare_sparsevlm_pruning_start_layer > 0, "compare_sparsevlm_pruning_start_layer must be positive for SparseVLM!"
            assert cfg.compare_sparsevlm_lambda >= 0.0, "compare_sparsevlm_lambda must be non-negative!"
            assert cfg.compare_sparsevlm_recycle_top_ratio >= 0.0, "compare_sparsevlm_recycle_top_ratio must be non-negative!"
            assert cfg.compare_sparsevlm_cluster_ratio >= 0.0, "compare_sparsevlm_cluster_ratio must be non-negative!"
            assert cfg.compare_sparsevlm_knn_k > 0, "compare_sparsevlm_knn_k must be positive!"
    if cfg.use_vlapruner_oft:
        assert cfg.visual_tokens > 0, "visual_tokens must be positive for VLA-Pruner-OFT!"
        assert 0 < cfg.vlapruner_keep_visual_tokens <= cfg.visual_tokens, (
            "vlapruner_keep_visual_tokens must be in (0, visual_tokens]."
        )
        assert cfg.vlapruner_pruning_layer > 0, "vlapruner_pruning_layer is 1-based and must be positive!"
        assert cfg.vlapruner_temporal_window > 0, "vlapruner_temporal_window must be positive!"
    if cfg.use_vlaadp_oft:
        assert cfg.visual_tokens > 0, "visual_tokens must be positive for VLA-ADP-OFT!"
        assert 0 < cfg.vlaadp_keep_visual_tokens <= cfg.visual_tokens, (
            "vlaadp_keep_visual_tokens must be in (0, visual_tokens]."
        )
        assert cfg.num_images_in_input > 0, "num_images_in_input must be positive for VLA-ADP-OFT!"
        assert cfg.vlaadp_qk_layer >= 0, "vlaadp_qk_layer must be non-negative!"
        assert cfg.vlaadp_cold_start_windows >= 0, "vlaadp_cold_start_windows must be non-negative!"
        assert cfg.vlaadp_limit_max_consecutive_pruned > 0, "vlaadp_limit_max_consecutive_pruned must be positive!"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def average_tracker_value(items, key: str) -> Optional[float]:
    """Average a numeric key across a list of stats dictionaries."""
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def safe_ratio_pct(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Return percentage numerator / denominator, guarding against empty or zero values."""
    if numerator is None or denominator is None:
        return None
    denominator = float(denominator)
    if abs(denominator) <= 1e-9:
        return None
    return float(numerator) / denominator * 100.0


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    average_budget_horizon_steps = int(np.ceil(float(max_steps) / max(1, int(cfg.num_open_loop_steps))))
    vlapruner_ref = getattr(model, "_vlapruner_patch_ref", None)
    if vlapruner_ref is not None and hasattr(vlapruner_ref, "reset_episode_history"):
        vlapruner_ref.reset_episode_history(average_budget_horizon_steps=average_budget_horizon_steps)
    vlaadp_ref = getattr(model, "_vlaadp_patch_ref", None)
    if vlaadp_ref is not None and hasattr(vlaadp_ref, "reset_episode_history"):
        vlaadp_ref.reset_episode_history(average_budget_horizon_steps=average_budget_horizon_steps)

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK: # 检查配置里的 “开环步数” 是否和模型 / 算法预设的 “动作块长度” 一致
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
              f"({NUM_ACTIONS_CHUNK}) constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                if vlaadp_ref is not None and hasattr(vlaadp_ref, "observe_trajectory_step"):
                    try:
                        vlaadp_ref.observe_trajectory_step(obs["robot0_eef_pos"], obs["robot0_eef_quat"])
                    except Exception:
                        pass

                # Query model to get action
                # [LATENCY] Start Timer
                t0_inf = time.perf_counter()
                
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                
                # [LATENCY] End Timer & Record
                t_inf = (time.perf_counter() - t0_inf) * 1000.0
                LATENCY_TRACKER.append(t_inf)
                model_t_inf = getattr(model, "_last_predict_action_latency_ms", None)
                if model_t_inf is not None:
                    MODEL_LATENCY_TRACKER.append(float(model_t_inf))
                timing_stats = dict(getattr(model, "_last_action_timing_stats", {}) or {})

                patch_ref = getattr(model, "_autoprune_patch_ref", None)
                if patch_ref is None:
                    patch_ref = getattr(model, "_compare_patch_ref", None)
                if patch_ref is None:
                    patch_ref = getattr(model, "_vlapruner_patch_ref", None)
                if patch_ref is None:
                    patch_ref = getattr(model, "_vlaadp_patch_ref", None)
                if patch_ref is not None:
                    stats = dict(getattr(patch_ref, "latest_stats", {}) or {})
                    keep_count_schedule = stats.get("keep_count_schedule") or []
                    final_visual_tokens = int(stats.get("final_visual_tokens", 0) or 0)
                    if keep_count_schedule:
                        final_visual_tokens = int(sum(keep_count_schedule[-1]))

                    active_layers = stats.get("active_pruning_layers")
                    if isinstance(active_layers, list):
                        active_layers_count = int(len(active_layers))
                    else:
                        active_layers_count = int(stats.get("active_pruning_layers", 0) or 0)

                    AUTOPRUNE_STATS_TRACKER.append(
                        {
                            "latency_ms": float(t_inf),
                            "model_latency_ms": float(model_t_inf) if model_t_inf is not None else 0.0,
                            "image_prep_latency_ms": float(timing_stats.get("image_prep_latency_ms", 0.0) or 0.0),
                            "tokenization_latency_ms": float(timing_stats.get("tokenization_latency_ms", 0.0) or 0.0),
                            "proprio_latency_ms": float(timing_stats.get("proprio_latency_ms", 0.0) or 0.0),
                            "predict_action_latency_ms": float(timing_stats.get("predict_action_latency_ms", 0.0) or 0.0),
                            "total_action_latency_ms": float(timing_stats.get("total_action_latency_ms", 0.0) or 0.0),
                            "non_model_overhead_latency_ms": float(timing_stats.get("non_model_overhead_latency_ms", 0.0) or 0.0),
                            "raw_total_budget": float(stats.get("raw_total_budget", 0.0) or 0.0),
                            "active_pruning_layers_count": active_layers_count,
                            "final_visual_tokens": int(final_visual_tokens),
                            "complexity_signal": float(stats.get("complexity_signal", 0.0) or 0.0),
                            "view_keep_counts": list(stats.get("view_keep_counts", []) or []),
                            "view_score_masses": list(stats.get("view_score_masses", []) or []),
                            "decoder_layers_recorded": int(stats.get("decoder_layers_recorded", 0) or 0),
                            "decoder_layers_missing": int(stats.get("decoder_layers_missing", 0) or 0),
                            "multimodal_build_latency_ms": float(stats.get("multimodal_build_latency_ms", 0.0) or 0.0),
                            "prefill_latency_ms": float(stats.get("prefill_latency_ms", 0.0) or 0.0),
                            "decode_latency_ms": float(stats.get("decode_latency_ms", 0.0) or 0.0),
                            "decoder_forward_latency_ms": float(stats.get("decoder_forward_latency_ms", 0.0) or 0.0),
                            "pruned_decoder_compute_latency_ms": float(stats.get("pruned_decoder_compute_latency_ms", 0.0) or 0.0),
                            "pruning_layer_latency_ms": float(stats.get("pruning_layer_latency_ms", 0.0) or 0.0),
                            "fastv_score_latency_ms": float(stats.get("fastv_score_latency_ms", 0.0) or 0.0),
                            "fastv_select_latency_ms": float(stats.get("fastv_select_latency_ms", 0.0) or 0.0),
                            "fastv_gather_latency_ms": float(stats.get("fastv_gather_latency_ms", 0.0) or 0.0),
                            "fastv_overhead_latency_ms": float(stats.get("fastv_overhead_latency_ms", 0.0) or 0.0),
                            "sparsevlm_rater_latency_ms": float(stats.get("sparsevlm_rater_latency_ms", 0.0) or 0.0),
                            "sparsevlm_priority_latency_ms": float(stats.get("sparsevlm_priority_latency_ms", 0.0) or 0.0),
                            "sparsevlm_rank_budget_latency_ms": float(stats.get("sparsevlm_rank_budget_latency_ms", 0.0) or 0.0),
                            "sparsevlm_recycle_latency_ms": float(stats.get("sparsevlm_recycle_latency_ms", 0.0) or 0.0),
                            "sparsevlm_gather_latency_ms": float(stats.get("sparsevlm_gather_latency_ms", 0.0) or 0.0),
                            "sparsevlm_overhead_latency_ms": float(stats.get("sparsevlm_overhead_latency_ms", 0.0) or 0.0),
                            "compare_pruning_overhead_latency_ms": float(
                                stats.get("compare_pruning_overhead_latency_ms", 0.0) or 0.0
                            ),
                            "estimated_dense_decoder_latency_ms_linear_proxy": float(
                                stats.get("estimated_dense_decoder_latency_ms_linear_proxy", 0.0) or 0.0
                            ),
                            "estimated_saved_decoder_compute_latency_ms_linear_proxy": float(
                                stats.get("estimated_saved_decoder_compute_latency_ms_linear_proxy", 0.0) or 0.0
                            ),
                            "estimated_net_decoder_latency_delta_ms_linear_proxy": float(
                                stats.get("estimated_net_decoder_latency_delta_ms_linear_proxy", 0.0) or 0.0
                            ),
                            "original_decoder_token_sum": int(stats.get("original_decoder_token_sum", 0) or 0),
                            "actual_decoder_token_sum": int(stats.get("actual_decoder_token_sum", 0) or 0),
                            "saved_decoder_token_sum": int(stats.get("saved_decoder_token_sum", 0) or 0),
                            "flops_reduction": float(
                                stats.get("flops_reduction", 0.0)
                                or (patch_ref.get_flops_reduction() if hasattr(patch_ref, "get_flops_reduction") else 0.0)
                            ),
                        }
                    )
                
                action_queue.extend(actions)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = process_action(action, cfg.model_family)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if vlaadp_ref is not None and hasattr(vlaadp_ref, "observe_trajectory_step"):
                try:
                    vlaadp_ref.observe_trajectory_step(obs["robot0_eef_pos"], obs["robot0_eef_quat"])
                except Exception:
                    pass
            if vlaadp_ref is not None and len(action_queue) == 0 and hasattr(vlaadp_ref, "close_action_window"):
                try:
                    vlaadp_ref.close_action_window()
                except Exception:
                    pass
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)
        log_message(traceback.format_exc(), log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)
        
        # [LATENCY] Log Average
        if len(LATENCY_TRACKER) > 0:
            avg_lat = sum(LATENCY_TRACKER) / len(LATENCY_TRACKER)
            log_message(f"Avg Latency: {avg_lat:.2f} ms (Count: {len(LATENCY_TRACKER)})", log_file)
        if len(MODEL_LATENCY_TRACKER) > 0:
            avg_model_lat = sum(MODEL_LATENCY_TRACKER) / len(MODEL_LATENCY_TRACKER)
            log_message(f"Avg Model Latency: {avg_model_lat:.2f} ms (Count: {len(MODEL_LATENCY_TRACKER)})", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)
    LATENCY_TRACKER.clear()
    MODEL_LATENCY_TRACKER.clear()
    DECODER_LATENCY_TRACKER.clear()
    PRUNING_LAYER_LATENCY_TRACKER.clear()
    AUTOPRUNE_STATS_TRACKER.clear()

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)

    # ===============================================================================
    # ### [Start] AutoPrune Integrated Block ###
    # ===============================================================================
    ENABLE_AUTOPRUNE = cfg.use_auto_prune
    ENABLE_COMPARE_PRUNING = cfg.use_compare_pruning
    ENABLE_VLAPRUNER_OFT = cfg.use_vlapruner_oft
    ENABLE_VLAADP_OFT = cfg.use_vlaadp_oft

    enabled_pruners = (
        int(ENABLE_AUTOPRUNE)
        + int(ENABLE_COMPARE_PRUNING)
        + int(ENABLE_VLAPRUNER_OFT)
        + int(ENABLE_VLAADP_OFT)
    )
    if enabled_pruners > 1:
        raise ValueError("Please enable only one pruning method: AutoPrune, compare pruning, VLA-Pruner-OFT, or VLA-ADP-OFT.")

    if ENABLE_AUTOPRUNE:
        print("\n" + "="*40)
        print("✂️  Activating AutoPrune (Dynamic Schedule)...")
        print("="*40 + "\n")
        
        # === 核心修复：自动递归寻找真正的 Llama 模型 ===
        def find_llama_model_with_layers(module):
            # 1. 如果当前模块直接有 layers，且是列表，那就是它了
            if hasattr(module, "layers") and len(module.layers) > 0:
                return module
            
            # 2. 常见的子模块名称列表 (按优先级尝试)
            search_attrs = ["model", "language_model", "backbone", "llm", "transformer"]
            
            for attr in search_attrs:
                if hasattr(module, attr):
                    sub_module = getattr(module, attr)
                    # 递归查找
                    found = find_llama_model_with_layers(sub_module)
                    if found is not None:
                        return found
            return None

        # 开始查找
        target_model = find_llama_model_with_layers(model)
        
        if target_model is None:
            # 如果没找到，打印结构帮助调试
            print("❌ Error: Could not find Transformer layers in the model!")
            print(f"Top-level structure: {model}")
            # 也可以尝试强行指定（备用方案）
            # target_model = model.model.language_model.model 
        else:
            print(f"✅ Found target model with {len(target_model.layers)} layers.")
            
            try:
                patch_cls = AutoPrunePatchLegacy if cfg.autoprune_patch_variant == "legacy_budget" else AutoPrunePatchModern
                patcher = patch_cls(
                    model=model,    # 传入顶层 OpenVLA 模型，内部再定位并 patch Llama 层与动作预测入口
                    target_token_num=cfg.target_token_num,
                    visual_tokens=cfg.visual_tokens,
                    x0=cfg.x0,
                    k0=cfg.k0,
                    gamma=cfg.gamma,
                    start_layer=cfg.pruning_start_layer,
                    text_sample_ratio=cfg.text_sample_ratio,
                    safe_ratio=cfg.safe_ratio,
                    min_keep_per_view=cfg.min_keep_per_view,
                    budget_mode=cfg.budget_mode,
                    absolute_anchor_mode=cfg.absolute_anchor_mode,
                    absolute_budget_scale=cfg.absolute_budget_scale,
                    absolute_auto_bounds=cfg.absolute_auto_bounds,
                    enable_adaptive_layer_policy=cfg.enable_adaptive_layer_policy,
                    adaptive_layer_choices=cfg.adaptive_layer_choices,
                    adaptive_layer_mu=cfg.adaptive_layer_mu,
                    adaptive_layer_tau=cfg.adaptive_layer_tau,
                    enable_oneshot_fallback=cfg.enable_oneshot_fallback,
                    oneshot_fallback_max_layers=cfg.oneshot_fallback_max_layers,
                    oneshot_fallback_budget_margin=cfg.oneshot_fallback_budget_margin,
                    oneshot_fallback_frontload_ratio=cfg.oneshot_fallback_frontload_ratio,
                )
                patcher.apply()
                setattr(model, "_autoprune_patch_ref", patcher)
                print("✅ AutoPrune successfully patched!")
            except Exception as e:
                print(f"❌ Failed to apply AutoPrune: {e}")
            import traceback
            traceback.print_exc()

    if ENABLE_VLAPRUNER_OFT:
        print("\n" + "="*40)
        print("✂️  Activating VLA-Pruner-OFT (unified visual token pool)...")
        print("="*40 + "\n")
        try:
            vlapruner_cfg = VLAPrunerOFTConfig(
                enabled=True,
                keep_visual_tokens=cfg.vlapruner_keep_visual_tokens,
                visual_tokens=cfg.visual_tokens,
                pruning_layer=cfg.vlapruner_pruning_layer,
                temporal_window=cfg.vlapruner_temporal_window,
                temporal_gamma=cfg.vlapruner_temporal_gamma,
                warmup_steps=cfg.vlapruner_warmup_steps,
                action_attention_layer_start=cfg.vlapruner_action_attention_layer_start,
            )
            vlapruner_patcher = VLAPrunerOFTPatch(model=model, config=vlapruner_cfg)
            vlapruner_patcher.apply()
            setattr(model, "_vlapruner_patch_ref", vlapruner_patcher)
            print("✅ VLA-Pruner-OFT successfully patched!")
        except Exception as e:
            print(f"❌ Failed to apply VLA-Pruner-OFT: {e}")
            import traceback
            traceback.print_exc()

    if ENABLE_VLAADP_OFT:
        print("\n" + "="*40)
        print("✂️  Activating VLA-ADP-OFT (pre-LLM per-view QK pruning)...")
        print("="*40 + "\n")
        try:
            vlaadp_cfg = VLAADPOFTConfig(
                enabled=True,
                keep_visual_tokens=cfg.vlaadp_keep_visual_tokens,
                visual_tokens=cfg.visual_tokens,
                num_images_in_input=cfg.num_images_in_input,
                qk_layer=cfg.vlaadp_qk_layer,
                main_view_weight=cfg.vlaadp_main_view_weight,
                wrist_view_weight=cfg.vlaadp_wrist_view_weight,
                use_dynamic_visual_strategy=cfg.vlaadp_use_dynamic_visual_strategy,
                decision_method=cfg.vlaadp_decision_method,
                adjacent_variant=cfg.vlaadp_adjacent_variant,
                adjacent_extrema_window=cfg.vlaadp_adjacent_extrema_window,
                adjacent_lookback=cfg.vlaadp_adjacent_lookback,
                adjacent_last_state=cfg.vlaadp_adjacent_last_state,
                initial_state=cfg.vlaadp_initial_state,
                delta_method=cfg.vlaadp_delta_method,
                min_delta_pos=cfg.vlaadp_min_delta_pos,
                min_delta_rot=cfg.vlaadp_min_delta_rot,
                hysteresis_up=cfg.vlaadp_hysteresis_up,
                hysteresis_down=cfg.vlaadp_hysteresis_down,
                tol_equal=cfg.vlaadp_tol_equal,
                cold_start_windows=cfg.vlaadp_cold_start_windows,
                limit_consecutive_pruned_enabled=cfg.vlaadp_limit_consecutive_pruned_enabled,
                limit_max_consecutive_pruned=cfg.vlaadp_limit_max_consecutive_pruned,
            )
            vlaadp_patcher = VLAADPOFTPatch(model=model, config=vlaadp_cfg)
            vlaadp_patcher.apply()
            setattr(model, "_vlaadp_patch_ref", vlaadp_patcher)
            print("✅ VLA-ADP-OFT successfully patched!")
        except Exception as e:
            print(f"❌ Failed to apply VLA-ADP-OFT: {e}")
            import traceback
            traceback.print_exc()

    if ENABLE_COMPARE_PRUNING:
        print("\n" + "="*40)
        print(f"🧪 Activating compare pruning baseline: {cfg.compare_pruning_method}")
        print("="*40 + "\n")
        try:
            compare_cfg = ComparePruningConfig(
                method=cfg.compare_pruning_method,
                visual_token_num=cfg.compare_visual_token_num,
                num_images_in_input=cfg.num_images_in_input,
                important_ratio=cfg.compare_important_ratio,
                layer_list=tuple(cfg.compare_layer_list),
                image_token_ratio_list=tuple(cfg.compare_image_token_ratio_list),
                fastv_pruning_layer=cfg.compare_fastv_pruning_layer,
                sparsevlm_pruning_start_layer=cfg.compare_sparsevlm_pruning_start_layer,
                sparsevlm_lambda=cfg.compare_sparsevlm_lambda,
                sparsevlm_recycle_top_ratio=cfg.compare_sparsevlm_recycle_top_ratio,
                sparsevlm_cluster_ratio=cfg.compare_sparsevlm_cluster_ratio,
                sparsevlm_rank_tolerance=cfg.compare_sparsevlm_rank_tolerance,
                sparsevlm_knn_k=cfg.compare_sparsevlm_knn_k,
                min_keep_per_view=cfg.min_keep_per_view,
                enable_attention_probe=cfg.compare_enable_attention_probe,
                patch_scope=cfg.compare_patch_scope,
            )
            compare_patcher = ComparePruningPatch(model=model, config=compare_cfg)
            compare_patcher.apply()
            setattr(model, "_compare_patch_ref", compare_patcher)
            print("✅ Compare pruning baseline successfully patched!")
        except Exception as e:
            print(f"❌ Failed to apply compare pruning baseline: {e}")
            import traceback
            traceback.print_exc()
            
    # ===============================================================================
    # ### [End] AutoPrune Integrated Block ###
    # ===============================================================================

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Determine tasks to evaluate
    if cfg.task_id is not None:
        tasks_to_eval = [cfg.task_id]
    else:
        tasks_to_eval = range(num_tasks)

    for task_id in tqdm.tqdm(tasks_to_eval):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    if len(LATENCY_TRACKER) > 0:
        avg_lat = sum(LATENCY_TRACKER) / len(LATENCY_TRACKER)
        log_message(f"Avg Latency: {avg_lat:.2f} ms (Count: {len(LATENCY_TRACKER)})", log_file)
    if len(MODEL_LATENCY_TRACKER) > 0:
        avg_model_lat = sum(MODEL_LATENCY_TRACKER) / len(MODEL_LATENCY_TRACKER)
        log_message(f"Avg Model Latency: {avg_model_lat:.2f} ms (Count: {len(MODEL_LATENCY_TRACKER)})", log_file)

    if len(AUTOPRUNE_STATS_TRACKER) > 0:
        avg_budget = sum(item["raw_total_budget"] for item in AUTOPRUNE_STATS_TRACKER) / len(AUTOPRUNE_STATS_TRACKER)
        avg_layers = sum(item["active_pruning_layers_count"] for item in AUTOPRUNE_STATS_TRACKER) / len(AUTOPRUNE_STATS_TRACKER)
        avg_final_visual = sum(item["final_visual_tokens"] for item in AUTOPRUNE_STATS_TRACKER) / len(AUTOPRUNE_STATS_TRACKER)
        avg_flops_reduction = sum(item.get("flops_reduction", 0.0) for item in AUTOPRUNE_STATS_TRACKER) / len(AUTOPRUNE_STATS_TRACKER)
        avg_decoder_layers_recorded = sum(item.get("decoder_layers_recorded", 0) for item in AUTOPRUNE_STATS_TRACKER) / len(AUTOPRUNE_STATS_TRACKER)
        avg_decoder_layers_missing = sum(item.get("decoder_layers_missing", 0) for item in AUTOPRUNE_STATS_TRACKER) / len(AUTOPRUNE_STATS_TRACKER)
        min_budget = min(item["raw_total_budget"] for item in AUTOPRUNE_STATS_TRACKER)
        max_budget = max(item["raw_total_budget"] for item in AUTOPRUNE_STATS_TRACKER)
        max_views = max((len(item.get("view_keep_counts", [])) for item in AUTOPRUNE_STATS_TRACKER), default=0)
        avg_view_keep = []
        avg_view_mass = []
        for view_idx in range(max_views):
            keep_values = [item["view_keep_counts"][view_idx] for item in AUTOPRUNE_STATS_TRACKER if len(item.get("view_keep_counts", [])) > view_idx]
            mass_values = [item["view_score_masses"][view_idx] for item in AUTOPRUNE_STATS_TRACKER if len(item.get("view_score_masses", [])) > view_idx]
            if keep_values:
                avg_view_keep.append(sum(keep_values) / len(keep_values))
            if mass_values:
                avg_view_mass.append(sum(mass_values) / len(mass_values))
        log_message(
            f"AutoPrune stats: avg_raw_total_budget={avg_budget:.2f}, budget_range=[{min_budget:.2f}, {max_budget:.2f}], "
            f"avg_active_pruning_layers={avg_layers:.2f}, avg_final_visual_tokens={avg_final_visual:.2f}, "
            f"avg_flops_reduction={avg_flops_reduction * 100.0:.2f}%, "
            f"avg_decoder_layers_recorded={avg_decoder_layers_recorded:.2f}, "
            f"avg_decoder_layers_missing={avg_decoder_layers_missing:.2f}",
            log_file,
        )
        if avg_view_keep:
            log_message(
                f"AutoPrune view stats: avg_view_keep_counts={[round(x, 2) for x in avg_view_keep]}, "
                f"avg_view_score_masses={[round(x, 4) for x in avg_view_mass]}",
                log_file,
            )
        avg_image_prep_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "image_prep_latency_ms")
        avg_tokenization_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "tokenization_latency_ms")
        avg_proprio_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "proprio_latency_ms")
        avg_predict_action_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "predict_action_latency_ms")
        avg_non_model_overhead_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "non_model_overhead_latency_ms")
        avg_multimodal_build_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "multimodal_build_latency_ms")
        avg_prefill_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "prefill_latency_ms")
        avg_decode_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "decode_latency_ms")
        avg_decoder_forward_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "decoder_forward_latency_ms")
        avg_pruned_decoder_compute_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "pruned_decoder_compute_latency_ms")
        avg_pruning_layer_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "pruning_layer_latency_ms")
        avg_fastv_score_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "fastv_score_latency_ms")
        avg_fastv_select_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "fastv_select_latency_ms")
        avg_fastv_gather_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "fastv_gather_latency_ms")
        avg_fastv_overhead_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "fastv_overhead_latency_ms")
        avg_sparsevlm_rater_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "sparsevlm_rater_latency_ms")
        avg_sparsevlm_priority_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "sparsevlm_priority_latency_ms")
        avg_sparsevlm_rank_budget_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "sparsevlm_rank_budget_latency_ms")
        avg_sparsevlm_recycle_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "sparsevlm_recycle_latency_ms")
        avg_sparsevlm_gather_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "sparsevlm_gather_latency_ms")
        avg_sparsevlm_overhead_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "sparsevlm_overhead_latency_ms")
        avg_compare_pruning_overhead_lat = average_tracker_value(
            AUTOPRUNE_STATS_TRACKER,
            "compare_pruning_overhead_latency_ms",
        )
        avg_dense_decoder_lat_est = average_tracker_value(
            AUTOPRUNE_STATS_TRACKER,
            "estimated_dense_decoder_latency_ms_linear_proxy",
        )
        avg_saved_decoder_lat_est = average_tracker_value(
            AUTOPRUNE_STATS_TRACKER,
            "estimated_saved_decoder_compute_latency_ms_linear_proxy",
        )
        avg_net_decoder_delta_est = average_tracker_value(
            AUTOPRUNE_STATS_TRACKER,
            "estimated_net_decoder_latency_delta_ms_linear_proxy",
        )
        avg_original_decoder_token_sum = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "original_decoder_token_sum")
        avg_actual_decoder_token_sum = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "actual_decoder_token_sum")
        avg_saved_decoder_token_sum = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "saved_decoder_token_sum")
        avg_total_action_lat = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "total_action_latency_ms")
        avg_latency_total_ms = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "latency_ms")
        avg_model_latency_total_ms = average_tracker_value(AUTOPRUNE_STATS_TRACKER, "model_latency_ms")

        avg_prefill_share_of_model_pct = safe_ratio_pct(avg_prefill_lat, avg_predict_action_lat)
        avg_decode_share_of_model_pct = safe_ratio_pct(avg_decode_lat, avg_predict_action_lat)
        avg_non_model_share_of_total_pct = safe_ratio_pct(avg_non_model_overhead_lat, avg_total_action_lat)
        avg_compare_overhead_share_of_model_pct = safe_ratio_pct(
            avg_compare_pruning_overhead_lat,
            avg_predict_action_lat,
        )
        avg_saved_decoder_token_ratio_pct = safe_ratio_pct(
            avg_saved_decoder_token_sum,
            avg_original_decoder_token_sum,
        )

        avg_estimated_dense_predict_action_lat = None
        avg_estimated_dense_total_action_lat = None
        avg_estimated_saved_predict_action_lat = None
        avg_estimated_saved_total_action_lat = None
        if (
            avg_multimodal_build_lat is not None
            and avg_dense_decoder_lat_est is not None
            and avg_decode_lat is not None
        ):
            avg_estimated_dense_predict_action_lat = (
                float(avg_multimodal_build_lat)
                + float(avg_dense_decoder_lat_est)
                + float(avg_decode_lat)
            )
        if avg_estimated_dense_predict_action_lat is not None and avg_non_model_overhead_lat is not None:
            avg_estimated_dense_total_action_lat = (
                float(avg_estimated_dense_predict_action_lat) + float(avg_non_model_overhead_lat)
            )
        if avg_predict_action_lat is not None and avg_saved_decoder_lat_est is not None:
            avg_estimated_saved_predict_action_lat = min(
                float(avg_saved_decoder_lat_est),
                float(avg_predict_action_lat),
            )
        if avg_total_action_lat is not None and avg_estimated_dense_total_action_lat is not None:
            avg_estimated_saved_total_action_lat = max(
                0.0,
                float(avg_estimated_dense_total_action_lat) - float(avg_total_action_lat),
            )

        if avg_image_prep_lat is not None:
            log_message(
                "Latency breakdown: "
                f"avg_image_prep_latency={avg_image_prep_lat:.2f} ms, "
                f"avg_tokenization_latency={avg_tokenization_lat:.2f} ms, "
                f"avg_proprio_latency={avg_proprio_lat:.2f} ms, "
                f"avg_predict_action_latency={avg_predict_action_lat:.2f} ms, "
                f"avg_non_model_overhead_latency={avg_non_model_overhead_lat:.2f} ms",
                log_file,
            )
        if avg_prefill_lat is not None:
            log_message(
                "Model stage breakdown: "
                f"avg_multimodal_build_latency={avg_multimodal_build_lat:.2f} ms, "
                f"avg_prefill_latency={avg_prefill_lat:.2f} ms, "
                f"avg_decoder_forward_latency={avg_decoder_forward_lat:.2f} ms, "
                f"avg_decode_latency={avg_decode_lat:.2f} ms",
                log_file,
            )
        if avg_prefill_share_of_model_pct is not None:
            log_message(
                "Latency composition: "
                f"avg_prefill_share_of_model={avg_prefill_share_of_model_pct:.2f}%, "
                f"avg_decode_share_of_model={avg_decode_share_of_model_pct:.2f}%, "
                f"avg_non_model_share_of_total={avg_non_model_share_of_total_pct:.2f}%",
                log_file,
            )
        if avg_fastv_overhead_lat is not None:
            log_message(
                "FastV stage breakdown: "
                f"avg_pruning_layer_latency={avg_pruning_layer_lat:.2f} ms, "
                f"avg_fastv_score_latency={avg_fastv_score_lat:.2f} ms, "
                f"avg_fastv_select_latency={avg_fastv_select_lat:.2f} ms, "
                f"avg_fastv_gather_latency={avg_fastv_gather_lat:.2f} ms, "
                f"avg_fastv_overhead_latency={avg_fastv_overhead_lat:.2f} ms",
                log_file,
            )
        if avg_sparsevlm_overhead_lat is not None and avg_sparsevlm_overhead_lat > 0.0:
            log_message(
                "SparseVLM stage breakdown: "
                f"avg_sparsevlm_rater_latency={avg_sparsevlm_rater_lat:.2f} ms, "
                f"avg_sparsevlm_priority_latency={avg_sparsevlm_priority_lat:.2f} ms, "
                f"avg_sparsevlm_rank_budget_latency={avg_sparsevlm_rank_budget_lat:.2f} ms, "
                f"avg_sparsevlm_recycle_latency={avg_sparsevlm_recycle_lat:.2f} ms, "
                f"avg_sparsevlm_gather_latency={avg_sparsevlm_gather_lat:.2f} ms, "
                f"avg_sparsevlm_overhead_latency={avg_sparsevlm_overhead_lat:.2f} ms",
                log_file,
            )
        if avg_compare_pruning_overhead_lat is not None and avg_compare_overhead_share_of_model_pct is not None:
            log_message(
                "Pruning overhead summary: "
                f"avg_compare_pruning_overhead_latency={avg_compare_pruning_overhead_lat:.2f} ms, "
                f"avg_compare_pruning_overhead_share_of_model={avg_compare_overhead_share_of_model_pct:.2f}%",
                log_file,
            )
        if avg_dense_decoder_lat_est is not None:
            log_message(
                "Decoder saving estimate (linear proxy): "
                f"avg_pruned_decoder_compute_latency={avg_pruned_decoder_compute_lat:.2f} ms, "
                f"avg_dense_decoder_latency_est={avg_dense_decoder_lat_est:.2f} ms, "
                f"avg_saved_decoder_compute_latency_est={avg_saved_decoder_lat_est:.2f} ms, "
                f"avg_net_decoder_latency_delta_est={avg_net_decoder_delta_est:.2f} ms",
                log_file,
            )
        if avg_estimated_dense_predict_action_lat is not None:
            log_message(
                "Estimated dense-vs-pruned latency: "
                f"avg_predict_action_latency_pruned={avg_predict_action_lat:.2f} ms, "
                f"avg_predict_action_latency_dense_est={avg_estimated_dense_predict_action_lat:.2f} ms, "
                f"avg_saved_predict_action_latency_est={avg_estimated_saved_predict_action_lat:.2f} ms, "
                f"avg_total_action_latency_pruned={avg_total_action_lat:.2f} ms, "
                f"avg_total_action_latency_dense_est={avg_estimated_dense_total_action_lat:.2f} ms, "
                f"avg_saved_total_action_latency_est={avg_estimated_saved_total_action_lat:.2f} ms",
                log_file,
            )
        if avg_original_decoder_token_sum is not None:
            log_message(
                "Decoder token stats: "
                f"avg_original_decoder_token_sum={avg_original_decoder_token_sum:.2f}, "
                f"avg_actual_decoder_token_sum={avg_actual_decoder_token_sum:.2f}, "
                f"avg_saved_decoder_token_sum={avg_saved_decoder_token_sum:.2f}",
                log_file,
            )
        if avg_saved_decoder_token_ratio_pct is not None:
            log_message(
                "Token saving ratio: "
                f"avg_saved_decoder_token_ratio={avg_saved_decoder_token_ratio_pct:.2f}%, "
                f"avg_latency_total={avg_latency_total_ms:.2f} ms, "
                f"avg_model_latency_total={avg_model_latency_total_ms:.2f} ms",
                log_file,
            )
    
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
