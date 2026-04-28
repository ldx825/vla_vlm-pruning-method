import importlib.util
import torch
import numpy as np
import PIL.Image
import time
import types
import os
from scipy.fftpack import idct
from safetensors.torch import load_file as safetensors_load_file
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig, StoppingCriteria, StoppingCriteriaList
from typing import Optional, Union, Dict, Any, List, Tuple, Callable
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download
import json
from autoprune_nora import NoraAutoPruneConfig, NoraAutoPrunePatch
from compare_pruning_patch_nora import NoraComparePruningConfig, NoraComparePruningPatch


def has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def normalize_resize_size(resize_size: Union[int, tuple[int, int]]) -> tuple[int, int]:
    if isinstance(resize_size, int):
        return (resize_size, resize_size)
    if len(resize_size) != 2:
        raise ValueError(f"Expected resize size to have length 2, got {resize_size!r}")
    return tuple(int(dim) for dim in resize_size)


def compose_dual_view_image(
    image_primary: PIL.Image.Image,
    image_wrist: PIL.Image.Image,
    output_size: tuple[int, int],
) -> PIL.Image.Image:
    width, height = output_size[1], output_size[0]
    top_height = height // 2
    bottom_height = height - top_height

    primary_resized = image_primary.resize((width, top_height), PIL.Image.Resampling.BILINEAR)
    wrist_resized = image_wrist.resize((width, bottom_height), PIL.Image.Resampling.BILINEAR)

    canvas = PIL.Image.new("RGB", (width, height))
    canvas.paste(primary_resized, (0, 0))
    canvas.paste(wrist_resized, (0, top_height))
    return canvas


class _ExpectedActionCoeffStoppingCriteria(StoppingCriteria):
    def __init__(self, action_token_min: int, action_token_max: int, expected_coeffs: int, bpe_tokenizer):
        self.action_token_min = int(action_token_min)
        self.action_token_max = int(action_token_max)
        self.expected_coeffs = int(expected_coeffs)
        self.bpe_tokenizer = bpe_tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return False
        seq = input_ids[0]
        action_mask = (self.action_token_min <= seq) & (seq <= self.action_token_max)
        action_ids = (seq[action_mask] - self.action_token_min).detach().cpu().tolist()
        if not action_ids:
            return False
        decoded = self.bpe_tokenizer.decode(action_ids)
        return len(decoded) >= self.expected_coeffs

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

class Nora:
    
    # Define action token range and normalization bounds as class attributes
    # These are specific to the model's vocabulary and task
    _ACTION_TOKEN_MIN = 151665
    _ACTION_TOKEN_MAX = 153712


    def __init__(
        self,
        model_path: str ="declare-lab/nora",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16, # Make dtype configurable
        image_resize_size: Union[int, tuple[int, int]] = (224, 224),
        lora_adapter_path: Optional[str] = None,
        full_model_state_path: Optional[str] = None,
        dual_view_mode: str = "stitched",
        use_anyres: bool = False,
        autoprune_config: Optional[NoraAutoPruneConfig] = None,
        compare_pruning_config: Optional[NoraComparePruningConfig] = None,
        force_attn_implementation: Optional[str] = None,
        force_output_attentions: bool = False,
        enable_stage_profiling: bool = True,
        stop_at_expected_action_length: bool = False,
    ):
        """
        Initializes the QwenRoboInference model and processors.

        Args:
            model_path (str): Hugging Face model ID or local path for the main
                            Qwen 2.5 VL model and processor.
            device (Optional[str]): The device to use for inference (e.g., "cuda:0", "cpu").
                                     If None, automatically detects CUDA or falls back to CPU.
            torch_dtype (torch.dtype): The data type to use for the model.
                                       Defaults to torch.bfloat16.
        Raises:
            RuntimeError: If models or processors fail to load, or device is unavailable.
        """
        self.image_resize_size = normalize_resize_size(image_resize_size)
        self.dual_view_mode = dual_view_mode
        self.use_anyres = use_anyres
        self.anyres_pixels = int(self.image_resize_size[0] * self.image_resize_size[1])
        self.autoprune_patch = None
        self.last_inference_profile: Dict[str, float] = {}
        self.force_output_attentions = force_output_attentions
        self.enable_stage_profiling = enable_stage_profiling
        self.stop_at_expected_action_length = stop_at_expected_action_length

        if self.use_anyres and self.dual_view_mode == "stitched":
            raise ValueError("AnyRes mode currently supports only dual_view_mode='separate'.")

        # --- Device Setup ---
        self.libero_keys= {'libero_object':[np.array( [
        -0.5383928418159485,
        -0.8758928775787354,
        -0.9375,
        -0.06964285671710968,
        -0.11678571254014969,
        -0.15964286029338837,
        0.0
      ]),np.array([
        0.8464285731315613,
        0.84375,
        0.9375,
        0.08142857253551483,
        0.14892856776714325,
        0.0867857113480568,
        1.0
      ])],'libero_object_min':[np.array([-0.8839286 , -0.9375    , -0.9375    , -0.15000001, -0.29035714,
        -0.32892856, -1.        ]),np.array([0.9375    , 0.89196426, 0.9375    , 0.17678571, 0.35035715,
        0.18107143, 1.        ])],
      'libero_10':[np.array([
        -0.6348214149475098,
        -0.7741071581840515,
        -0.7633928656578064,
        -0.09749999642372131,
        -0.14819999992847435,
        -0.2742857038974762,
        0.0
      ]),np.array([
        0.7714285850524902,
        0.8464285731315613,
        0.9375,
        0.13928571343421936,
        0.15964286029338837,
        0.3246428668498993,
        1.0
      ])],'libero_90':[np.array([
        -0.6348214149475098,
        -0.7741071581840515,
        -0.7633928656578064,
        -0.09749999642372131,
        -0.14819999992847435,
        -0.2742857038974762,
        0.0
      ]),np.array([
        0.7714285850524902,
        0.8464285731315613,
        0.9375,
        0.13928571343421936,
        0.15964286029338837,
        0.3246428668498993,
        1.0
      ])],'libero_goal':[np.array([-0.8785714507102966,
        -0.7553571462631226,
        -0.9375,
        -0.1510714292526245,
        -0.1639285683631897,
        -0.13777500048279764,
        0.0])
       ,np.array([0.9375,
        0.9107142686843872,
        0.9375,
        0.20357142388820648,
        0.26357144117355347,
        0.375,
        1.0])],'libero_spatial':[np.array([
        -0.7454732114076613,
        -0.6616071462631226,
        -0.9375,
        -0.1071428582072258,
        -0.20678570866584778,
        -0.1842857152223587,
        0.0
      ]),np.array([
        0.9375,
        0.8758928775787354,
        0.9321428537368774,
        0.1039285734295845,
        0.17678570747375488,
        0.14571428298950195,
        1.0
      ])],
      'libero_spatial_min':[np.array([-0.9375    , -0.9375    , -0.9375    , -0.1875    , -0.36750001,
        -0.36000001, -1]),
        np.array([0.9375    , 0.9375    , 0.9375    , 0.19714285, 0.33642858,
        0.375     , 1.        ])]}

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
        else:
            self.device = device
            print(f"Using specified device: {self.device}")

        if self.device.startswith("cuda"):
             if not torch.cuda.is_available():
                  raise RuntimeError(f"CUDA is not available, but device '{self.device}' was specified.")
             gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
             if gpu_id >= torch.cuda.device_count():
                  raise RuntimeError(f"CUDA device {gpu_id} not available. Only {torch.cuda.device_count()} devices found.")

        # --- Load Fast Tokenizer ---
        try:
            print(f"Loading fast tokenizer from: physical-intelligence/fast")
            self.fast_tokenizer = AutoProcessor.from_pretrained(
                "physical-intelligence/fast", trust_remote_code=True
            )
            # Ensure required attributes are set/exist
           
            self.fast_tokenizer.action_dim = 7 # Set default if not in config
            print("Setting action_dim  to 7.")
           
            self.fast_tokenizer.time_horizon = 1 # Set default if not in config
            print("Setting time horizon to 1.")

        except Exception as e:
            raise RuntimeError(
                f"Error loading fast tokenizer: {e}. "
            )

        # --- Load Main Processor ---
        try:
            print(f"Loading main processor from: {model_path}")
            # Assuming the main processor is saved in the same location as the model
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            if self.use_anyres:
                self.processor.image_processor.min_pixels = self.anyres_pixels
                self.processor.image_processor.max_pixels = self.anyres_pixels
                self.processor.image_processor.size = {
                    "shortest_edge": self.anyres_pixels,
                    "longest_edge": self.anyres_pixels,
                }
        except Exception as e:
            raise RuntimeError(f"Error loading main processor from {model_path}: {e}")

        # --- Load Main Model ---
        try:
            print(f"Loading model from: {model_path}")
            attn_implementation = force_attn_implementation
            flash_attn_available = has_flash_attn()
            if (
                attn_implementation is None
                and compare_pruning_config is not None
                and compare_pruning_config.enabled
            ):
                # FastV / SparseVLM both need decoder attentions on their pruning layers.
                attn_implementation = "eager"
            elif attn_implementation is None and autoprune_config is not None and autoprune_config.enabled:
                # When we can get pruning scores from the lightweight manual probe,
                # the actual decoder forward can still stay on FlashAttention/SDPA.
                no_prune_budget = (
                    int(getattr(autoprune_config, "b_min", 0) or 0) >= int(getattr(autoprune_config, "visual_tokens", 0) or 0)
                    and int(getattr(autoprune_config, "b_max", 0) or 0) >= int(getattr(autoprune_config, "visual_tokens", 0) or 0)
                )
                flash_probe_mode = (
                    autoprune_config.prefer_flash_attention
                    and (autoprune_config.manual_attention_probe or no_prune_budget)
                )
                if flash_probe_mode:
                    if flash_attn_available:
                        attn_implementation = "flash_attention_2"
                    else:
                        attn_implementation = "sdpa"
                        print(
                            "Warning: flash_attn is not installed; falling back to SDPA for "
                            "manual-probe AutoPrune."
                        )
                else:
                    # If the pruning path depends on model-returned attentions, Qwen needs eager.
                    attn_implementation = "eager"

            if attn_implementation == "flash_attention_2" and not flash_attn_available:
                fallback_impl = "sdpa"
                print(
                    "Warning: flash_attention_2 was requested but flash_attn is unavailable; "
                    f"falling back to {fallback_impl}."
                )
                attn_implementation = fallback_impl

            print(f"Resolved attention implementation: {attn_implementation}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
            #    attn_implementation="flash_attention_2", # Comment out this line if there is an error with flash attention
            )

            if lora_adapter_path:
                try:
                    from peft import PeftModel
                except ImportError as exc:
                    raise RuntimeError("A LoRA adapter path was provided but peft is not installed.") from exc
                self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)

            if full_model_state_path:
                if not os.path.isfile(full_model_state_path):
                    raise RuntimeError(f"Full model state file not found: {full_model_state_path}")
                print(f"Loading additional model state from: {full_model_state_path}")
                state_dict = safetensors_load_file(full_model_state_path, device="cpu")
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                print(
                    "Loaded additional model state with "
                    f"missing_keys={len(missing_keys)}, unexpected_keys={len(unexpected_keys)}"
                )
                if missing_keys:
                    print(f"Missing key preview: {missing_keys[:10]}")
                if unexpected_keys:
                    print(f"Unexpected key preview: {unexpected_keys[:10]}")

            if autoprune_config is not None and autoprune_config.enabled:
                self.autoprune_patch = NoraAutoPrunePatch(self.model, autoprune_config)
                self.autoprune_patch.apply()
                if autoprune_config.disable_kv_cache:
                    self.model.config.use_cache = False
                    self.model.generation_config.use_cache = False
                    print("AutoPrune enabled for inference (NORA), KV cache disabled by config.")
                else:
                    self.model.config.use_cache = True
                    self.model.generation_config.use_cache = True
                    print("AutoPrune enabled for inference (NORA), KV cache kept enabled.")
            elif compare_pruning_config is not None and compare_pruning_config.enabled:
                self.autoprune_patch = NoraComparePruningPatch(self.model, compare_pruning_config)
                self.autoprune_patch.apply()
                self.model.config.use_cache = True
                self.model.generation_config.use_cache = True
                print(
                    "Compare pruning baseline enabled for inference (NORA): "
                    f"{compare_pruning_config.method}"
                )

            self.model.to(self.device)
            # Load generation config and set specific parameters
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
            self.model.generation_config.do_sample = False # Ensure deterministic output
            if compare_pruning_config is not None and compare_pruning_config.enabled:
                self.model.generation_config.use_cache = True

            self.model.eval() # Set the model to evaluation mode

            repo_id = "declare-lab/nora"
            filename = "norm_stats.json"

            norm_stats_path = os.environ.get("NORA_NORM_STATS_PATH")
            if norm_stats_path and not os.path.exists(norm_stats_path):
                raise FileNotFoundError(f"NORA_NORM_STATS_PATH does not exist: {norm_stats_path}")

            if norm_stats_path:
                file_path = norm_stats_path
            else:
                try:
                    file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
                except Exception:
                    # Fall back to the previous online behavior when the cache is missing.
                    file_path = hf_hub_download(repo_id=repo_id, filename=filename)

            # Load the JSON file
            with open(file_path, "r") as f:
                norm_stats = json.load(f)
            self.norm_stats = norm_stats
            

        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")

        print("Model and processors loaded successfully.")

    def _decode_action_tokens(self, generated_ids: torch.Tensor) -> np.ndarray:
        action_token_mask = (self._ACTION_TOKEN_MIN <= generated_ids[0]) & (generated_ids[0] <= self._ACTION_TOKEN_MAX)
        action_token_indices = torch.where(action_token_mask)[0]
        if len(action_token_indices) == 0:
            raise RuntimeError("Model generation did not contain any action tokens.")

        action_token_ids = (generated_ids[0][action_token_indices] - self._ACTION_TOKEN_MIN).detach().cpu().tolist()
        time_horizon = int(self.fast_tokenizer.time_horizon or 1)
        action_dim = int(self.fast_tokenizer.action_dim or 7)
        expected_coeffs = time_horizon * action_dim

        decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(action_token_ids)
        decoded_coeffs = np.array([ord(char) + self.fast_tokenizer.min_token for char in decoded_tokens], dtype=np.float32)

        if decoded_coeffs.size == 0:
            raise RuntimeError("Model generation did not decode into any DCT coefficients.")
        if decoded_coeffs.size < expected_coeffs:
            print(f"Warning: decoded {decoded_coeffs.size} coefficients, padding to {expected_coeffs}.")
            decoded_coeffs = np.pad(decoded_coeffs, (0, expected_coeffs - decoded_coeffs.size), constant_values=0.0)
        elif decoded_coeffs.size > expected_coeffs:
            print(f"Warning: decoded {decoded_coeffs.size} coefficients, truncating to {expected_coeffs}.")
            decoded_coeffs = decoded_coeffs[:expected_coeffs]

        decoded_coeffs = decoded_coeffs.reshape(time_horizon, action_dim)
        decoded_actions = idct(decoded_coeffs / self.fast_tokenizer.scale, axis=0, norm="ortho")
        return decoded_actions[None, ...]

    def _count_action_tokens(self, generated_ids: torch.Tensor) -> int:
        action_token_mask = (self._ACTION_TOKEN_MIN <= generated_ids[0]) & (generated_ids[0] <= self._ACTION_TOKEN_MAX)
        return int(action_token_mask.sum().item())
    
    @torch.inference_mode()
    def inference(
        self,
        image_primary: np.ndarray,
        image_wrist: Optional[np.ndarray],
        instruction: str,
        unnorm_key: str = None,
        unnormalizer=None,
    ) -> np.ndarray:
        infer_t0 = time.perf_counter()

        # Stage profiling target (per call):
        # generate_seconds ~= vit_seconds + prefill_seconds + decode_model_seconds + residual.
        prefill_forward_seconds = 0.0
        decode_forward_seconds = 0.0
        vit_forward_seconds = 0.0

        cuda_timing = self.enable_stage_profiling and self.device.startswith("cuda") and torch.cuda.is_available()
        forward_event_spans: List[Tuple[torch.cuda.Event, torch.cuda.Event, str]] = []
        vit_event_spans: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        forward_call_index = 0

        if not isinstance(image_primary, PIL.Image.Image):
            image_primary = PIL.Image.fromarray(image_primary)
        if image_wrist is not None and not isinstance(image_wrist, PIL.Image.Image):
            image_wrist = PIL.Image.fromarray(image_wrist)

        if self.autoprune_patch is not None:
            self.autoprune_patch.reset_runtime_state()

        if image_wrist is not None and self.dual_view_mode == "stitched":
            image_primary = compose_dual_view_image(image_primary, image_wrist, self.image_resize_size)
            image_wrist = None

        primary_image = {"type": "image", "image": image_primary}
        if self.use_anyres:
            primary_image["min_pixels"] = self.anyres_pixels
            primary_image["max_pixels"] = self.anyres_pixels
        else:
            primary_image["resized_height"] = self.image_resize_size[0]
            primary_image["resized_width"] = self.image_resize_size[1]

        content = [primary_image]
        if image_wrist is not None:
            wrist_image_content = {"type": "image", "image": image_wrist}
            if self.use_anyres:
                wrist_image_content["min_pixels"] = self.anyres_pixels
                wrist_image_content["max_pixels"] = self.anyres_pixels
            else:
                wrist_image_content["resized_height"] = self.image_resize_size[0]
                wrist_image_content["resized_width"] = self.image_resize_size[1]
            content.append(wrist_image_content)
        content.append({"type": "text", "text": instruction})

        messages = [{"role": "user", "content": content}]
        prep_t0 = time.perf_counter()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        if self.autoprune_patch is not None:
            vision_config = getattr(getattr(self.model, "config", None), "vision_config", None)
            spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
            self.autoprune_patch.prepare_input_layout(
                input_ids=inputs.get("input_ids"),
                image_grid_thw=inputs.get("image_grid_thw"),
                attention_mask=inputs.get("attention_mask"),
                image_token_id=getattr(getattr(self.model, "config", None), "image_token_id", None),
                num_views=self.autoprune_patch.config.num_views,
                dual_view_mode=self.dual_view_mode,
                spatial_merge_size=spatial_merge_size,
            )
        prep_t1 = time.perf_counter()

        gen_t0 = time.perf_counter()

        nonlocal_prefill = [0.0]
        nonlocal_decode = [0.0]
        nonlocal_vit = [0.0]

        timed_forward_module = None
        original_model_forward = None
        original_vision_forward: Optional[Callable[..., Any]] = None
        vision_module = None

        if self.enable_stage_profiling:
            timed_forward_module = getattr(self.model, "model", self.model)
            original_model_forward = timed_forward_module.forward
            for attr_name in ("visual", "vision_tower", "vision_model"):
                candidate = getattr(timed_forward_module, attr_name, None)
                if candidate is None:
                    candidate = getattr(self.model, attr_name, None)
                if candidate is not None:
                    vision_module = candidate
                    break

            def _timed_model_forward(model_self, *args, **kwargs):
                nonlocal forward_call_index
                stage_name = "prefill" if forward_call_index == 0 else "decode"
                forward_call_index += 1
                if cuda_timing:
                    start_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    output = original_model_forward(*args, **kwargs)
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    forward_event_spans.append((start_event, end_event, stage_name))
                    return output

                start_time = time.perf_counter()
                output = original_model_forward(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                if stage_name == "prefill":
                    nonlocal_prefill[0] += elapsed
                else:
                    nonlocal_decode[0] += elapsed
                return output

            timed_forward_module.forward = types.MethodType(_timed_model_forward, timed_forward_module)

            if vision_module is not None:
                original_vision_forward = vision_module.forward

                def _timed_vision_forward(vision_self, *args, **kwargs):
                    if cuda_timing:
                        start_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        output = original_vision_forward(*args, **kwargs)
                        end_event = torch.cuda.Event(enable_timing=True)
                        end_event.record()
                        vit_event_spans.append((start_event, end_event))
                        return output

                    start_time = time.perf_counter()
                    output = original_vision_forward(*args, **kwargs)
                    nonlocal_vit[0] += (time.perf_counter() - start_time)
                    return output

                vision_module.forward = types.MethodType(_timed_vision_forward, vision_module)

        generate_kwargs = dict(inputs)
        if self.force_output_attentions:
            generate_kwargs.update(
                output_attentions=True,
                return_dict_in_generate=True,
            )
        time_horizon = int(self.fast_tokenizer.time_horizon or 1)
        action_dim = int(self.fast_tokenizer.action_dim or 7)
        expected_coeffs = time_horizon * action_dim
        if self.stop_at_expected_action_length:
            # Nora-LIBERO action decoding is extremely compact in practice: across
            # prior formal runs on this machine, normal generations stayed within
            # 20 new tokens while pathological outliers exploded far beyond that.
            # Keep the semantic stop criterion, but also add a hard decode cap so
            # rare bad generations cannot dominate latency.
            generate_kwargs.setdefault("max_new_tokens", max(20, int(np.ceil(expected_coeffs * 0.55))))
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([
                _ExpectedActionCoeffStoppingCriteria(
                    action_token_min=self._ACTION_TOKEN_MIN,
                    action_token_max=self._ACTION_TOKEN_MAX,
                    expected_coeffs=expected_coeffs,
                    bpe_tokenizer=self.fast_tokenizer.bpe_tokenizer,
                )
            ])

        should_compress_prefill = (
            self.autoprune_patch is not None
            and hasattr(self.autoprune_patch, "compress_generate_inputs")
            and (
                not hasattr(self.autoprune_patch, "should_compress_generate_inputs")
                or self.autoprune_patch.should_compress_generate_inputs(generate_kwargs)
            )
        )
        if should_compress_prefill:
            generate_kwargs = self.autoprune_patch.compress_generate_inputs(self.model, generate_kwargs)

        try:
            generated = self.model.generate(**generate_kwargs)
        finally:
            if self.enable_stage_profiling and timed_forward_module is not None and original_model_forward is not None:
                timed_forward_module.forward = original_model_forward
            if self.enable_stage_profiling and vision_module is not None and original_vision_forward is not None:
                vision_module.forward = original_vision_forward

        if self.enable_stage_profiling and cuda_timing:
            torch.cuda.synchronize(device=self.device)
            for start_event, end_event, stage_name in forward_event_spans:
                elapsed_sec = start_event.elapsed_time(end_event) / 1000.0
                if stage_name == "prefill":
                    prefill_forward_seconds += elapsed_sec
                else:
                    decode_forward_seconds += elapsed_sec
            for start_event, end_event in vit_event_spans:
                vit_forward_seconds += start_event.elapsed_time(end_event) / 1000.0
        elif self.enable_stage_profiling:
            prefill_forward_seconds = nonlocal_prefill[0]
            decode_forward_seconds = nonlocal_decode[0]
            vit_forward_seconds = nonlocal_vit[0]

        generated_ids = generated.sequences if self.force_output_attentions else generated
        gen_t1 = time.perf_counter()

        decode_t0 = time.perf_counter()
        prompt_source_ids = generate_kwargs.get("input_ids", inputs.get("input_ids"))
        prompt_token_count = int(prompt_source_ids.shape[1]) if prompt_source_ids is not None else 0
        generated_token_count = max(int(generated_ids.shape[1]) - prompt_token_count, 0)
        action_token_count = self._count_action_tokens(generated_ids)
        decode_to_action_ratio = float(generated_token_count / max(action_token_count, 1))
        output_action = self._decode_action_tokens(generated_ids)

        preprocess_seconds = prep_t1 - prep_t0
        generate_seconds = gen_t1 - gen_t0
        decode_seconds = time.perf_counter() - decode_t0

        patch_total = 0.0
        force_attn = 0.0
        manual_probe = 0.0
        metric = 0.0
        rank = 0.0
        budget_schedule = 0.0
        topk = 0.0
        gather = 0.0
        compaction_bookkeeping = 0.0
        last_tokens_before = 0.0
        last_tokens_after = 0.0
        schedule_len = 0.0
        pruned_layers = 0.0
        layout_ready = 0.0
        layout_query_tokens = 0.0
        layout_view0_tokens = 0.0
        layout_view1_tokens = 0.0
        manual_attention_probe_layers = 0.0
        manual_probe_qk_cache_hits = 0.0
        manual_probe_qk_cache_misses = 0.0
        manual_probe_reused_layers = 0.0
        layer0_total_calls = 0.0
        layer0_short_seq_calls = 0.0
        layer0_long_seq_calls = 0.0
        layer0_past_none_calls = 0.0
        layer0_past_present_calls = 0.0
        flops_reduction = 0.0
        decoder_layers_recorded = 0.0
        decoder_layers_missing = 0.0
        final_visual_tokens = 0.0
        raw_total_budget = 0.0
        active_pruning_layers = 0.0
        expected_compacted_seq_len = 0.0
        decode_cache_seq_len_layer0 = 0.0
        decode_expected_visual_tokens_layer0 = 0.0
        decode_expected_total_seq_len_layer0 = 0.0
        patch_total = 0.0
        force_attn = 0.0
        force_attn_self_attn = 0.0
        force_attn_mlp = 0.0
        force_attn_residual_norm = 0.0
        regular_forward = 0.0
        regular_self_attn = 0.0
        regular_mlp = 0.0
        regular_residual_norm = 0.0
        manual_probe = 0.0
        metric = 0.0
        rank = 0.0
        budget_schedule = 0.0
        topk = 0.0
        gather = 0.0
        compaction_bookkeeping = 0.0
        last_tokens_before = 0.0
        last_tokens_after = 0.0
        schedule_len = 0.0
        pruned_layers = 0.0
        layout_ready = 0.0
        layout_query_tokens = 0.0
        layout_view0_tokens = 0.0
        layout_view1_tokens = 0.0
        manual_attention_probe_layers = 0.0
        attention_mask_gathered = 0.0
        attention_mask_dropped = 0.0
        cache_layer0_seq_len = 0.0
        cache_min_seq_len = 0.0
        cache_max_seq_len = 0.0
        cache_unique_seq_lens = 0.0
        if self.autoprune_patch is not None:
            stats = self.autoprune_patch.runtime_stats or {}
            patch_total = float(stats.get("patch_total_seconds", 0.0) or 0.0)
            force_attn = float(stats.get("force_attn_seconds", 0.0) or 0.0)
            force_attn_self_attn = float(stats.get("force_attn_self_attn_seconds", 0.0) or 0.0)
            force_attn_mlp = float(stats.get("force_attn_mlp_seconds", 0.0) or 0.0)
            force_attn_residual_norm = float(stats.get("force_attn_residual_norm_seconds", 0.0) or 0.0)
            regular_forward = float(stats.get("regular_forward_seconds", 0.0) or 0.0)
            regular_self_attn = float(stats.get("regular_self_attn_seconds", 0.0) or 0.0)
            regular_mlp = float(stats.get("regular_mlp_seconds", 0.0) or 0.0)
            regular_residual_norm = float(stats.get("regular_residual_norm_seconds", 0.0) or 0.0)
            manual_probe = float(stats.get("manual_probe_seconds", 0.0) or 0.0)
            metric = float(stats.get("metric_seconds", 0.0) or 0.0)
            rank = float(stats.get("rank_seconds", 0.0) or 0.0)
            budget_schedule = float(stats.get("budget_schedule_seconds", 0.0) or 0.0)
            topk = float(stats.get("topk_seconds", 0.0) or 0.0)
            gather = float(stats.get("gather_seconds", 0.0) or 0.0)
            compaction_bookkeeping = float(stats.get("compaction_bookkeeping_seconds", 0.0) or 0.0)
            last_tokens_before = float(stats.get("last_tokens_before", 0.0) or 0.0)
            last_tokens_after = float(stats.get("last_tokens_after", 0.0) or 0.0)
            schedule_len = float(stats.get("schedule_len", 0.0) or 0.0)
            pruned_layers = float(stats.get("pruned_layers", 0.0) or 0.0)
            layout_ready = float(stats.get("layout_ready", 0.0) or 0.0)
            layout_query_tokens = float(stats.get("layout_query_tokens", 0.0) or 0.0)
            layout_view0_tokens = float(stats.get("layout_view0_tokens", 0.0) or 0.0)
            layout_view1_tokens = float(stats.get("layout_view1_tokens", 0.0) or 0.0)
            manual_attention_probe_layers = float(stats.get("manual_attention_probe_layers", 0.0) or 0.0)
            manual_probe_qk_cache_hits = float(stats.get("manual_probe_qk_cache_hits", 0.0) or 0.0)
            manual_probe_qk_cache_misses = float(stats.get("manual_probe_qk_cache_misses", 0.0) or 0.0)
            manual_probe_reused_layers = float(stats.get("manual_probe_reused_layers", 0.0) or 0.0)
            layer0_total_calls = float(stats.get("layer0_total_calls", 0.0) or 0.0)
            layer0_short_seq_calls = float(stats.get("layer0_short_seq_calls", 0.0) or 0.0)
            layer0_long_seq_calls = float(stats.get("layer0_long_seq_calls", 0.0) or 0.0)
            layer0_past_none_calls = float(stats.get("layer0_past_none_calls", 0.0) or 0.0)
            layer0_past_present_calls = float(stats.get("layer0_past_present_calls", 0.0) or 0.0)
            flops_reduction = float(stats.get("flops_reduction", 0.0) or 0.0)
            decoder_layers_recorded = float(stats.get("decoder_layers_recorded", 0.0) or 0.0)
            decoder_layers_missing = float(stats.get("decoder_layers_missing", 0.0) or 0.0)
            final_visual_tokens = float(stats.get("final_visual_tokens", 0.0) or 0.0)
            raw_total_budget = float(stats.get("raw_total_budget", 0.0) or 0.0)
            active_pruning_layers = float(stats.get("active_pruning_layers", 0.0) or 0.0)
            expected_compacted_seq_len = float(stats.get("expected_compacted_seq_len", 0.0) or 0.0)
            decode_cache_seq_len_layer0 = float(stats.get("decode_cache_seq_len_layer0", 0.0) or 0.0)
            decode_expected_visual_tokens_layer0 = float(stats.get("decode_expected_visual_tokens_layer0", 0.0) or 0.0)
            decode_expected_total_seq_len_layer0 = float(stats.get("decode_expected_total_seq_len_layer0", 0.0) or 0.0)
            attention_mask_gathered = float(stats.get("attention_mask_gathered", 0.0) or 0.0)
            attention_mask_dropped = float(stats.get("attention_mask_dropped", 0.0) or 0.0)
            cache_layer0_seq_len = float(stats.get("cache_layer0_seq_len", 0.0) or 0.0)
            cache_min_seq_len = float(stats.get("cache_min_seq_len", 0.0) or 0.0)
            cache_max_seq_len = float(stats.get("cache_max_seq_len", 0.0) or 0.0)
            cache_unique_seq_lens = float(stats.get("cache_unique_seq_lens", 0.0) or 0.0)

        self.last_inference_profile = {
            "preprocess_seconds": float(preprocess_seconds),
            "generate_seconds": float(generate_seconds),
            "decode_seconds": float(decode_seconds),
            "inference_total_seconds": float(time.perf_counter() - infer_t0),
            "vit_seconds": float(vit_forward_seconds),
            "prefill_seconds": float(max(prefill_forward_seconds - vit_forward_seconds, 0.0)),
            "decode_model_seconds": float(decode_forward_seconds),
            "patch_total_seconds": patch_total,
            "force_attn_seconds": force_attn,
            "force_attn_self_attn_seconds": force_attn_self_attn,
            "force_attn_mlp_seconds": force_attn_mlp,
            "force_attn_residual_norm_seconds": force_attn_residual_norm,
            "regular_forward_seconds": regular_forward,
            "regular_self_attn_seconds": regular_self_attn,
            "regular_mlp_seconds": regular_mlp,
            "regular_residual_norm_seconds": regular_residual_norm,
            "manual_probe_seconds": manual_probe,
            "metric_seconds": metric,
            "rank_seconds": rank,
            "budget_schedule_seconds": budget_schedule,
            "topk_seconds": topk,
            "gather_seconds": gather,
            "compaction_bookkeeping_seconds": compaction_bookkeeping,
            "last_tokens_before": last_tokens_before,
            "last_tokens_after": last_tokens_after,
            "schedule_len": schedule_len,
            "pruned_layers": pruned_layers,
            "layout_ready": layout_ready,
            "layout_query_tokens": layout_query_tokens,
            "layout_view0_tokens": layout_view0_tokens,
            "layout_view1_tokens": layout_view1_tokens,
            "manual_attention_probe_layers": manual_attention_probe_layers,
            "manual_probe_qk_cache_hits": manual_probe_qk_cache_hits,
            "manual_probe_qk_cache_misses": manual_probe_qk_cache_misses,
            "manual_probe_reused_layers": manual_probe_reused_layers,
            "generated_token_count": float(generated_token_count),
            "action_token_count": float(action_token_count),
            "decode_to_action_ratio": float(decode_to_action_ratio),
            "expected_coeff_count": float(expected_coeffs),
            "layer0_total_calls": layer0_total_calls,
            "layer0_short_seq_calls": layer0_short_seq_calls,
            "layer0_long_seq_calls": layer0_long_seq_calls,
            "layer0_past_none_calls": layer0_past_none_calls,
            "layer0_past_present_calls": layer0_past_present_calls,
            "flops_reduction": flops_reduction,
            "decoder_layers_recorded": decoder_layers_recorded,
            "decoder_layers_missing": decoder_layers_missing,
            "final_visual_tokens": final_visual_tokens,
            "raw_total_budget": raw_total_budget,
            "active_pruning_layers": active_pruning_layers,
            "expected_compacted_seq_len": expected_compacted_seq_len,
            "decode_cache_seq_len_layer0": decode_cache_seq_len_layer0,
            "decode_expected_visual_tokens_layer0": decode_expected_visual_tokens_layer0,
            "decode_expected_total_seq_len_layer0": decode_expected_total_seq_len_layer0,
            "attention_mask_gathered": attention_mask_gathered,
            "attention_mask_dropped": attention_mask_dropped,
            "cache_layer0_seq_len": cache_layer0_seq_len,
            "cache_min_seq_len": cache_min_seq_len,
            "cache_max_seq_len": cache_max_seq_len,
            "cache_unique_seq_lens": cache_unique_seq_lens,
        }

        if unnormalizer is not None:
            unnormalized_action = unnormalizer({"action": output_action})
            return unnormalized_action["action"]

        if "libero" in unnorm_key:
            action_low, action_high = self.libero_keys[unnorm_key]
        else:
            action_norm_stats = self.get_action_stats(unnorm_key)
            action_high = np.array(action_norm_stats["q99"])
            action_low = np.array(action_norm_stats["q01"])

        unnorm_actions = 0.5 * (output_action + 1) * (action_high - action_low) + action_low
        return np.array(unnorm_actions[0])

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    
    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
    
