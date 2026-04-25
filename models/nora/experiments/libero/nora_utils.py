import torch
import numpy as np
import PIL.Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from typing import Optional, Union, Dict, Any
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download
import json
import re
import sys
import time
from pathlib import Path

try:
    from nora_baseline_pruning import NoraBaselinePruningConfig, maybe_apply_nora_baseline_pruning
except Exception:
    _ROOT = Path(__file__).resolve().parents[4]
    _BASELINE_DIR = _ROOT / "methods" / "auto_prune" / "nora" / "experiments" / "libero"
    if str(_BASELINE_DIR) not in sys.path:
        sys.path.insert(0, str(_BASELINE_DIR))
    try:
        from nora_baseline_pruning import NoraBaselinePruningConfig, maybe_apply_nora_baseline_pruning
    except Exception:
        NoraBaselinePruningConfig = None
        maybe_apply_nora_baseline_pruning = None

try:
    from autoprune_nora import NoraAutoPruneConfig, maybe_apply_nora_autoprune
except Exception:
    _ROOT = Path(__file__).resolve().parents[4]
    _AUTOPRUNE_DIR = _ROOT / "methods" / "auto_prune" / "nora" / "experiments" / "libero"
    if str(_AUTOPRUNE_DIR) not in sys.path:
        sys.path.insert(0, str(_AUTOPRUNE_DIR))
    try:
        from autoprune_nora import NoraAutoPruneConfig, maybe_apply_nora_autoprune
    except Exception:
        NoraAutoPruneConfig = None
        maybe_apply_nora_autoprune = None

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
    _ACTION_TOKEN_RE = re.compile(r"<robot_action_(\d+)>")


    def __init__(
        self,
        model_path: str ="declare-lab/nora",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16, # Make dtype configurable
        autoprune_config: Optional[Any] = None,
        baseline_pruning_config: Optional[Any] = None,
        **kwargs,
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
        image_resize_size = kwargs.get("image_resize_size", (224, 224))
        if isinstance(image_resize_size, int):
            image_resize_size = (int(image_resize_size), int(image_resize_size))
        self.image_resize_size = tuple(int(v) for v in image_resize_size)
        self.dual_view_mode = str(kwargs.get("dual_view_mode", "separate"))
        self.use_anyres = bool(kwargs.get("use_anyres", False))

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
        except Exception as e:
            raise RuntimeError(f"Error loading main processor from {model_path}: {e}")

        full_model_state_path = kwargs.get("full_model_state_path")
        lora_adapter_path = kwargs.get("lora_adapter_path")

        # --- Load Main Model ---
        try:
            print(f"Loading model from: {model_path}")
            if full_model_state_path:
                from peft import LoraConfig, PeftConfig, get_peft_model
                from safetensors.torch import load_file

                print(f"Loading PEFT-wrapped local state from: {full_model_state_path}")
                adapter_config_path = lora_adapter_path or str(Path(full_model_state_path).parent / "adapter")
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                self.model = Qwen2_5_VLForConditionalGeneration(config)
                state_dict = load_file(str(full_model_state_path), device="cpu")
                try:
                    peft_config = PeftConfig.from_pretrained(adapter_config_path)
                except Exception as peft_config_error:
                    lora_a_key = next(
                        key for key in state_dict
                        if key.endswith(".lora_A.default.weight")
                    )
                    inferred_rank = int(state_dict[lora_a_key].shape[0])
                    print(
                        "Could not load adapter_config.json from "
                        f"{adapter_config_path}: {peft_config_error}. "
                        f"Inferring LoRA config with rank={inferred_rank}."
                    )
                    peft_config = LoraConfig(
                        r=inferred_rank,
                        lora_alpha=2 * inferred_rank,
                        lora_dropout=0.0,
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=[
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                            "merger.mlp.0",
                            "merger.mlp.2",
                        ],
                    )
                self.model = get_peft_model(self.model, peft_config)
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print(
                    "Loaded local PEFT state: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )
                if missing[:8]:
                    print(f"First missing keys: {missing[:8]}")
                if unexpected[:8]:
                    print(f"First unexpected keys: {unexpected[:8]}")
                self.model = self.model.merge_and_unload()
            else:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                #    attn_implementation="flash_attention_2", # Comment out this line if there is an error with flash attention
                )
            self.model.to(self.device)
            # Load generation config and set specific parameters
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
            self.model.generation_config.do_sample = False # Ensure deterministic output

            self.model.eval() # Set the model to evaluation mode
            self.autoprune_patch = None
            if (
                autoprune_config is not None
                and getattr(autoprune_config, "enabled", False)
                and maybe_apply_nora_autoprune is not None
            ):
                self.autoprune_patch = maybe_apply_nora_autoprune(
                    self.model,
                    autoprune_config,
                )
            self.baseline_pruning_patch = None
            if (
                baseline_pruning_config is not None
                and getattr(baseline_pruning_config, "enabled", False)
                and maybe_apply_nora_baseline_pruning is not None
            ):
                self.baseline_pruning_patch = maybe_apply_nora_baseline_pruning(
                    self.model,
                    baseline_pruning_config,
                )

            repo_id = "declare-lab/nora"
            filename = "norm_stats.json"

            # Download the norm_stats locally (only downloads once; cached)
            file_path = hf_hub_download(repo_id=repo_id, filename=filename)

            # Load the JSON file
            with open(file_path, "r") as f:
                norm_stats = json.load(f)
            self.norm_stats = norm_stats
            

        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")

        print("Model and processors loaded successfully.")

    def _extract_fast_action_token_ids(self, generated_ids: torch.Tensor, prompt_len: int = 0) -> list[int]:
        seq = generated_ids[0]
        if prompt_len > 0 and prompt_len < int(seq.shape[0]):
            seq = seq[prompt_len:]

        # Nora training serializes one action chunk as a contiguous span of
        # <robot_action_x> special tokens in the assistant response. For
        # extraction we therefore only keep the first contiguous action span in
        # the generated continuation instead of collecting every matching token
        # from the whole sequence.
        action_mask = (self._ACTION_TOKEN_MIN <= seq) & (seq <= self._ACTION_TOKEN_MAX)
        action_positions = torch.where(action_mask)[0]
        if action_positions.numel() <= 0:
            return []

        start = int(action_positions[0].item())
        end = start
        seq_len = int(seq.shape[0])
        while end < seq_len and int(self._ACTION_TOKEN_MIN) <= int(seq[end].item()) <= int(self._ACTION_TOKEN_MAX):
            end += 1

        action_span = seq[start:end]
        if action_span.numel() <= 0:
            return []
        return (action_span - self._ACTION_TOKEN_MIN).detach().cpu().tolist()

    def _decode_fast_action_tokens(self, fast_action_token_ids: list[int]) -> np.ndarray:
        action_dim = int(getattr(self.fast_tokenizer, "action_dim", 7))
        target_horizon = int(getattr(self.fast_tokenizer, "time_horizon", 1))
        fallback = np.zeros((1, target_horizon, action_dim), dtype=np.float32)
        if not fast_action_token_ids:
            return fallback
        try:
            decoded_action = self.fast_tokenizer.decode(
                [fast_action_token_ids],
                time_horizon=target_horizon,
                action_dim=action_dim,
            )
        except Exception as exc:
            print(f"Error decoding action tokens with fast tokenizer: {exc}")
            print(f"Tokens: {fast_action_token_ids}")
            return fallback
        if not isinstance(decoded_action, np.ndarray):
            decoded_action = np.array(decoded_action, dtype=np.float32)
        return decoded_action.astype(np.float32, copy=False)
    
    @torch.inference_mode()
    def inference(self, image: np.ndarray, image_wrist_or_instruction=None, instruction: Optional[str] = None, unnorm_key: str = None,unnormalizer=None) -> np.ndarray:
        """
        Performs inference to get robotic actions based on an image and instruction.

        Args:
            image (np.ndarray): The input image as a NumPy array (H, W, C).
            instruction (str): The natural language instruction.
            unnorm_key (str, optional): Key to select normalization statistics for unnormalizing actions.
            unnormalizer (lerobot.policies.normalize.Unnormalize, optional): If a Lerobot Unnormalizer is provided, it will be used to unnormalize the action.

        Returns:
            np.ndarray: The predicted unnormalized robotic action array.
        """
        inference_t0 = time.perf_counter()
        self.last_inference_profile = None
        preprocess_t0 = time.perf_counter()
        if instruction is None:
            instruction = image_wrist_or_instruction
        # --- Prepare Inputs ---
        # Ensure image is PIL Image for processor compatibility
        if not isinstance(image, PIL.Image.Image):
             image = PIL.Image.fromarray(image)

        resize_h, resize_w = self.image_resize_size
        content = [
            {
                "type": "image",
                "image": image,
                "resized_height": resize_h,
                "resized_width": resize_w,
            }
        ]
        image_wrist = image_wrist_or_instruction if instruction is not None else None
        if image_wrist is not None:
            if not isinstance(image_wrist, PIL.Image.Image):
                image_wrist = PIL.Image.fromarray(image_wrist)
            if self.dual_view_mode == "stitched":
                stitched = PIL.Image.new("RGB", (image.width + image_wrist.width, max(image.height, image_wrist.height)))
                stitched.paste(image.convert("RGB"), (0, 0))
                stitched.paste(image_wrist.convert("RGB"), (image.width, 0))
                content[0]["image"] = stitched
                content[0]["resized_width"] = resize_w * 2
            else:
                content.append(
                    {
                        "type": "image",
                        "image": image_wrist,
                        "resized_height": resize_h,
                        "resized_width": resize_w,
                    }
                )

        content.append({"type": "text", "text": instruction})

        # Construct messages in the expected chat format.
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Apply chat template to get the text input for the model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision information (depends on your process_vision_info function)
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs for the model using the main processor
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        preprocess_seconds = time.perf_counter() - preprocess_t0

        # --- Generate Output ---
        generate_t0 = time.perf_counter()
        generated_ids = self.model.generate(**inputs)
        generate_seconds = time.perf_counter() - generate_t0
        patch_stats = {}
        patch_ref = getattr(self.model, "_nora_baseline_pruning_patch", None) or getattr(self.model, "_autoprune_patch", None)
        if patch_ref is not None and hasattr(patch_ref, "get_runtime_stats"):
            patch_stats = patch_ref.get_runtime_stats()
        self.last_inference_profile = {
            "inference_total_seconds": time.perf_counter() - inference_t0,
            "preprocess_seconds": preprocess_seconds,
            "generate_seconds": generate_seconds,
            "decode_seconds": 0.0,
            "vit_seconds": 0.0,
            "prefill_seconds": 0.0,
            "decode_model_seconds": 0.0,
            **patch_stats,
        }

    

        # --- Extract and Decode Action ---
        # Find the indices of tokens within the action token range
       
        
        prompt_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        fast_action_token_ids = self._extract_fast_action_token_ids(generated_ids, prompt_len=prompt_len)
        if self.last_inference_profile is not None:
            self.last_inference_profile["generated_token_count"] = float(generated_ids.shape[-1])
            self.last_inference_profile["prompt_token_count"] = float(prompt_len)
            self.last_inference_profile["action_token_count"] = float(len(fast_action_token_ids))
            self.last_inference_profile["expected_coeff_count"] = float(
                int(getattr(self.fast_tokenizer, "time_horizon", 1)) * int(getattr(self.fast_tokenizer, "action_dim", 7))
            )
        output_action = self._decode_fast_action_tokens(fast_action_token_ids)
        
        if unnormalizer is not None: ## If a Lerobot Unnormalizer is provided, use it to unnormalize the action
            #
            unnormalized_action = unnormalizer({'action':output_action})
            if self.last_inference_profile is not None:
                self.last_inference_profile["inference_total_seconds"] = time.perf_counter() - inference_t0
            return unnormalized_action['action']
       

        # --- Denormalize Action ---
        # Assuming output_action is a numpy array of shape (1, time_horizon, action_dim)
        # and the values are in the range [-1, 1]
        # The formula is: unnormalized = 0.5 * (normalized + 1) * (high - low) + low

        '''We use the norm stats computed from OpenVLA https://arxiv.org/abs/2406.09246'''
        if 'libero' in unnorm_key:
            action_low,action_high = self.libero_keys[unnorm_key]
        else:
            action_norm_stats = self.get_action_stats(unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])

        unnorm_actions = (
            0.5 * (output_action + 1) * (action_high - action_low)
            + action_low
        )

      

        if self.last_inference_profile is not None:
            self.last_inference_profile["inference_total_seconds"] = time.perf_counter() - inference_t0
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
    
