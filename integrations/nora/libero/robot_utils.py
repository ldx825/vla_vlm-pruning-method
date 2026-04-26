"""Utility helpers for NORA LIBERO evaluation."""

import os
import random
import time
from typing import Any, Union

import numpy as np
import torch


DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

MODEL_IMAGE_SIZES = {
	"nora": 224,
	"openvla": 224,
}


def normalize_resize_size(resize_size: Any) -> tuple[int, int]:
	if isinstance(resize_size, int):
		return (resize_size, resize_size)
	if isinstance(resize_size, tuple) and len(resize_size) == 2:
		return (int(resize_size[0]), int(resize_size[1]))
	if isinstance(resize_size, list) and len(resize_size) == 2:
		return (int(resize_size[0]), int(resize_size[1]))
	raise ValueError(f"Expected resize size to be an int or length-2 tuple/list, got {resize_size!r}")


def set_seed_everywhere(seed: int) -> None:
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	os.environ["PYTHONHASHSEED"] = str(seed)


def get_image_resize_size(cfg: Any) -> Union[int, tuple[int, int]]:
	configured_resize = getattr(cfg, "image_resize_resolution", None)
	if configured_resize is not None:
		return normalize_resize_size(configured_resize)
	model_family = getattr(cfg, "model_family", "nora")
	return normalize_resize_size(MODEL_IMAGE_SIZES.get(model_family, 224))


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
	normalized_action = np.array(action, copy=True)
	normalized_action[..., -1] = 2 * normalized_action[..., -1] - 1
	if binarize:
		normalized_action[..., -1] = np.sign(normalized_action[..., -1])
	return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
	inverted_action = np.array(action, copy=True)
	inverted_action[..., -1] *= -1.0
	return inverted_action
