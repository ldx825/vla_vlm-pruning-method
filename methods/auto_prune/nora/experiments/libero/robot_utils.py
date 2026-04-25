"""Robot utility shim for the NORA pruning eval entrypoint."""

import os
import random
import time
from typing import Any

import numpy as np
import torch

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_image_resize_size(cfg: Any):
    return tuple(getattr(cfg, "image_resize_resolution", (448, 448)))


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized_action = action.copy()
    normalized_action[..., -1] = 2 * normalized_action[..., -1] - 1
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted_action = action.copy()
    inverted_action[..., -1] *= -1.0
    return inverted_action
