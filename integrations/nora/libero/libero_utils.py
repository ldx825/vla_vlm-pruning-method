"""Utils for evaluating NORA in LIBERO simulation environments."""

import math
import os
from typing import Union

import imageio
import numpy as np
from PIL import Image
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from robot_utils import DATE, DATE_TIME


try:
	import tensorflow as tf
except Exception:
	tf = None

try:
	if tf is not None:
		tf.config.set_visible_devices([], "GPU")
except RuntimeError:
	pass


def get_libero_env(task, model_family, resolution=448):
	task_description = task.language
	task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
	render_gpu_device_id = int(os.environ.get("NORA_LIBERO_RENDER_GPU_DEVICE_ID", "-1"))
	env_args = {
		"bddl_file_name": task_bddl_file,
		"camera_heights": resolution,
		"camera_widths": resolution,
		"render_gpu_device_id": render_gpu_device_id,
	}
	env = OffScreenRenderEnv(**env_args)
	env.seed(0)
	return env, task_description


def get_libero_dummy_action(model_family: str):
	return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img: np.ndarray, resize_size: Union[int, tuple[int, int]]) -> np.ndarray:
	if isinstance(resize_size, int):
		size = (resize_size, resize_size)
	else:
		size = resize_size
	if tf is None:
		# Match the original TensorFlow path's high-quality Lanczos resizing
		# without requiring TensorFlow in debug/eval-only environments.
		pil_img = Image.fromarray(img.astype(np.uint8, copy=False))
		return np.asarray(pil_img.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)
	img = tf.image.encode_jpeg(img)
	img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
	img = tf.image.resize(img, size, method="lanczos3", antialias=True)
	img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
	return img.numpy()


def get_libero_images(obs, resize_size, use_anyres=False):
	head_image = obs["agentview_image"]
	wrist_image = obs["robot0_eye_in_hand_image"]
	head_image = head_image[::-1, ::-1]
	wrist_image = wrist_image[::-1, ::-1]
	if use_anyres:
		return {
			"image_primary": head_image,
			"image_wrist": wrist_image,
		}
	return {
		"image_primary": resize_image(head_image, resize_size),
		"image_wrist": resize_image(wrist_image, resize_size),
	}


def get_libero_video_frame(obs) -> np.ndarray:
	return obs["agentview_image"][::-1, ::-1]


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
	rollout_root = os.environ.get("NORA_ROLLOUT_DIR")
	if not rollout_root:
		rollout_root = "/cache/ma-user/nora_rollouts" if os.path.isdir("/cache/ma-user") else "./rollouts"
	rollout_dir = os.path.join(rollout_root, DATE)
	os.makedirs(rollout_dir, exist_ok=True)
	processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
	mp4_path = (
		f"{rollout_dir}/{DATE_TIME}--nora--episode={idx}--success={success}"
		f"--task={processed_task_description}.mp4"
	)
	video_writer = imageio.get_writer(mp4_path, fps=30)
	for img in rollout_images:
		video_writer.append_data(img)
	video_writer.close()
	print(f"Saved rollout MP4 at path {mp4_path}")
	if log_file is not None:
		log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
	return mp4_path


def quat2axisangle(quat):
	quat = np.array(quat, copy=True)
	quat[3] = np.clip(quat[3], -1.0, 1.0)
	den = np.sqrt(1.0 - quat[3] * quat[3])
	if math.isclose(den, 0.0):
		return np.zeros(3)
	return (quat[:3] * 2.0 * math.acos(quat[3])) / den
