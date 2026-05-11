"""Compatibility entry point for the corrected FastV OpenVLA-OFT patch.

The actual implementation lives in `fastv_patch_improved_1.py`.  Keeping this
module name preserves the existing `run_libero_eval.py` import path and all
current command-line flags.
"""

from fastv_patch_improved_1 import FastVConfig, FastVPatch, maybe_apply_fastv_patch

__all__ = ["FastVConfig", "FastVPatch", "maybe_apply_fastv_patch"]
