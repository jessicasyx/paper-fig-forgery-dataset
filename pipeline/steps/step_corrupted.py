# pipeline/steps/step_corrupted.py
"""
Step: 生成 corrupted（挖洞图像 / LRS）
输入：real + keep_mask
输出：corrupted

keep_mask 语义：
- 白(255) = 保留
- 黑(0)   = 需要修改（挖洞）
"""

import os
import numpy as np
from typing import List, Tuple, Optional

from pipeline.utils.io import imread_cn, imwrite_cn


def _generate_single_corrupted(
    real_bgr: np.ndarray,
    mask_gray: np.ndarray,
    fill_color: List[int] = [255, 255, 255],
) -> np.ndarray:
    """
    生成单张挖洞图（corrupted）
    mask_gray: 0 表示要挖洞的区域
    """
    corrupted = real_bgr.copy()
    corrupted[mask_gray == 0] = fill_color
    return corrupted


def run_step_corrupted(
    real_path: str,
    mask_keep_path: str,
    corrupted_path: str,
    fill_color: List[int] = [255, 255, 255],
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """
    处理单个样本（one-id），生成 corrupted

    返回:
      (ok, err_msg)
    """
    try:
        if (not overwrite) and os.path.exists(corrupted_path):
            return True, "exists_skip"

        if not os.path.exists(real_path):
            return False, "real_not_found"

        if not os.path.exists(mask_keep_path):
            return False, "mask_not_found"

        real = imread_cn(real_path)
        mask = imread_cn(mask_keep_path, flags=0)  # cv2.IMREAD_GRAYSCALE = 0

        if real is None:
            return False, "read_real_failed"
        if mask is None:
            return False, "read_mask_failed"

        if real.shape[:2] != mask.shape[:2]:
            return False, f"size_mismatch real={real.shape[:2]} mask={mask.shape[:2]}"

        corrupted = _generate_single_corrupted(real, mask, fill_color)

        os.makedirs(os.path.dirname(corrupted_path), exist_ok=True)
        ok = imwrite_cn(corrupted, corrupted_path, ext=".png")
        if not ok:
            return False, "write_failed"

        return True, ""

    except Exception as e:
        return False, f"exception:{repr(e)}"
