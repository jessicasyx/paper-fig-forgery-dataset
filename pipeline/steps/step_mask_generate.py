# pipeline/steps/step_mask_generate.py
"""
Step: 生成 mask（单样本 one-id）
输入：real_path
输出：mask_path（PNG，keep_mask语义：黑=编辑/白=保留）
"""

import os
import cv2
import numpy as np
import torch
from typing import Dict, Tuple

from pipeline.generators.sam_masker import sam_generate_mask_image


def _read_image_bgr_bytes(path: str) -> np.ndarray:
    """
    读取图片（支持中文路径），返回 BGR
    """
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _save_mask_png(mask_gray: np.ndarray, out_path: str) -> bool:
    """
    保存 mask PNG（支持中文路径）
    """
    try:
        ok, encoded = cv2.imencode(".png", mask_gray)
        if not ok:
            return False
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(encoded.tobytes())
        return True
    except Exception:
        return False


def run_step_mask_generate(
    real_path: str,
    mask_out_path: str,
    mask_generator,
    device: str,
    config: Dict,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """
    处理单张图片，生成 mask
    返回：(ok, msg)
    """
    try:
        if (not overwrite) and os.path.exists(mask_out_path):
            return True, "exists_skip"

        if not os.path.exists(real_path):
            return False, "real_not_found"

        # 读图
        image_bgr = _read_image_bgr_bytes(real_path)
        if image_bgr is None:
            # fallback：创建一个 100x100 mask
            mask = np.zeros((100, 100), dtype=np.uint8)
            if not config.get("invert_mask", True):
                mask = 255 - mask
            ok = _save_mask_png(mask, mask_out_path)
            return (ok, "read_failed_fallback_saved" if ok else "read_failed_fallback_save_failed")

        orig_h, orig_w = image_bgr.shape[:2]

        # SAM 推理生成 mask
        mask_gray, info = sam_generate_mask_image(
            image_bgr=image_bgr,
            mask_generator=mask_generator,
            original_hw=(orig_h, orig_w),
            resize_enabled=config.get("resize_enabled", True),
            max_image_size=config.get("max_image_size", 800),
            invert_mask=config.get("invert_mask", True),
        )

        ok = _save_mask_png(mask_gray, mask_out_path)
        if not ok:
            return False, "mask_save_failed"

        return True, info

    except Exception as e:
        return False, f"exception:{repr(e)}"

    finally:
        # 清理显存（可选）
        if device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
