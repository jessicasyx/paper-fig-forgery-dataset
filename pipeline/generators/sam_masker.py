# pipeline/generators/sam_masker.py
"""
SAM Mask Generator
负责：
- 默认配置
- 初始化 SAM 模型
- 选择最佳 mask
- 对单张图像生成 mask（返回mask图像数组）
"""

import cv2
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def get_default_sam_config() -> Dict:
    """
    默认 SAM 配置（你之前调优过的版本）
    """
    return {
        "model_type": "vit_b",      # "vit_b", "vit_l", "vit_h"
        "max_image_size": 800,      # 缩放最大边长
        "resize_enabled": True,
        "invert_mask": True,        # True=keep_mask: 黑=编辑区域，白=保留区域
        "sam_params": {
            "points_per_side": 16,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "crop_n_layers": 0,
            "min_mask_region_area": 500,
        }
    }


def initialize_sam_model(
    checkpoint_path: str,
    model_type: str = "vit_b",
    sam_params: Optional[Dict] = None
) -> Tuple[SamAutomaticMaskGenerator, str]:
    """
    初始化 SAM 模型与 mask generator（只需初始化一次）
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("[SAM] Loading model...")
    print(f"[SAM] device = {device}")

    if device.startswith("cuda"):
        print(f"[SAM] GPU = {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[SAM] VRAM = {mem_gb:.2f} GB")

    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
    except Exception as e:
        raise RuntimeError(f"SAM model load failed: {repr(e)}")

    if sam_params is None:
        sam_params = get_default_sam_config()["sam_params"]

    mask_generator = SamAutomaticMaskGenerator(model=sam, **sam_params)
    print("[SAM] ✅ model loaded.")
    return mask_generator, device


def select_best_mask(masks: List[Dict], image_area: int) -> Optional[Dict]:
    """
    选择最佳 mask（你原来的策略）
    """
    if not masks:
        return None


    for m in masks:
        m["area_ratio"] = m["area"] / max(image_area, 1)

    # 优先：10%~30%
    optimal = [m for m in masks if 0.10 <= m["area_ratio"] <= 0.30]
    if optimal:
        return min(optimal, key=lambda m: abs(m["area_ratio"] - 0.20))

    # 退一步：5%~40%
    acceptable = [m for m in masks if 0.05 <= m["area_ratio"] <= 0.40]
    if acceptable:
        return min(acceptable, key=lambda m: abs(m["area_ratio"] - 0.20))

    # 再退：面积第二大
    if len(masks) >= 2:
        masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
        return masks_sorted[1]

    # 最后：最大
    return max(masks, key=lambda m: m["area"])


def make_fallback_mask(h: int, w: int, invert_mask: bool = True) -> np.ndarray:
    """
    生成 fallback mask：全黑或全白（与你之前一致）
    默认：全黑 -> invert后变全白（表示“全保留”）
    """
    roi_mask = np.zeros((h, w), dtype=np.uint8)  # 0=黑
    if not invert_mask:
        roi_mask = 255 - roi_mask
    return roi_mask


def sam_generate_mask_image(
    image_bgr: np.ndarray,
    mask_generator: SamAutomaticMaskGenerator,
    original_hw: Tuple[int, int],
    resize_enabled: bool = True,
    max_image_size: int = 800,
    invert_mask: bool = True,
) -> Tuple[np.ndarray, str]:
    """
    对单张图像生成 mask（返回 mask_gray uint8 + 描述信息）
    输出 mask 是 keep_mask 语义：
      invert_mask=True -> 黑=编辑区域，白=保留区域
    """
    orig_h, orig_w = original_hw

    if image_bgr is None:
        mask = make_fallback_mask(orig_h, orig_w, invert_mask=invert_mask)
        return mask, "read_failed_fallback"

    # 记录原始尺寸
    h0, w0 = image_bgr.shape[:2]

    # 可选缩放：提升速度/省显存
    if resize_enabled:
        max_dim = max(h0, w0)
        if max_dim > max_image_size:
            scale = max_image_size / max_dim
            new_w = int(w0 * scale)
            new_h = int(h0 * scale)
            image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image_rgb)

    if len(masks) == 0:
        mask = make_fallback_mask(orig_h, orig_w, invert_mask=invert_mask)
        return mask, "no_mask_fallback"

    cur_area = image_rgb.shape[0] * image_rgb.shape[1]
    best = select_best_mask(masks, cur_area)
    if best is None:
        mask = make_fallback_mask(orig_h, orig_w, invert_mask=invert_mask)
        return mask, "select_failed_fallback"

    roi_mask = (best["segmentation"].astype(np.uint8) * 255)  # 255=前景区域
    ratio = best.get("area_ratio", 0) * 100.0

    # resize回原始尺寸（如果缩放过）
    if roi_mask.shape[0] != orig_h or roi_mask.shape[1] != orig_w:
        roi_mask = cv2.resize(roi_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 反转成 keep_mask：黑=编辑，白=保留
    if invert_mask:
        roi_mask = 255 - roi_mask

    msg = f"ok (masks={len(masks)}, selected_area={ratio:.1f}%)"
    return roi_mask, msg
