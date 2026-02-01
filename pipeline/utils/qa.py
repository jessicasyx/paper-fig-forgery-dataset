import cv2
import numpy as np

def eval_mask_qa(mask_path: str, real_shape: tuple = None) -> dict:
    """
    Mask QA - 硬筛选（必须筛）
    
    1.1 必杀项（直接淘汰）：
        - 读不出来 / 尺寸不一致
        - 全黑 / 全白
        - area_ratio < 0.005 或 area_ratio > 0.50
    
    1.2 分层（不淘汰，只标记）：
        - Tier A（优先）：0.05–0.20
        - Tier B（可用）：0.02–0.40（排除 A 的部分）
        - Tier C（保留但标记）：0.005–0.02 和 0.40–0.50
    
    返回：
        {
            "hard_fail": bool,  # 是否必杀（淘汰）
            "reason": str,      # 失败原因
            "area_ratio": float,
            "tier": str,        # "A" / "B" / "C" / "FAIL"
            "is_all_black": bool,
            "is_all_white": bool,
        }
    """
    result = {
        "hard_fail": False,
        "reason": "ok",
        "area_ratio": 0.0,
        "tier": "FAIL",
        "is_all_black": False,
        "is_all_white": False,
    }
    
    # 1. 读取 mask 图像（支持中文路径）
    try:
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        result["hard_fail"] = True
        result["reason"] = f"read_failed: {repr(e)}"
        return result
    
    if mask is None:
        result["hard_fail"] = True
        result["reason"] = "mask_is_none"
        return result
    
    # 2. 检查尺寸一致性（如果提供了real_shape）
    if real_shape is not None:
        if mask.shape[:2] != real_shape[:2]:
            result["hard_fail"] = True
            result["reason"] = f"size_mismatch: mask={mask.shape[:2]}, real={real_shape[:2]}"
            return result
    
    # 3. 检查全黑 / 全白
    mask_min = np.min(mask)
    mask_max = np.max(mask)
    
    if mask_min == mask_max:
        if mask_min == 0:
            result["hard_fail"] = True
            result["reason"] = "all_black"
            result["is_all_black"] = True
            return result
        elif mask_min == 255:
            result["hard_fail"] = True
            result["reason"] = "all_white"
            result["is_all_white"] = True
            return result
    
    # 4. 计算 area_ratio（伪造区域占比）
    # 假设：0 = 伪造区域，255 = 保留区域
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    area_ratio = np.sum(binary_mask == 0) / binary_mask.size
    result["area_ratio"] = float(area_ratio)
    
    # 5. 必杀项：area_ratio 超出范围
    if area_ratio < 0.005 or area_ratio > 0.50:
        result["hard_fail"] = True
        result["reason"] = f"area_ratio_out_of_range: {area_ratio:.4f} (must be 0.005-0.50)"
        result["tier"] = "FAIL"
        return result
    
    # 6. 分层标记（不淘汰）
    if 0.05 <= area_ratio <= 0.20:
        result["tier"] = "A"  # 优先
    elif 0.02 <= area_ratio < 0.05 or 0.20 < area_ratio <= 0.40:
        result["tier"] = "B"  # 可用
    elif 0.005 <= area_ratio < 0.02 or 0.40 < area_ratio <= 0.50:
        result["tier"] = "C"  # 保留但标记
    else:
        result["tier"] = "FAIL"
    
    result["reason"] = "ok"
    return result


def eval_fake_qa(real_path: str, fake_path: str, mask_path: str) -> dict:
    """
    Fake QA - 不严格筛，但"灾难级 hard fail"需要筛掉
    
    绝大多数样本：只打分、只记录，不淘汰
    只有灾难级：淘汰（否则污染数据集）
    
    2.1 通用核心指标（RePaint & Copy-Move 都用）：
        - diff_in = mean(|fake-real| on mask)
        - diff_out = mean(|fake-real| on ~mask)
        - ratio = diff_in / (diff_out + eps)
    
    2.2 灾难级 hard fail（筛掉）：
        - fake 读不出来 / 尺寸不对
        - fake 文件为空或明显损坏
        - 几乎没有伪造发生：diff_in 极小（接近 0）
        - 大范围污染：diff_out 极大（mask 外改得太多）
    
    返回：
        {
            "hard_fail": bool,
            "reason": str,
            "diff_in": float,      # mask内平均差异
            "diff_out": float,     # mask外平均差异
            "ratio": float,        # diff_in / diff_out
            "score": float,        # 综合评分（0-100）
        }
    """
    result = {
        "hard_fail": False,
        "reason": "ok",
        "diff_in": 0.0,
        "diff_out": 0.0,
        "ratio": 0.0,
        "score": 0.0,
    }
    
    # 1. 读取 real 图像
    try:
        real = cv2.imdecode(np.fromfile(real_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        result["hard_fail"] = True
        result["reason"] = f"read_real_failed: {repr(e)}"
        return result
    
    if real is None:
        result["hard_fail"] = True
        result["reason"] = "real_is_none"
        return result
    
    # 2. 读取 fake 图像
    try:
        fake = cv2.imdecode(np.fromfile(fake_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        result["hard_fail"] = True
        result["reason"] = f"read_fake_failed: {repr(e)}"
        return result
    
    if fake is None:
        result["hard_fail"] = True
        result["reason"] = "fake_is_none"
        return result
    
    # 3. 检查尺寸一致性
    if real.shape != fake.shape:
        result["hard_fail"] = True
        result["reason"] = f"size_mismatch: real={real.shape}, fake={fake.shape}"
        return result
    
    # 4. 读取 mask
    try:
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        result["hard_fail"] = True
        result["reason"] = f"read_mask_failed: {repr(e)}"
        return result
    
    if mask is None:
        result["hard_fail"] = True
        result["reason"] = "mask_is_none"
        return result
    
    # 5. 检查 mask 尺寸
    if mask.shape[:2] != real.shape[:2]:
        result["hard_fail"] = True
        result["reason"] = f"mask_size_mismatch: mask={mask.shape[:2]}, real={real.shape[:2]}"
        return result
    
    # 6. 二值化 mask（0 = 伪造区域，255 = 保留区域）
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask_in = (binary_mask == 0)   # 伪造区域（True）
    mask_out = (binary_mask == 255) # 保留区域（True）
    
    # 7. 计算差异
    diff = np.abs(fake.astype(np.float32) - real.astype(np.float32))
    
    # diff_in: mask 内的平均差异
    if np.sum(mask_in) > 0:
        diff_in = np.mean(diff[mask_in])
    else:
        diff_in = 0.0
    
    # diff_out: mask 外的平均差异
    if np.sum(mask_out) > 0:
        diff_out = np.mean(diff[mask_out])
    else:
        diff_out = 0.0
    
    result["diff_in"] = float(diff_in)
    result["diff_out"] = float(diff_out)
    
    # 8. 计算 ratio
    eps = 1e-6
    ratio = diff_in / (diff_out + eps)
    result["ratio"] = float(ratio)
    
    # 9. 灾难级检查
    # 9.1 几乎没有伪造发生（diff_in 极小）
    if diff_in < 1.0:  # 平均差异小于1（几乎没变化）
        result["hard_fail"] = True
        result["reason"] = f"no_forgery: diff_in={diff_in:.2f} (too small)"
        return result
    
    # 9.2 大范围污染（diff_out 极大）
    if diff_out > 30.0:  # mask 外平均差异大于30（改得太多）
        result["hard_fail"] = True
        result["reason"] = f"large_contamination: diff_out={diff_out:.2f} (too large)"
        return result
    
    # 10. 计算综合评分（0-100）
    # 这个评分只是参考，不用于淘汰
    score = 50.0  # 基础分
    
    # diff_in 越大越好（说明伪造明显）
    if diff_in >= 10:
        score += 25
    elif diff_in >= 5:
        score += 15
    elif diff_in >= 2:
        score += 10
    
    # diff_out 越小越好（说明没有污染）
    if diff_out <= 5:
        score += 25
    elif diff_out <= 10:
        score += 15
    elif diff_out <= 20:
        score += 10
    
    result["score"] = float(score)
    result["reason"] = "ok"
    
    return result
