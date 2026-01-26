import cv2
import numpy as np

def eval_mask_qa(mask_path:str, min_ratio=0.05, max_ratio=0.20, max_edge=0.10) -> dict:
    """
    评估 mask 区域的质量，返回评估结果（占比、边界一致性评分）。
    """
    # 读取 mask 图像（支持中文路径）
    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"无法读取图像文件: {mask_path}")

    # 二值化处理，0为伪造区域，255为保留区域
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # 使用膨胀和腐蚀操作来提取边界
    dilated_mask = cv2.dilate(binary_mask, np.ones((5, 5), np.uint8))  # 膨胀
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8))  # 腐蚀

    # 计算边界差异（膨胀与腐蚀后的差异）
    edge_diff = np.abs(dilated_mask - eroded_mask)
    
    # 计算边界差异的均值，越小表示边界越连续
    edge_diff_mean = np.mean(edge_diff)

    # 计算 mask 区域占比（伪造区域占总图像比例）
    mask_ratio = np.sum(binary_mask == 0) / binary_mask.size  # 0表示伪造区域

    # 判断 mask 是否合格（面积范围和边界一致性）
    pass_ratio = (min_ratio <= mask_ratio <= max_ratio)
    pass_edge  = (edge_diff_mean <= max_edge)

    passed = pass_ratio and pass_edge

    # reason ：给出不合格的具体偏离值
    if passed:
        reason = "ok"
    else:
        reasons = []
        if not pass_ratio:
            reason_ratio = f"ratio_out({mask_ratio:.4f}), range: [{min_ratio}, {max_ratio}]"
            reasons.append(reason_ratio)
        if not pass_edge:
            reason_edge = f"edge_out({edge_diff_mean:.4f}), max_edge: {max_edge}"
            reasons.append(reason_edge)
        reason = " | ".join(reasons)

    return {
        "mask_ratio": float(mask_ratio),
        "edge_diff_mean": edge_diff_mean, 
        "pass": bool(passed),
        "reason": reason,
        "range": [min_ratio, max_ratio],
        "max_edge": max_edge,
    }
   

def eval_fake_qa(real_path, fake_path) -> dict:
    """
    评估 fake 图像的质量，返回评估结果（差异、合法性）。
    """
    # 读取 real 和 fake 图像（支持中文路径）
    real = cv2.imdecode(np.fromfile(real_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    fake = cv2.imdecode(np.fromfile(fake_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    # 计算 fake 和 real 的差异,差异越小表示生成的 fake 越真实。
    diff = np.abs(fake - real)
    diff_mean = np.mean(diff)  # 平均差异
    # 计算差异大于 10 的像素比例，表示图像差异的“局部严重性”。
    diff_ratio = np.mean(diff > 10)  # 假设 10 为阈值，表示差异较大的像素

    # fake 合法性检查：判断 fake 图像是否全黑或全白，防止生成的 fake 图像没有任何有效内容。
    fake_valid = np.sum(fake > 0) > 0 and np.sum(fake < 255) > 0  # 非全黑非全白

    # 计算 fake 和 real 的相似度（如果需要更详细的判断）
    similarity = np.mean(fake == real)

    # 判断 fake 是否通过 QA
    passed = fake_valid and diff_mean < 20

    reason = "ok" if passed else "fake_invalid"

    return {
        "diff_mean": diff_mean,
        "diff_ratio": diff_ratio,
        "fake_valid": fake_valid,
        "similarity": similarity,
        "pass": passed,        # 加入 pass 字段
        "reason": reason
    }
