# pipeline/steps/step_copy_move.py

import cv2
import numpy as np
from pipeline.generators.sam_masker import sam_generate_mask_image,initialize_sam_model
from segment_anything import SamAutomaticMaskGenerator

def generate_mask_with_sam(image, checkpoint_path, model_type="vit_l", max_image_size=800, resize_enabled=True):
    """
    使用 SAM 生成掩码
    :param image: 输入图像
    :param checkpoint_path: SAM 模型检查点路径
    :param model_type: 模型类型（vit_b, vit_l, vit_h）
    :param max_image_size: 缩放最大边长
    :param resize_enabled: 是否启用缩放
    :return: 掩码图像
    """
    # 加载模型生成器
    mask_generator, device = initialize_sam_model(checkpoint_path, model_type)
    
    # 使用 SAM 生成掩码
    mask, msg = sam_generate_mask_image(image, mask_generator, image.shape[:2], resize_enabled, max_image_size)
    return mask, msg

def adjust_brightness_contrast(image, alpha=1.2, beta=50):
    """
    调整图像的亮度和对比度
    :param image: 输入图像
    :param alpha: 对比度（1.0-3.0）
    :param beta: 亮度（0-100）
    :return: 调整后的图像
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def select_source_and_target_from_mask(image, mask, region_size=(50, 50)):
    """
    选择源区域（黑色区域）和目标区域（白色区域）从生成的 mask 中
    :param image: 输入图像
    :param mask: SAM 生成的二值化掩码（黑色部分为源区域，白色部分为目标区域）
    :param region_size: 选择的区域大小
    :return: 源区域和目标区域的坐标
    """
    H, W, _ = image.shape
    pw, ph = region_size

    # 获取源区域（黑色区域）
    source_mask = (mask == 0)  # 黑色区域
    source_coords = np.argwhere(source_mask)  # 获取源区域的坐标
    if len(source_coords) > 0:
        sy, sx = source_coords[0]  # 选择第一个源区域坐标
        source_region = image[sy:sy + ph, sx:sx + pw]
    else:
        source_region = image[0:ph, 0:pw]

    # 获取目标区域（白色区域）
    target_mask = (mask == 255)  # 白色区域
    target_coords = np.argwhere(target_mask)  # 获取目标区域的坐标
    if len(target_coords) > 0:
        dy, dx = target_coords[0]  # 选择第一个目标区域坐标
        target_region = (dx, dy, pw, ph)
    else:
        target_region = (0, 0, pw, ph)

    return source_region, target_region

def feathered_edge_mask(mask, feather_width=5):
    """
    对掩码进行羽化处理，使边缘更加平滑
    :param mask: 输入掩码（二值化图像）
    :param feather_width: 羽化宽度（像素）
    :return: 羽化后的掩码
    """
    # 使用高斯模糊进行羽化
    feathered = cv2.GaussianBlur(mask, (feather_width * 2 + 1, feather_width * 2 + 1), 0)
    return feathered

def paste_source_to_target(image, source_region, target_coordinates, feathered_mask):
    """
    将源区域粘贴到目标区域，使用羽化掩码进行混合
    :param image: 原始图像
    :param source_region: 源区域图像
    :param target_coordinates: 目标区域坐标 (x, y, w, h)
    :param feathered_mask: 羽化后的掩码
    :return: 粘贴后的图像
    """
    result = image.copy()
    dx, dy, pw, ph = target_coordinates
    
    # 确保源区域和目标区域大小匹配
    if source_region.shape[0] != ph or source_region.shape[1] != pw:
        source_region = cv2.resize(source_region, (pw, ph))
    
    # 确保不超出图像边界
    h, w = image.shape[:2]
    if dy + ph > h:
        ph = h - dy
        source_region = source_region[:ph, :]
    if dx + pw > w:
        pw = w - dx
        source_region = source_region[:, :pw]
    
    # 提取对应区域的羽化掩码
    mask_region = feathered_mask[dy:dy+ph, dx:dx+pw]
    
    # 归一化掩码到 [0, 1]
    if mask_region.max() > 1:
        mask_region = mask_region.astype(np.float32) / 255.0
    else:
        mask_region = mask_region.astype(np.float32)
    
    # 扩展掩码维度以匹配彩色图像
    if len(mask_region.shape) == 2:
        mask_region = np.expand_dims(mask_region, axis=2)
    
    # 使用掩码进行混合
    target_region = result[dy:dy+ph, dx:dx+pw]
    blended = (source_region * mask_region + target_region * (1 - mask_region)).astype(np.uint8)
    result[dy:dy+ph, dx:dx+pw] = blended
    
    return result
