# pipeline/generators/copy_move.py

import os
import cv2
import numpy as np
from pathlib import Path
from pipeline.steps.step_copy_move import generate_mask_with_sam, feathered_edge_mask, adjust_brightness_contrast

def simulate_jpeg_compression(image, quality=50):
    """
    模拟 JPEG 压缩过程
    :param image: 输入图像
    :param quality: 压缩质量（0 - 最低，100 - 最高）
    :return: 模拟压缩后的图像
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_image = cv2.imencode('.jpg', image, encode_param)
    
    # 解压（模拟 JPEG 解压）
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
    
    return decoded_image

def extract_object_from_mask(image, mask):
    """
    从 mask 中提取对象区域
    :param image: 输入图像
    :param mask: SAM 生成的 mask（黑=编辑区域，白=保留区域）
    :return: (对象图像, 对象mask, 边界框(x, y, w, h))
    """
    # 反转 mask：黑色区域是我们要复制的对象
    object_mask = (mask == 0).astype(np.uint8) * 255
    
    # 找到对象的轮廓
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, None, None
    
    # 选择最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 提取对象区域
    object_img = image[y:y+h, x:x+w].copy()
    object_mask_region = object_mask[y:y+h, x:x+w].copy()
    
    return object_img, object_mask_region, (x, y, w, h)

def find_valid_paste_location(image_shape, object_size, source_bbox, margin=20):
    """
    找到一个有效的粘贴位置（不与源区域重叠）
    :param image_shape: 图像尺寸 (h, w)
    :param object_size: 对象尺寸 (w, h)
    :param source_bbox: 源区域边界框 (x, y, w, h)
    :param margin: 与源区域的最小间距
    :return: 目标位置 (x, y) 或 None
    """
    h, w = image_shape[:2]
    obj_w, obj_h = object_size
    src_x, src_y, src_w, src_h = source_bbox
    
    # 尝试多次随机选择位置
    for _ in range(50):
        # 随机选择位置
        target_x = np.random.randint(0, max(1, w - obj_w))
        target_y = np.random.randint(0, max(1, h - obj_h))
        
        # 检查是否与源区域重叠
        if (target_x + obj_w < src_x - margin or target_x > src_x + src_w + margin or
            target_y + obj_h < src_y - margin or target_y > src_y + src_h + margin):
            return (target_x, target_y)
    
    return None

def paste_object_with_mask(image, object_img, object_mask, target_pos, feather_width=5):
    """
    将对象粘贴到目标位置，使用 mask 进行混合
    :param image: 原始图像
    :param object_img: 对象图像
    :param object_mask: 对象 mask
    :param target_pos: 目标位置 (x, y)
    :param feather_width: 羽化宽度
    :return: 粘贴后的图像
    """
    result = image.copy()
    tx, ty = target_pos
    h, w = object_img.shape[:2]
    
    # 确保不超出边界
    img_h, img_w = image.shape[:2]
    if ty + h > img_h:
        h = img_h - ty
        object_img = object_img[:h, :]
        object_mask = object_mask[:h, :]
    if tx + w > img_w:
        w = img_w - tx
        object_img = object_img[:, :w]
        object_mask = object_mask[:, :w]
    
    # 羽化 mask
    feathered_mask = feathered_edge_mask(object_mask, feather_width)
    
    # 归一化 mask 到 [0, 1]
    mask_normalized = feathered_mask.astype(np.float32) / 255.0
    if len(mask_normalized.shape) == 2:
        mask_normalized = np.expand_dims(mask_normalized, axis=2)
    
    # 混合
    target_region = result[ty:ty+h, tx:tx+w]
    blended = (object_img * mask_normalized + target_region * (1 - mask_normalized)).astype(np.uint8)
    result[ty:ty+h, tx:tx+w] = blended
    
    return result

def load_or_generate_mask(real_image, image_name, checkpoint_path, mask_dir, model_type="vit_l"):
    """
    加载已有的 mask 或生成新的 mask
    :param real_image: 输入图像
    :param image_name: 图像文件名（不含扩展名）
    :param checkpoint_path: SAM 模型检查点路径
    :param mask_dir: mask 存储目录
    :param model_type: 模型类型
    :return: mask 图像
    """
    # 构建 mask 文件路径
    mask_path = os.path.join(mask_dir, f"{image_name}.png")
    
    # 检查 mask 是否已存在
    if os.path.exists(mask_path):
        print(f"  [1/6] 加载已有 mask: {image_name}.png")
        try:
            # 使用支持中文路径的方式读取
            with open(mask_path, "rb") as f:
                mask_data = np.frombuffer(f.read(), np.uint8)
            mask = cv2.imdecode(mask_data, cv2.IMREAD_GRAYSCALE)
            
            if mask is not None:
                print(f"  Mask 加载成功")
                return mask
            else:
                print(f"  [WARN] Mask 文件损坏，重新生成...")
        except Exception as e:
            print(f"  [WARN] 读取 mask 失败: {repr(e)}，重新生成...")
    
    # mask 不存在或损坏，使用 SAM 生成
    print(f"  [1/6] 使用 SAM 生成 mask...")
    mask, msg = generate_mask_with_sam(real_image, checkpoint_path, model_type)
    print(f"  SAM: {msg}")
    
    # 保存 mask 到文件（支持中文路径）
    try:
        os.makedirs(mask_dir, exist_ok=True)
        ok, encoded_mask = cv2.imencode('.png', mask)
        if ok:
            with open(mask_path, 'wb') as f:
                f.write(encoded_mask.tobytes())
            print(f"  Mask 已保存: {image_name}.png")
        else:
            print(f"  [WARN] Mask 编码失败，未保存")
    except Exception as e:
        print(f"  [WARN] 保存 mask 失败: {repr(e)}")
    
    return mask

def generate_fake_images_from_real(real_image, checkpoint_path, num_samples=10, region_size=(50, 50), model_type="vit_l", feather_width=5, image_name=None, mask_dir=None):
    """
    从真实图像生成伪造图像（Copy-Move）
    :param real_image: 输入图像
    :param checkpoint_path: SAM 模型检查点路径
    :param num_samples: 伪造图像的数量
    :param region_size: 未使用（保留兼容性）
    :param model_type: 模型类型（vit_b, vit_l, vit_h）
    :param feather_width: 羽化宽度
    :param image_name: 图像文件名（用于 mask 缓存）
    :param mask_dir: mask 存储目录
    :return: 生成的伪造图像列表
    """
    # 1. 加载或生成 mask
    if image_name and mask_dir:
        mask = load_or_generate_mask(real_image, image_name, checkpoint_path, mask_dir, model_type)
    else:
        # 兼容旧代码：直接生成 mask
        print("  [1/6] 生成 SAM mask...")
        mask, msg = generate_mask_with_sam(real_image, checkpoint_path, model_type)
        print(f"  SAM: {msg}")
    
    # 2. 从 mask 中提取对象
    print("  [2/6] 提取对象区域...")
    object_img, object_mask, source_bbox = extract_object_from_mask(real_image, mask)
    
    if object_img is None:
        print("  [ERROR] 无法从 mask 中提取对象")
        return []
    
    print(f"  对象大小: {object_img.shape[1]}x{object_img.shape[0]}")
    
    fake_images = []
    print(f"  [3/6] 生成 {num_samples} 张伪造图像...")
    
    for i in range(num_samples):
        # 3. 找到有效的粘贴位置
        target_pos = find_valid_paste_location(
            real_image.shape, 
            (object_img.shape[1], object_img.shape[0]), 
            source_bbox
        )
        
        if target_pos is None:
            print(f"    样本 {i+1}: 无法找到有效粘贴位置，跳过")
            continue
        
        # 4. 粘贴对象到目标位置
        fake_image = paste_object_with_mask(real_image, object_img, object_mask, target_pos, feather_width)
        
        # 5. 调整亮度和对比度（轻微变化）
        fake_image = adjust_brightness_contrast(fake_image, alpha=1.05, beta=10)
        
        # 6. 模拟 JPEG 压缩
        compressed_fake_image = simulate_jpeg_compression(fake_image, quality=85)
        
        fake_images.append(compressed_fake_image)
    
    print(f"  [4/6] 成功生成 {len(fake_images)} 张伪造图像")
    
    return fake_images

def save_fake_images(fake_images, output_path="data/fake/copy_move/"):
    for idx, fake_image in enumerate(fake_images):
        fake_image_path = f"{output_path}fake_image_{idx}.jpg"
        cv2.imwrite(fake_image_path, fake_image)
        print(f"Saved fake image at {fake_image_path}")
