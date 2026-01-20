import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import sys
import time

# 设置标准输出编码为 UTF-8，解决中文显示问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 获取项目根目录（脚本在 scripts/02_mask/ 下，需要往上退两级）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # 往上退两级到项目根目录

# ==================== 配置参数 ====================
# 设置图片路径和权重路径（基于项目根目录的绝对路径）
IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "real", "其他_2010073273_0_4.png")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"  # 可以选择 "vit_b", "vit_l", "vit_h"

# 图像缩放配置
MAX_IMAGE_SIZE = 800  # 图像最大边长（像素），确保8GB显存够用
RESIZE_ENABLED = True  # 是否启用图像缩放

# 掩码颜色配置
INVERT_MASK = True  # True=掩码区域黑色(0)，背景白色(255)；False=掩码区域白色(255)，背景黑色(0)

# SAM 参数配置
SAM_PARAMS = {
    'points_per_side': 16,  # 采样点数量（16可降低显存占用）
    'pred_iou_thresh': 0.88,
    'stability_score_thresh': 0.95,
    'crop_n_layers': 0,  # 不裁剪，大幅降低显存
    'min_mask_region_area': 500  # 过滤小区域
}

# 选择第几大的区域（1=最大，2=第二大）
SELECT_NTH_LARGEST = 2
# ==================================================

print("="*70)
print("单张图像分割工具 - SAM Single Processing")
print("="*70)
print(f"\n配置信息：")
print(f"  - 项目根目录：{PROJECT_ROOT}")
print(f"  - 图片路径：{IMAGE_PATH}")
print(f"  - 模型路径：{CKPT_PATH}")
print(f"  - 模型类型：{MODEL_TYPE}")
print(f"  - 图像缩放：{'启用' if RESIZE_ENABLED else '禁用'} (最大边长: {MAX_IMAGE_SIZE}px)")
print(f"  - 掩码模式：{'反转(掩码黑/背景白)' if INVERT_MASK else '正常(掩码白/背景黑)'}")
print(f"  - 选择区域：第 {SELECT_NTH_LARGEST} 大")

# 检查图片是否存在
if not os.path.exists(IMAGE_PATH):
    print(f"\n❌ 错误：图片不存在 {IMAGE_PATH}")
    sys.exit(1)

# 读取图片（使用 cv2.imdecode 解决中文路径问题）
print(f"\n正在读取图片...")
start_time = time.time()

try:
    with open(IMAGE_PATH, 'rb') as f:
        image_data = np.frombuffer(f.read(), np.uint8)
        image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    if image_bgr is None:
        raise ValueError("图片解码失败")
    
    # 获取原始尺寸
    original_height, original_width = image_bgr.shape[:2]
    print(f"  ✅ 读取成功")
    print(f"  - 原始尺寸：{original_width} x {original_height}")
    
    # 如果启用缩放且图像过大，则缩小图像
    if RESIZE_ENABLED:
        max_dim = max(original_height, original_width)
        if max_dim > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max_dim
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            image_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"  - 缩放后尺寸：{new_width} x {new_height} (缩放比例: {scale:.2f})")
    
    # 转换颜色空间
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
except Exception as e:
    print(f"  ❌ 读取失败：{e}")
    sys.exit(1)

# 加载 SAM 模型
print(f"\n正在加载 SAM 模型...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"  - 使用设备：{device}")

if device == "cuda:0":
    print(f"  - GPU名称：{torch.cuda.get_device_name(0)}")
    print(f"  - GPU显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

try:
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CKPT_PATH)
    sam.to(device=device)
    print(f"  ✅ 模型加载成功")
except Exception as e:
    print(f"  ❌ 模型加载失败：{e}")
    sys.exit(1)

# 使用 SAM 自动生成候选区域
print(f"\n正在生成掩码...")
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    **SAM_PARAMS
)

try:
    masks = mask_generator.generate(image_rgb)
    print(f"  ✅ 生成了 {len(masks)} 个候选区域")
    
    if len(masks) == 0:
        print(f"  ⚠️  警告：未生成任何掩码，将使用全黑掩码")
        # 创建全黑掩码（如果启用了反转，则会变成全白）
        roi_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
        use_full_black = True
    else:
        use_full_black = False
    
    if not use_full_black:
        # 显示前5个最大区域的面积
        sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)
        print(f"\n  前 {min(5, len(masks))} 个最大区域的面积：")
        for i, mask in enumerate(sorted_masks[:5], 1):
            area_pixels = mask["area"]
            area_percent = (area_pixels / (image_rgb.shape[0] * image_rgb.shape[1])) * 100
            print(f"    {i}. 面积: {area_pixels} 像素 ({area_percent:.2f}%)")
        
        # 选择第N大的掩码
        if len(masks) < SELECT_NTH_LARGEST:
            print(f"\n  ⚠️  警告：只有 {len(masks)} 个区域，无法选择第 {SELECT_NTH_LARGEST} 大，使用最大的")
            roi = sorted_masks[0]
            selected_index = 1
        else:
            roi = sorted_masks[SELECT_NTH_LARGEST - 1]
            selected_index = SELECT_NTH_LARGEST
        
        roi_mask = (roi["segmentation"].astype(np.uint8) * 255)
        print(f"\n  ✅ 已选择第 {selected_index} 大的区域")
        
        # 如果图像被缩放了，将掩码恢复到原始尺寸
        if RESIZE_ENABLED and (roi_mask.shape[0] != original_height or roi_mask.shape[1] != original_width):
            roi_mask = cv2.resize(roi_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            print(f"  - 掩码已恢复到原始尺寸：{original_width} x {original_height}")
    else:
        # 使用全黑掩码，需要恢复到原始尺寸
        if roi_mask.shape[0] != original_height or roi_mask.shape[1] != original_width:
            roi_mask = cv2.resize(roi_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        print(f"\n  ✅ 使用全黑掩码（尺寸：{original_width} x {original_height}）")
    
    # 根据配置反转掩码（掩码区域变黑色，背景变白色）
    if INVERT_MASK:
        roi_mask = 255 - roi_mask  # 反转：0变255，255变0
        print(f"  - 掩码已反转（掩码区域=黑色，背景=白色）")
    
except Exception as e:
    print(f"  ❌ 生成掩码失败：{e}")
    sys.exit(1)

# 保存 ROI mask 图像（使用 cv2.imencode 解决中文路径问题）
output_filename = os.path.basename(IMAGE_PATH)
output_path = os.path.join(PROJECT_ROOT, "data", "mask", output_filename)

# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"\n正在保存掩码...")
try:
    # 使用 imencode 编码后写入文件，避免中文路径问题
    success, encoded_image = cv2.imencode('.png', roi_mask)
    if success:
        with open(output_path, 'wb') as f:
            f.write(encoded_image.tobytes())
        
        total_time = time.time() - start_time
        print(f"  ✅ 保存成功")
        print(f"  - 输出路径：{output_path}")
        print(f"  - 总耗时：{total_time:.2f} 秒")
    else:
        print(f"  ❌ 保存失败：编码错误")
        sys.exit(1)
        
except Exception as e:
    print(f"  ❌ 保存失败：{e}")
    sys.exit(1)

print("\n" + "="*70)
print("处理完成！")
print("="*70)
