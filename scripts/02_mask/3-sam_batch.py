import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import sys
from pathlib import Path
from tqdm import tqdm
import time

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 配置参数
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "real")  # 输入图片目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "mask")  # 输出掩码目录
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")  # 日志目录
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"  # 可选: "vit_b", "vit_l", "vit_h"

# 图像缩放配置
MAX_IMAGE_SIZE = 800  # 图像最大边长（像素），更保守的设置确保8GB显存够用
RESIZE_ENABLED = True  # 是否启用图像缩放

# 掩码颜色配置
INVERT_MASK = True  # True=掩码区域黑色(0)，背景白色(255)；False=掩码区域白色(255)，背景黑色(0)

# SAM 参数配置
SAM_PARAMS = {
    'points_per_side': 16,  # 从32降到16（减少采样点，降低显存占用）
    'pred_iou_thresh': 0.88,
    'stability_score_thresh': 0.95,
    'crop_n_layers': 0,  # 从1降到0（不裁剪，大幅降低显存）
    'min_mask_region_area': 500  # 从200提高到500（过滤小区域）
}

# 是否跳过已处理的图片
SKIP_EXISTING = True

print("="*70)
print("批量图像分割工具 - SAM Batch Processing")
print("="*70)
print(f"\n配置信息：")
print(f"  - 输入目录：{INPUT_DIR}")
print(f"  - 输出目录：{OUTPUT_DIR}")
print(f"  - 模型类型：{MODEL_TYPE}")
print(f"  - 图像缩放：{'启用' if RESIZE_ENABLED else '禁用'} (最大边长: {MAX_IMAGE_SIZE}px)")
print(f"  - 掩码模式：{'反转(掩码黑/背景白)' if INVERT_MASK else '正常(掩码白/背景黑)'}")
print(f"  - 跳过已处理：{SKIP_EXISTING}")

# 创建输出目录和日志目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 获取所有图片文件
image_extensions = ['.png']
image_files = []

for ext in image_extensions:
    image_files.extend(Path(INPUT_DIR).glob(f"*{ext}"))
    image_files.extend(Path(INPUT_DIR).glob(f"*{ext.upper()}"))

# 保持real文件夹里的顺序
# image_files = sorted(list(set(image_files)))  # 去重并排序


print(f"\n找到 {len(image_files)} 张图片")

if len(image_files) == 0:
    print(f"❌ 错误：在 {INPUT_DIR} 中没有找到图片文件")
    sys.exit(1)

# 检查已处理的图片
if SKIP_EXISTING:
    unprocessed_files = []
    for img_file in image_files:
        output_filename = img_file.stem + ".png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        if not os.path.exists(output_path):
            unprocessed_files.append(img_file)
    
    skipped_count = len(image_files) - len(unprocessed_files)
    if skipped_count > 0:
        print(f"  - 跳过已处理：{skipped_count} 张")
        print(f"  - 待处理：{len(unprocessed_files)} 张")
    
    image_files = unprocessed_files

if len(image_files) == 0:
    print("\n✅ 所有图片已处理完成！")
    sys.exit(0)

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

# 创建掩码生成器
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    **SAM_PARAMS
)

# 统计信息
stats = {
    'success': 0,
    'failed': 0,
    'total_time': 0,
    'failed_files': []
}

# 批量处理函数
def process_image(image_path):
    """处理单张图片"""
    try:
        # 读取图片（支持中文路径）
        with open(image_path, 'rb') as f:
            image_data = np.frombuffer(f.read(), np.uint8)
            image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            return False, "无法读取图片"
        
        # 获取原始尺寸
        original_height, original_width = image_bgr.shape[:2]
        
        # 如果启用缩放且图像过大，则缩小图像
        if RESIZE_ENABLED:
            max_dim = max(original_height, original_width)
            if max_dim > MAX_IMAGE_SIZE:
                scale = MAX_IMAGE_SIZE / max_dim
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                image_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 生成掩码
        masks = mask_generator.generate(image_rgb)
        
        if len(masks) == 0:
            return False, "未生成任何掩码"
        
        # 选择面积第二大的掩码
        if len(masks) < 2:
            # 如果只有一个mask，退而求其次选择面积最大的
            roi = max(masks, key=lambda m: m["area"])
        else:
            sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)
            roi = sorted_masks[1]
        roi_mask = (roi["segmentation"].astype(np.uint8) * 255)
        
        # 如果图像被缩放了，将掩码恢复到原始尺寸
        if RESIZE_ENABLED and (roi_mask.shape[0] != original_height or roi_mask.shape[1] != original_width):
            roi_mask = cv2.resize(roi_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        
        # 根据配置反转掩码（掩码区域变黑色，背景变白色）
        if INVERT_MASK:
            roi_mask = 255 - roi_mask  # 反转：0变255，255变0
        
        # 保存掩码（支持中文路径）
        output_filename = image_path.stem + ".png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        success, encoded_image = cv2.imencode('.png', roi_mask)
        if success:
            with open(output_path, 'wb') as f:
                f.write(encoded_image.tobytes())
            return True, f"成功 (生成{len(masks)}个区域)"
        else:
            return False, "保存失败"
            
    except Exception as e:
        return False, str(e)

# 开始批量处理
print("\n" + "="*70)
print("开始批量处理")
print("="*70 + "\n")

start_time = time.time()

try:
    # 使用 tqdm 显示进度条
    for img_file in tqdm(image_files, desc="处理进度", unit="张"):
        img_start_time = time.time()
        
        success, message = process_image(img_file)
        
        img_time = time.time() - img_start_time
        stats['total_time'] += img_time
        
        if success:
            stats['success'] += 1
            # tqdm.write(f"✅ {img_file.name}: {message} ({img_time:.2f}秒)")
        else:
            stats['failed'] += 1
            stats['failed_files'].append((img_file.name, message))
            tqdm.write(f"❌ {img_file.name}: {message}")
        
        # 清理GPU显存（如果使用GPU）
        if device == "cuda:0":
            try:
                torch.cuda.empty_cache()  # 清空缓存
                torch.cuda.synchronize()  # 同步GPU操作
            except RuntimeError as e:
                # 如果清理显存失败，记录错误但继续处理
                tqdm.write(f"⚠️  显存清理失败: {str(e)}")
                
except KeyboardInterrupt:
    print("\n\n⚠️  用户中断处理")
except Exception as e:
    print(f"\n\n❌ 处理过程中发生错误: {e}")
finally:
    total_time = time.time() - start_time

# 输出统计结果
print("\n" + "="*70)
print("处理完成 - 统计报告")
print("="*70)

print(f"\n处理结果：")
print(f"  ✅ 成功：{stats['success']} 张")
print(f"  ❌ 失败：{stats['failed']} 张")
print(f"  📊 成功率：{stats['success']/(stats['success']+stats['failed'])*100:.1f}%")

print(f"\n时间统计：")
print(f"  - 总耗时：{total_time:.2f} 秒")
print(f"  - 平均每张：{stats['total_time']/(stats['success']+stats['failed']):.2f} 秒")

if stats['failed'] > 0:
    print(f"\n失败的文件：")
    for filename, reason in stats['failed_files']:
        print(f"  - {filename}: {reason}")

print(f"\n输出目录：{OUTPUT_DIR}")
print("="*70)

# 生成处理日志
timestamp = time.strftime('%Y%m%d_%H%M%S')
log_filename = f"batch_processing_mask{timestamp}.log"
log_path = os.path.join(LOG_DIR, log_filename)

with open(log_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("批量图像分割处理日志\n")
    f.write("="*70 + "\n\n")
    f.write(f"处理时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"输入目录：{INPUT_DIR}\n")
    f.write(f"输出目录：{OUTPUT_DIR}\n")
    f.write(f"模型类型：{MODEL_TYPE}\n")
    f.write(f"掩码模式：{'反转(掩码黑/背景白)' if INVERT_MASK else '正常(掩码白/背景黑)'}\n")
    f.write(f"使用设备：{device}\n\n")
    f.write(f"处理结果：\n")
    f.write(f"  - 成功：{stats['success']} 张\n")
    f.write(f"  - 失败：{stats['failed']} 张\n")
    f.write(f"  - 成功率：{stats['success']/(stats['success']+stats['failed'])*100:.1f}%\n\n")
    f.write(f"时间统计：\n")
    f.write(f"  - 总耗时：{total_time:.2f} 秒\n")
    f.write(f"  - 平均每张：{stats['total_time']/(stats['success']+stats['failed']):.2f} 秒\n\n")
    
    if stats['failed'] > 0:
        f.write("失败的文件：\n")
        for filename, reason in stats['failed_files']:
            f.write(f"  - {filename}: {reason}\n")

print(f"\n日志已保存到：{log_path}")

