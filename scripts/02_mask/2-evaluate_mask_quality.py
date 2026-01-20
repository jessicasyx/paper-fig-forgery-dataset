import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import sys

# 设置标准输出编码为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 设置路径
IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "real", "其他_60234579_5_0.png")
CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"

print("="*70)
print("局部伪造掩码质量评估工具")
print("="*70)

# 读取图片
with open(IMAGE_PATH, 'rb') as f:
    image_data = np.frombuffer(f.read(), np.uint8)
    image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

if image_bgr is None:
    raise FileNotFoundError(f"无法读取图片：{IMAGE_PATH}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_height, image_width = image_bgr.shape[:2]
total_pixels = image_height * image_width

print(f"\n图像信息：")
print(f"  - 尺寸：{image_width} x {image_height}")
print(f"  - 总像素：{total_pixels:,}")

# 加载 SAM 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  - 使用设备：{device}")

sam = sam_model_registry[MODEL_TYPE](checkpoint=CKPT_PATH)
sam.to(device=device)

# 生成候选区域
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    min_mask_region_area=200
)

masks = mask_generator.generate(image_rgb)
print(f"\n生成了 {len(masks)} 个候选区域")

# ========== 掩码质量评估函数 ==========

def evaluate_mask_quality(mask, image_shape):
    """
    评估掩码是否适合作为局部伪造的目标区域
    
    返回：quality_score (0-100), reasons (评估理由)
    """
    segmentation = mask["segmentation"]
    area = mask["area"]
    bbox = mask["bbox"]  # [x, y, width, height]
    
    height, width = image_shape[:2]
    total_pixels = height * width
    
    # 计算各项指标
    area_ratio = (area / total_pixels) * 100  # 面积占比（百分比）
    
    # 边界框信息
    x, y, w, h = bbox
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999  # 宽高比
    
    # 距离边缘的最小距离
    min_edge_distance = min(x, y, width - (x + w), height - (y + h))
    edge_distance_ratio = (min_edge_distance / min(width, height)) * 100
    
    # 紧凑度（面积 / 边界框面积）
    bbox_area = w * h
    compactness = (area / bbox_area) * 100 if bbox_area > 0 else 0
    
    # 评分系统
    score = 0
    reasons = []
    
    # 1. 面积占比评分（40分）
    if 10 <= area_ratio <= 30:
        score += 40
        reasons.append(f"✅ 面积占比理想 ({area_ratio:.1f}%)")
    elif 5 <= area_ratio < 10 or 30 < area_ratio <= 40:
        score += 25
        reasons.append(f"⚠️  面积占比可接受 ({area_ratio:.1f}%)")
    elif area_ratio < 5:
        score += 10
        reasons.append(f"❌ 面积过小 ({area_ratio:.1f}%)")
    else:
        score += 5
        reasons.append(f"❌ 面积过大 ({area_ratio:.1f}%)")
    
    # 2. 形状评分（30分）
    if aspect_ratio <= 3:
        score += 30
        reasons.append(f"✅ 形状合理 (宽高比: {aspect_ratio:.1f})")
    elif aspect_ratio <= 5:
        score += 20
        reasons.append(f"⚠️  形状略长 (宽高比: {aspect_ratio:.1f})")
    else:
        score += 5
        reasons.append(f"❌ 形状过于狭长 (宽高比: {aspect_ratio:.1f})")
    
    # 3. 位置评分（20分）
    if edge_distance_ratio >= 5:
        score += 20
        reasons.append(f"✅ 位置合适，远离边缘")
    elif edge_distance_ratio >= 2:
        score += 10
        reasons.append(f"⚠️  位置可接受")
    else:
        score += 0
        reasons.append(f"❌ 过于靠近边缘")
    
    # 4. 紧凑度评分（10分）
    if compactness >= 60:
        score += 10
        reasons.append(f"✅ 区域紧凑 (紧凑度: {compactness:.1f}%)")
    elif compactness >= 40:
        score += 5
        reasons.append(f"⚠️  区域较分散 (紧凑度: {compactness:.1f}%)")
    else:
        score += 0
        reasons.append(f"❌ 区域过于分散 (紧凑度: {compactness:.1f}%)")
    
    return score, reasons, {
        'area_ratio': area_ratio,
        'aspect_ratio': aspect_ratio,
        'edge_distance_ratio': edge_distance_ratio,
        'compactness': compactness
    }

# ========== 评估所有掩码 ==========

print("\n" + "="*70)
print("掩码质量评估结果")
print("="*70)

# 按面积排序
sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)

# 评估每个掩码
evaluated_masks = []
for idx, mask in enumerate(sorted_masks):
    score, reasons, metrics = evaluate_mask_quality(mask, image_bgr.shape)
    evaluated_masks.append({
        'index': idx,
        'mask': mask,
        'score': score,
        'reasons': reasons,
        'metrics': metrics
    })

# 按质量分数排序
evaluated_masks.sort(key=lambda x: x['score'], reverse=True)

# 显示前5个最适合的掩码
print("\n【前5个最适合做局部伪造的掩码】\n")
for i, item in enumerate(evaluated_masks[:5]):
    print(f"排名 {i+1}：质量分数 {item['score']}/100")
    print(f"  面积：{item['mask']['area']:,} 像素 ({item['metrics']['area_ratio']:.2f}%)")
    print(f"  边界框：{item['mask']['bbox']}")
    for reason in item['reasons']:
        print(f"  {reason}")
    print()

# 保存最佳掩码
print("="*70)
print("保存最佳掩码")
print("="*70)

best_masks_dir = os.path.join(PROJECT_ROOT, "data", "mask", "best_for_forgery")
os.makedirs(best_masks_dir, exist_ok=True)

# 保存前5个最佳掩码
for i, item in enumerate(evaluated_masks[:5]):
    mask_image = (item['mask']["segmentation"].astype(np.uint8) * 255)
    
    filename = f"best_{i+1}_score_{item['score']}_area_{item['metrics']['area_ratio']:.1f}pct.png"
    output_path = os.path.join(best_masks_dir, filename)
    
    success, encoded_image = cv2.imencode('.png', mask_image)
    if success:
        with open(output_path, 'wb') as f:
            f.write(encoded_image.tobytes())
        print(f"✅ 已保存：{filename}")

print(f"\n所有最佳掩码已保存到：{best_masks_dir}")

# ========== 统计分析 ==========
print("\n" + "="*70)
print("统计分析")
print("="*70)

high_quality = [m for m in evaluated_masks if m['score'] >= 70]
medium_quality = [m for m in evaluated_masks if 50 <= m['score'] < 70]
low_quality = [m for m in evaluated_masks if m['score'] < 50]

print(f"\n质量分布：")
print(f"  - 高质量（≥70分）：{len(high_quality)} 个")
print(f"  - 中等质量（50-69分）：{len(medium_quality)} 个")
print(f"  - 低质量（<50分）：{len(low_quality)} 个")

print(f"\n推荐使用：")
if len(high_quality) > 0:
    print(f"  ✅ 优先使用前 {min(5, len(high_quality))} 个高质量掩码")
elif len(medium_quality) > 0:
    print(f"  ⚠️  可以使用中等质量掩码，但需要人工筛选")
else:
    print(f"  ❌ 建议调整SAM参数或更换图像")

print("\n" + "="*70)

