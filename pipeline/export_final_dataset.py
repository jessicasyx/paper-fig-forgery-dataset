"""
生成最终交付数据集
从 meta.jsonl 中提取成功生成的 fake 图像信息，生成符合要求的 JSON 格式
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

# 项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# 类别映射（根据文件名前缀）
CATEGORY_MAPPING = {
    "其他": {"type": "Others", "sub_type": "other"},
    "染色": {"type": "Microscopy", "sub_type": "staining"},
    "示意": {"type": "Diagram", "sub_type": "schematic"},
    "流程": {"type": "Diagram", "sub_type": "flowchart"},
    "统计": {"type": "Chart", "sub_type": "statistics"},
    "封面": {"type": "Others", "sub_type": "cover"},
    "照片": {"type": "Photo", "sub_type": "photo"},
    "模型": {"type": "Model", "sub_type": "3d_model"},
    "结构": {"type": "Diagram", "sub_type": "structure"},
}


def parse_filename(filename: str) -> Dict:
    """
    从文件名解析信息
    格式：类别_ID_数字1_数字2.png
    例如：其他_2010000000_1_0.png
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    
    result = {
        "original_name": filename,
        "stem": stem,
        "category": "其他",  # 默认
        "type": "Others",
        "sub_type": "other",
    }
    
    if len(parts) >= 1:
        category = parts[0]
        result["category"] = category
        
        # 查找类别映射
        if category in CATEGORY_MAPPING:
            result["type"] = CATEGORY_MAPPING[category]["type"]
            result["sub_type"] = CATEGORY_MAPPING[category]["sub_type"]
    
    return result


def load_meta_jsonl(meta_path: str) -> Dict[str, List[Dict]]:
    """
    加载 meta.jsonl，按 sample_id 分组
    """
    records = defaultdict(list)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                sample_id = record.get("sample_id", "")
                records[sample_id].append(record)
            except json.JSONDecodeError as e:
                print(f"[WARN] 解析 JSON 失败: {e}")
                continue
    
    return records


def get_successful_samples(meta_records: Dict[str, List[Dict]], gen_type: str = "repaint") -> List[Dict]:
    """
    从 meta 记录中提取成功生成的样本
    """
    successful = []
    
    for sample_id, records in meta_records.items():
        # 查找该样本的最终状态
        repaint_done = False
        mask_path = None
        fake_path = None
        real_path = None
        
        for record in records:
            if record.get("type") != gen_type:
                continue
            
            stage = record.get("stage")
            status = record.get("status")
            paths = record.get("paths", {})
            
            # 收集路径信息
            if "mask" in paths:
                mask_path = paths["mask"]
            if "real" in paths:
                real_path = paths["real"]
            
            # 检查 repaint 是否成功
            if stage == "repaint" and status == "done":
                repaint_done = True
                if "fake" in paths:
                    fake_path = paths["fake"]
        
        # 如果 repaint 成功且文件存在
        if repaint_done and fake_path and os.path.exists(fake_path):
            if mask_path and os.path.exists(mask_path):
                successful.append({
                    "sample_id": sample_id,
                    "real_path": real_path,
                    "mask_path": mask_path,
                    "fake_path": fake_path,
                })
    
    return successful


def generate_final_dataset(
    meta_path: str,
    output_dir: str,
    gen_type: str = "repaint",
    copy_images: bool = True,
    with_text_default: bool = False,
    forgery_text_default: str = "No"
):
    """
    生成最终交付数据集
    
    :param meta_path: meta.jsonl 路径
    :param output_dir: 输出目录
    :param gen_type: 生成类型（repaint 或 copymove）
    :param copy_images: 是否复制图像到输出目录
    :param with_text_default: 默认的 with_text 值
    :param forgery_text_default: 默认的 forgery_text 值
    """
    print("=" * 70)
    print("生成最终交付数据集")
    print("=" * 70)
    print(f"Meta 文件: {meta_path}")
    print(f"输出目录: {output_dir}")
    print(f"生成类型: {gen_type}")
    print(f"复制图像: {copy_images}")
    print("=" * 70)
    
    # 1. 加载 meta 记录
    print("\n[1/4] 加载 meta 记录...")
    meta_records = load_meta_jsonl(meta_path)
    print(f"  加载了 {len(meta_records)} 个样本的记录")
    
    # 2. 提取成功的样本
    print("\n[2/4] 提取成功生成的样本...")
    successful_samples = get_successful_samples(meta_records, gen_type)
    print(f"  找到 {len(successful_samples)} 个成功生成的样本")
    
    if len(successful_samples) == 0:
        print("\n[ERROR] 没有找到成功生成的样本，退出。")
        return
    
    # 3. 创建输出目录结构
    print("\n[3/4] 创建输出目录...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if copy_images:
        images_dir = output_path / "images"
        fake_images_dir = images_dir / gen_type / "fake"
        mask_images_dir = images_dir / gen_type / "mask"
        
        fake_images_dir.mkdir(parents=True, exist_ok=True)
        mask_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"  创建图像目录: {images_dir}")
    
    # 4. 生成 JSON 记录
    print("\n[4/4] 生成 JSON 记录...")
    dataset_records = []
    
    for idx, sample in enumerate(successful_samples, 1):
        fake_path = sample["fake_path"]
        mask_path = sample["mask_path"]
        
        # 获取文件名
        fake_filename = os.path.basename(fake_path)
        mask_filename = os.path.basename(mask_path)
        
        # 解析文件名获取类别信息
        file_info = parse_filename(fake_filename)
        
        # 构建相对路径
        if copy_images:
            # 复制图像
            fake_dest = fake_images_dir / fake_filename
            mask_dest = mask_images_dir / mask_filename
            
            try:
                shutil.copy2(fake_path, fake_dest)
                shutil.copy2(mask_path, mask_dest)
            except Exception as e:
                print(f"  [WARN] 复制文件失败: {fake_filename}, 错误: {e}")
                continue
            
            # 使用相对路径
            inpaint_path = f"images/{gen_type}/fake/{fake_filename}"
            gt_path = f"images/{gen_type}/mask/{mask_filename}"
        else:
            # 使用绝对路径
            inpaint_path = fake_path
            gt_path = mask_path
        
        # 构建记录
        record = {
            "image_name": fake_filename,
            "type": file_info["type"],
            "sub_type": file_info["sub_type"],
            "with_text": with_text_default,
            "generative_model": gen_type,
            "inpaint_image_path": inpaint_path,
            "gt_image_path": gt_path,
            "forgery_text": forgery_text_default
        }
        
        dataset_records.append(record)
        
        if idx % 100 == 0:
            print(f"  处理进度: {idx}/{len(successful_samples)}")
    
    print(f"  生成了 {len(dataset_records)} 条记录")
    
    # 5. 保存 JSON 文件
    json_path = output_path / f"{gen_type}_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 数据集生成完成！")
    print(f"  JSON 文件: {json_path}")
    print(f"  总记录数: {len(dataset_records)}")
    if copy_images:
        print(f"  图像目录: {images_dir}")
    print("=" * 70)
    
    # 6. 打印统计信息
    print("\n【统计信息】")
    type_count = defaultdict(int)
    subtype_count = defaultdict(int)
    
    for record in dataset_records:
        type_count[record["type"]] += 1
        subtype_count[record["sub_type"]] += 1
    
    print("\n类别分布:")
    for t, count in sorted(type_count.items()):
        print(f"  {t}: {count}")
    
    print("\n子类别分布:")
    for st, count in sorted(subtype_count.items()):
        print(f"  {st}: {count}")
    
    return json_path, dataset_records


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成最终交付数据集")
    parser.add_argument(
        "--meta",
        type=str,
        default=os.path.join(PROJECT_ROOT, "meta", "meta.jsonl"),
        help="meta.jsonl 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "fake"),
        help="输出目录"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="repaint",
        choices=["repaint", "copymove"],
        help="生成类型"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="不复制图像，只生成 JSON（使用绝对路径）"
    )
    parser.add_argument(
        "--with-text",
        action="store_true",
        help="默认标记为包含文字（with_text=true）"
    )
    
    args = parser.parse_args()
    
    generate_final_dataset(
        meta_path=args.meta,
        output_dir=args.output,
        gen_type=args.type,
        copy_images=not args.no_copy,
        with_text_default=args.with_text,
        forgery_text_default="No"
    )


if __name__ == "__main__":
    main()

