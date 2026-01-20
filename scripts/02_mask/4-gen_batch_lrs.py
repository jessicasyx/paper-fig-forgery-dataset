# scripts/4-gen_batch_lrs.py
"""
批量生成挖洞图像（corrupted / LRS）工具脚本
✅ 这是旧脚本的“批处理入口”，保留循环/进度条/统计
✅ 核心单张处理逻辑复用 pipeline/steps/step_corrupted.py

用法示例：
python scripts/02_mask/4-gen_batch_lrs.py
python scripts/02_mask/4-gen_batch_lrs.py --skip-existing
python scripts/02_mask/4-gen_batch_lrs.py --max-len 50
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple

from tqdm import tqdm

# 让 scripts 能 import 到 pipeline（推荐：项目根目录执行）
# 如果你是直接双击跑，可能需要手动加 root 到 sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/02_mask/
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)  # scripts/
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)  # paper-fig-forgery-dataset/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.steps.step_corrupted import run_step_corrupted


def create_statistics_dict() -> Dict:
    return {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "mask_not_found": 0,
        "read_error": 0,
        "size_mismatch": 0,
        "write_failed": 0,
        "exception": 0,
        "failed_files": [],
        "total_time": 0.0,
    }


def update_statistics(stats: Dict, ok: bool, err: str, image_name: str) -> None:
    if ok:
        if err == "exists_skip":
            stats["skipped"] += 1
        else:
            stats["success"] += 1
        return

    stats["failed"] += 1
    stats["failed_files"].append((image_name, err))

    if "mask_not_found" in err:
        stats["mask_not_found"] += 1
    elif "read_" in err:
        stats["read_error"] += 1
    elif "size_mismatch" in err:
        stats["size_mismatch"] += 1
    elif "write_failed" in err:
        stats["write_failed"] += 1
    elif "exception:" in err:
        stats["exception"] += 1


def print_statistics(stats: Dict, output_dir: str) -> None:
    print("\n" + "=" * 70)
    print("处理完成 - 统计报告")
    print("=" * 70)

    total = stats["success"] + stats["failed"] + stats["skipped"]
    print(f"\n处理结果：")
    print(f"  ✅ 成功：{stats['success']} 张")
    print(f"  ⏭️ 跳过：{stats['skipped']} 张")
    print(f"  ❌ 失败：{stats['failed']} 张")

    if stats["failed"] > 0:
        print(f"\n失败分类：")
        print(f"  - mask缺失：{stats['mask_not_found']} 张")
        print(f"  - 读取错误：{stats['read_error']} 张")
        print(f"  - 尺寸不匹配：{stats['size_mismatch']} 张")
        print(f"  - 写入失败：{stats['write_failed']} 张")
        print(f"  - 异常：{stats['exception']} 张")

    if total > 0:
        success_rate = stats["success"] / total * 100.0
        print(f"\n📊 成功率（成功/总处理含跳过）：{success_rate:.1f}%")

    print(f"\n时间统计：")
    print(f"  - 总耗时：{stats['total_time']:.2f} 秒")
    if total > 0:
        print(f"  - 平均每张（含跳过）：{stats['total_time'] / total:.3f} 秒")

    if stats["failed"] > 0:
        print(f"\n失败样本（前10个）：")
        for filename, reason in stats["failed_files"][:10]:
            print(f"  - {filename}: {reason}")

    print(f"\n输出目录：{output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true", help="跳过已生成的 corrupted")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖输出")
    parser.add_argument("--max-len", type=int, default=0, help="最多处理多少张，0表示不限")
    args = parser.parse_args()

    real_dir = os.path.join(PROJECT_ROOT, "data", "real")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "mask")
    out_dir = os.path.join(PROJECT_ROOT, "data", "corrupted")

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("批量生成挖洞图像（corrupted / LRS）")
    print("=" * 70)
    print(f"real_dir     = {real_dir}")
    print(f"mask_dir     = {mask_dir}")
    print(f"output_dir   = {out_dir}")
    print(f"skip_existing= {args.skip_existing}")
    print(f"overwrite    = {args.overwrite}")
    print(f"max_len      = {args.max_len}")

    all_images = [f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f))]
    all_images.sort()

    if args.skip_existing and (not args.overwrite):
        image_list = []
        for name in all_images:
            out_path = os.path.join(out_dir, name)
            if not os.path.exists(out_path):
                image_list.append(name)
    else:
        image_list = all_images

    if args.max_len > 0:
        image_list = image_list[: args.max_len]

    if len(image_list) == 0:
        print("\n✅ 没有需要处理的图片（可能都已生成或目录为空）")
        return

    stats = create_statistics_dict()
    start = time.time()

    fill_color = [255, 255, 255]

    for image_name in tqdm(image_list, desc="处理进度", unit="张"):
        real_path = os.path.join(real_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)
        out_path = os.path.join(out_dir, image_name)

        ok, err = run_step_corrupted(
            real_path=real_path,
            mask_keep_path=mask_path,
            corrupted_path=out_path,
            fill_color=fill_color,
            overwrite=args.overwrite,
        )

        update_statistics(stats, ok, err, image_name)

        if not ok:
            tqdm.write(f"❌ {image_name}: {err}")

    stats["total_time"] = time.time() - start
    print_statistics(stats, out_dir)


if __name__ == "__main__":
    main()
