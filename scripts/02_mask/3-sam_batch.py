# scripts/3-sam_batch.py
"""
SAM 批量生成 mask（批处理入口）
- 循环 / tqdm / 统计在本脚本
- 单张生成复用 pipeline/steps/step_mask_generate.py
- SAM 模型初始化只做一次
python scripts/02_mask/3-sam_batch.py
python scripts/02_mask/3-sam_batch.py --skip-existing
python scripts/02_mask/3-sam_batch.py --overwrite
python scripts/02_mask/3-sam_batch.py --skip-existing --max-len 50


"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/02_mask/
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)  # scripts/
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)  # paper-fig-forgery-dataset/

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.generators.sam_masker import get_default_sam_config, initialize_sam_model
from pipeline.steps.step_mask_generate import run_step_mask_generate


def list_images(input_dir: str, exts: List[str] = [".png"]) -> List[Path]:
    """列出目录下所有指定后缀的图片文件（默认 .png）"""
    files = []
    p = Path(input_dir)
    for ext in exts:
        files += list(p.glob(f"*{ext}"))
        files += list(p.glob(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def main():
    parser = argparse.ArgumentParser(description="使用 SAM 批量生成掩码（mask）")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已经生成过 mask 的图片")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已有 mask 输出")
    parser.add_argument("--max-len", type=int, default=0, help="最多处理多少张图片（0 表示不限制）")
    args = parser.parse_args()

    # 路径
    real_dir = os.path.join(PROJECT_ROOT, "data", "real")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "mask")
    os.makedirs(mask_dir, exist_ok=True)

    # 配置
    config = get_default_sam_config()
    model_type = config["model_type"]

    sam_checkpoints = {
        "vit_b": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_b_01ec64.pth"),
        "vit_h": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_h_4b8939.pth"),
        "vit_l": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_l_0b3195.pth"),
    }
    ckpt = sam_checkpoints[model_type]

    # 打印配置
    print("=" * 70)
    print("SAM 批量生成 Mask")
    print("=" * 70)
    print(f"输入目录（real）   : {real_dir}")
    print(f"输出目录（mask）   : {mask_dir}")
    print(f"模型类型           : {model_type}")
    print(f"跳过已存在 mask    : {args.skip_existing}")
    print(f"覆盖已有 mask      : {args.overwrite}")
    print(f"最大处理数量       : {args.max_len if args.max_len > 0 else '不限制'}")
    print("=" * 70)

    # 收集图片
    imgs = list_images(real_dir, exts=[".png"])
    if not imgs:
        print("错误：未找到任何图片文件，请检查 data/real 是否为空。")
        return

    print(f"扫描到图片总数：{len(imgs)} 张")

    # 过滤已处理
    if args.skip_existing and (not args.overwrite):
        new_imgs = []
        for p in imgs:
            out_path = os.path.join(mask_dir, p.stem + ".png")
            if not os.path.exists(out_path):
                new_imgs.append(p)
        skipped_count = len(imgs) - len(new_imgs)
        imgs = new_imgs
        print(f"跳过已生成 mask：{skipped_count} 张")
        print(f"待处理图片数量  ：{len(imgs)} 张")

    # 限制 max_len
    if args.max_len > 0:
        imgs = imgs[: args.max_len]
        print(f"限制本次最多处理：{len(imgs)} 张")

    if not imgs:
        print("提示：所有图片都已经存在 mask，无需处理。")
        return

    # 初始化 SAM（只做一次）
    print("正在初始化 SAM 模型（只加载一次）...")
    mask_generator, device = initialize_sam_model(
        checkpoint_path=ckpt,
        model_type=model_type,
        sam_params=config.get("sam_params", None),
    )
    config["device"] = device
    print(f"SAM 初始化完成，当前设备：{device}")

    print("=" * 70)
    print("开始批量生成 mask")
    print("=" * 70)

    # 统计
    stats: Dict[str, int] = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
    }

    start = time.time()

    for img_path in tqdm(imgs, desc="生成进度", unit="张"):
        out_mask = os.path.join(mask_dir, img_path.stem + ".png")

        ok, msg = run_step_mask_generate(
            real_path=str(img_path),
            mask_out_path=out_mask,
            mask_generator=mask_generator,
            device=device,
            config=config,
            overwrite=args.overwrite,
        )

        if ok:
            if msg == "exists_skip":
                stats["skipped"] += 1
            else:
                stats["success"] += 1
        else:
            stats["failed"] += 1
            tqdm.write(f"失败：{img_path.name} | 原因：{msg}")

    cost = time.time() - start
    total = stats["success"] + stats["failed"] + stats["skipped"]

    print("=" * 70)
    print("批量处理完成（统计汇总）")
    print("=" * 70)
    print(f"总计处理：{total} 张")
    print(f"成功    ：{stats['success']} 张")
    print(f"跳过    ：{stats['skipped']} 张")
    print(f"失败    ：{stats['failed']} 张")
    print(f"总耗时  ：{cost:.2f} 秒")
    if total > 0:
        print(f"平均每张：{cost / total:.3f} 秒/张")
    print("=" * 70)


if __name__ == "__main__":
    main()
