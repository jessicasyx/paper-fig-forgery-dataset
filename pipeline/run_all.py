import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from tqdm import tqdm
"""
cd E:\GraduationProject\paper-fig-forgery-dataset
python pipeline\run_all.py --type repaint --max-len 1 --batch-size 1 --skip-existing

"""
# ---------------------------------------------------------
# 项目路径 & import 修复（保证从任何位置运行都能导入）
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # pipeline/
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)               # 项目根目录
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# 模块
# ---------------------------------------------------------
from pipeline.steps.step_corrupted import run_step_corrupted
from pipeline.steps.step_mask_generate import run_step_mask_generate
from pipeline.generators.sam_masker import get_default_sam_config, initialize_sam_model
from pipeline.generators.repaint_wrapper import run_repaint_batch
from pipeline.utils.ids import make_sample_id
from pipeline.utils.meta_writer import write_meta, write_failed
from pipeline.utils.qa import eval_mask_qa,eval_fake_qa

# ---------------------------------------------------------
# 小工具：列出 real_dir 里的图片
# ---------------------------------------------------------
def list_images(real_dir: str, exts: List[str] = [".png"]) -> List[Path]:
    p = Path(real_dir)
    files: List[Path] = []
    for ext in exts:
        files += list(p.glob(f"*{ext}"))
        files += list(p.glob(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--type",
        type=str,
        default="repaint",
        choices=["repaint"],
        help="目前先支持 repaint 流水线（mask -> corrupted -> repaint）",
    )

    # 断点续跑策略
    parser.add_argument("--skip-existing", action="store_true", help="存在输出就跳过（推荐开启）")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已有输出（谨慎使用）")

    # 调试参数
    parser.add_argument("--max-len", type=int, default=0, help="最多处理多少张，0表示不限制")

    # RePaint 参数
    parser.add_argument("--timestep-respacing", type=str, default="250", help="默认先用稳定的250")
    parser.add_argument("--batch-size", type=int, default=2, help="RePaint batch_size，受显存限制")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 固定目录（项目标准结构）
    # ---------------------------------------------------------
    real_dir = os.path.join(PROJECT_ROOT, "data", "real")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "mask")
    corrupted_dir = os.path.join(PROJECT_ROOT, "data", "corrupted")
    fake_dir = os.path.join(PROJECT_ROOT, "data", "fake", "repaint")

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(corrupted_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    repaint_root = os.path.join(PROJECT_ROOT, "external", "RePaint-main")
    base_conf = os.path.join(repaint_root, "confs", "test_inet256_thick.yml")

    print("=" * 70)
    print("run_all.py - 最小闭环流水线（mask -> corrupted -> repaint）")
    print("=" * 70)
    print(f"type                : {args.type}")
    print(f"PROJECT_ROOT         : {PROJECT_ROOT}")
    print(f"real_dir             : {real_dir}")
    print(f"mask_dir             : {mask_dir}")
    print(f"corrupted_dir         : {corrupted_dir}")
    print(f"fake_dir             : {fake_dir}")
    print(f"skip_existing         : {args.skip_existing}")
    print(f"overwrite             : {args.overwrite}")
    print(f"max_len               : {args.max_len}")
    print(f"timestep_respacing    : {args.timestep_respacing}")
    print(f"batch_size            : {args.batch_size}")

    # ---------------------------------------------------------
    # 收集图片
    # ---------------------------------------------------------
    imgs = list_images(real_dir, exts=[".png"])
    if not imgs:
        print("real_dir 下没有 png 图片，退出。")
        return

    if args.max_len > 0:
        imgs = imgs[: args.max_len]

    print(f"\n待处理图片数量：{len(imgs)}")

    # ---------------------------------------------------------
    # Step 0：判断 mask 是否需要补（缺了才初始化SAM）
    # ---------------------------------------------------------
    missing_masks: List[Path] = []
    for p in imgs:
        m = os.path.join(mask_dir, p.stem + ".png")
        if not os.path.exists(m):
            missing_masks.append(p)

    need_make_mask = len(missing_masks) > 0 and not args.overwrite
    if args.overwrite:
        # overwrite 时认为都要重做
        need_make_mask = True
        missing_masks = imgs

    print("\n步骤0：mask 检查")
    print(f"mask 缺失数量：{len(missing_masks)}")
    print(f"是否需要生成mask：{need_make_mask}")

    # ---------------------------------------------------------
    # Step 1：生成 mask（SAM） + 记录跳过
    # ---------------------------------------------------------
    print("\n步骤1：mask 处理（SAM，可选）")

    # ✅ 不管要不要生成，都先取到 model_type（用于meta记录）
    config = get_default_sam_config()
    model_type = config["model_type"]

    # 合格样本列表（通过 QA 的样本）
    valid_imgs = []

    # ✅ 1) 先把“已存在的mask跳过”全部写入meta（否则永远漏账）
    if not args.overwrite:
        for img_path in tqdm(imgs, desc="记录mask跳过", unit="张"):
            out_mask = os.path.join(mask_dir, img_path.stem + ".png")
            if os.path.exists(out_mask):
                sample_id = make_sample_id(img_path, real_dir)

                # 先做 Mask QA
                mask_qa = eval_mask_qa(out_mask)

                # 记录 mask 跳过的同时也记录 QA
                write_meta(
                    sample_id=sample_id,
                    stage="mask",
                    status="skip",
                    type="repaint",
                    paths={"real": str(img_path), "mask": out_mask},
                    params={"model_type": model_type},
                    qa={"mask_qa": mask_qa}
                )
                # ✅ 如果 Mask QA 通过，也要加入 valid_imgs
                if mask_qa["pass"]:
                    valid_imgs.append(img_path)

    # ✅ 2) 再决定要不要生成缺失的mask
    if need_make_mask:
        print("\n开始生成缺失 mask（SAM）")

        sam_checkpoints = {
            "vit_b": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_b_01ec64.pth"),
            "vit_h": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_h_4b8939.pth"),
            "vit_l": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_l_0b3195.pth"),
        }

        ckpt = sam_checkpoints.get(model_type)
        if ckpt is None or not os.path.exists(ckpt):
            print(f"找不到 SAM 权重：{ckpt}")
            return

        # 初始化 SAM（只做一次）
        mask_generator, device = initialize_sam_model(
            checkpoint_path=ckpt,
            model_type=model_type,
            sam_params=config.get("sam_params", None),
        )

        ok_cnt, fail_cnt = 0, 0

        for img_path in tqdm(missing_masks, desc="生成mask", unit="张"):
            out_mask = os.path.join(mask_dir, img_path.stem + ".png")
            sample_id = make_sample_id(img_path, real_dir)

            ok, msg = run_step_mask_generate(
                real_path=str(img_path),
                mask_out_path=out_mask,
                mask_generator=mask_generator,
                device=device,
                config=config,
                overwrite=args.overwrite,
            )

            if ok:
                ok_cnt += 1

                # 在 mask 生成后立即进行 QA
                mask_qa = eval_mask_qa(out_mask)

                write_meta(
                    sample_id=sample_id,
                    stage="mask",
                    status="success",
                    type="repaint",
                    paths={"real": str(img_path), "mask": out_mask},
                    params={"model_type": model_type},
                    qa={"mask_qa": mask_qa}  # 将 mask_qa 写入 meta
                )

                # 如果 Mask QA 通过，加入合格名单
                if mask_qa["pass"]:
                    valid_imgs.append(img_path)
            else:
                fail_cnt += 1
                tqdm.write(f"生成mask失败：{img_path.name} | {msg}")

                write_failed(
                    sample_id=sample_id,
                    stage="mask",
                    reason=str(msg),
                    type="repaint",
                    paths={"real": str(img_path), "mask_out": out_mask},
                    params={"model_type": model_type}
                )

        print(f"mask 生成完成：新生成 {ok_cnt}，失败 {fail_cnt}")
    # ---------------------------------------------------------
    # Step 2：生成 corrupted（挖洞图）
    # ---------------------------------------------------------
    print("\n步骤2：生成 corrupted（挖洞图）")
    print(f"合格图片数量（通过 Mask QA）：{len(valid_imgs)}")
    
    ok_cnt, skip_cnt, fail_cnt = 0, 0, 0

    # ✅ 给 Step3 repaint 批量用
    repaint_todo = []

    # ✅ 只处理通过 Mask QA 的合格图片
    for img_path in tqdm(valid_imgs, desc="生成corrupted", unit="张"):
        name = img_path.name
        real_path = os.path.join(real_dir, name)
        mask_path = os.path.join(mask_dir, img_path.stem + ".png")
        out_path = os.path.join(corrupted_dir, name)
        # ✅ 每张图一个 sample_id
        sample_id = make_sample_id(img_path, real_dir)

        ok, msg = run_step_corrupted(
        real_path=real_path,
        mask_keep_path=mask_path,
        corrupted_path=out_path,
        overwrite=args.overwrite,
    )


        if ok:
            if msg == "exists_skip":
                skip_cnt += 1

                # ✅ 记录：corrupted跳过
                write_meta(
                    sample_id=sample_id,
                    stage="corrupted",
                    status="skip",
                    type="repaint",
                    paths={"real": real_path, "mask": mask_path, "corrupted": out_path},
                    params={}
                )
            else:
                ok_cnt += 1

                # ✅ 记录：corrupted成功
                write_meta(
                    sample_id=sample_id,
                    stage="corrupted",
                    status="success",
                    type="repaint",
                    paths={"real": real_path, "mask": mask_path, "corrupted": out_path},
                    params={}
                )

            # ✅ 只要 corrupted 文件存在，就加入 repaint 批量队列
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                repaint_todo.append({
                    "sample_id": sample_id,
                    "name": name,
                    "real": real_path,
                    "mask": mask_path,
                    "corrupted": out_path
                })

        else:
            fail_cnt += 1
            tqdm.write(f"生成corrupted失败：{name} | {msg}")

            # ❌ 记录：corrupted失败
            write_failed(
                sample_id=sample_id,
                stage="corrupted",
                reason=str(msg),
                paths={"real": real_path, "mask": mask_path, "corrupted_out": out_path},
                params={}
            )


    print(f"corrupted 完成：新生成 {ok_cnt}，跳过 {skip_cnt}，失败 {fail_cnt}")

    # ---------------------------------------------------------
    # Step 3：调用 RePaint 批量生成 fake
    # ---------------------------------------------------------
    print("\n步骤3：调用 RePaint 批量生成 fake")

    if not os.path.exists(base_conf):
        print(f"找不到 RePaint 配置文件：{base_conf}")
        return
    # ✅ 3.1 先写 submitted（进入 repaint 队列的样本）
    submitted_cnt = 0
    for item in repaint_todo:
        expected_fake = os.path.join(fake_dir, item["name"])  # 默认输出同名文件
        write_meta(
            sample_id=item["sample_id"],
            stage="repaint",
            status="submitted",
            type="repaint",
            paths={
                "corrupted": item["corrupted"],
                "fake_expected": expected_fake
            },
            params={
                "timestep_respacing": args.timestep_respacing,
                "batch_size": args.batch_size,
                "max_len": args.max_len
            }
        )
        submitted_cnt += 1

    print(f"已提交 repaint 任务数：{submitted_cnt}")

    # ✅ 3.2 运行 RePaint batch
    ok, msg = run_repaint_batch(
        repaint_root=repaint_root,
        base_conf_path=base_conf,
        project_root=PROJECT_ROOT,
        timestep_respacing=args.timestep_respacing,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )

    # ❌ 3.3 如果 batch 运行失败：给这批样本统一记 failed
    if not ok:
        print("\nRePaint 运行失败：")
        print(msg)

        for item in repaint_todo:
            expected_fake = os.path.join(fake_dir, item["name"])
            write_failed(
                sample_id=item["sample_id"],
                stage="repaint",
                reason=f"repaint_batch_failed: {msg}",
                paths={
                    "corrupted": item["corrupted"],
                    "fake_expected": expected_fake
                },
                params={
                    "timestep_respacing": args.timestep_respacing,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len
                }
            )
        return

    print("\nRePaint 运行成功")
    print(f"fake 输出目录：{fake_dir}")
    print("流水线执行结束")
    print("=" * 70)
    # ✅ 3.4 跑完扫描 fake_dir：补写 success / failed
    success_cnt, fail_cnt = 0, 0

    for item in repaint_todo:
        fake_path = os.path.join(fake_dir, item["name"])  # 默认输出同名文件

        if os.path.exists(fake_path) and os.path.getsize(fake_path) > 0:
            success_cnt += 1
            write_meta(
                sample_id=item["sample_id"],
                stage="repaint",
                status="success",
                type="repaint",
                paths={
                    "real": item["real"],
                    "mask": item["mask"],
                    "corrupted": item["corrupted"],
                    "fake": fake_path
                },
                params={
                    "timestep_respacing": args.timestep_respacing,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len
                }
            )
            # ✅ fakeQA（注意 eval_fake_qa 需要有 pass 字段）
            fake_qa = eval_fake_qa(item["real"], fake_path)
            write_meta(
            sample_id=item["sample_id"],
            stage="qa",
            status="success",
            type="repaint",
            paths={
                "real": item["real"],
                "mask": item["mask"],
                "corrupted": item["corrupted"],
                "fake": fake_path
            },
            params={},
            qa={"fake_qa": fake_qa}   # mask_qa 不用重复算
            )
            
        
            

        else:
            fail_cnt += 1
            write_failed(
                sample_id=item["sample_id"],
                stage="repaint",
                reason="fake missing or empty after repaint_batch",
                paths={
                    "corrupted": item["corrupted"],
                    "fake_expected": fake_path
                },
                params={
                    "timestep_respacing": args.timestep_respacing,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len
                }
            )


if __name__ == "__main__":
    main()
