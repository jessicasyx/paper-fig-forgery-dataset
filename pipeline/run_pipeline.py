"""
统一的伪造图像生成流水线
支持 RePaint 和 Copy-Move 两种伪造方式

使用示例：
    # RePaint 流程（默认开启断点续跑）
    python pipeline/run_pipeline.py --type repaint 

    # 关闭断点续跑（强制重跑）
    python pipeline/run_pipeline.py --type repaint --no-skip-existing

    # Copy-Move 流程
    python pipeline/run_pipeline.py --type copymove --samples-per-image 4
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

# ---------------------------------------------------------
# 项目路径 & import 修复
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# 模块导入
# ---------------------------------------------------------
from pipeline.steps.step_corrupted import run_step_corrupted
from pipeline.steps.step_mask_generate import run_step_mask_generate
from pipeline.steps.step_mask_qa import run_step_mask_qa
from pipeline.steps.step_fake_qa import run_step_sample_qa
from pipeline.generators.sam_masker import get_default_sam_config, initialize_sam_model
from pipeline.generators.repaint_wrapper import run_repaint_batch
from pipeline.generators.copy_move import generate_fake_images_from_real
from pipeline.utils.ids import make_sample_id
from pipeline.utils.meta_writer import (
    write_meta_event,
    write_failed_event,
    load_meta_index,
    is_done_and_exists,
)

# ---------------------------------------------------------
# 工具函数
# ---------------------------------------------------------
def list_images(real_dir: str, exts: List[str] = [".png"]) -> List[Path]:
    """列出目录中的所有图片"""
    p = Path(real_dir)
    files: List[Path] = []
    for ext in exts:
        files += list(p.glob(f"*{ext}"))
        files += list(p.glob(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def read_image_with_chinese_path(img_path: str) -> np.ndarray:
    """读取图像（支持中文路径）"""
    try:
        with open(img_path, "rb") as f:
            img_data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] 读取图像失败: {img_path}, 错误: {repr(e)}")
        return None


def save_image_with_chinese_path(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """保存图像（支持中文路径）"""
    try:
        ok, encoded_img = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            with open(output_path, "wb") as f:
                f.write(encoded_img.tobytes())
            return True
        return False
    except Exception as e:
        print(f"[ERROR] 保存图像失败: {output_path}, 错误: {repr(e)}")
        return False


def _update_index(index, sample_id: str, stage: str, status: str, paths: dict):
    """让本次运行内的 index 也跟着更新（否则 index 只反映启动时的 meta）"""
    index.setdefault(sample_id, {})[stage] = {
        "sample_id": sample_id,
        "stage": stage,
        "status": status,
        "paths": paths or {},
    }


# ---------------------------------------------------------
# RePaint 流程
# ---------------------------------------------------------
def run_repaint_pipeline(args, index, imgs, real_dir, mask_dir, corrupted_dir, fake_dir):
    """
    RePaint 伪造流程：mask -> corrupted -> repaint
    """
    print("\n" + "=" * 70)
    print("RePaint 流水线（mask -> corrupted -> repaint）")
    print("=" * 70)

    repaint_root = os.path.join(PROJECT_ROOT, "external", "RePaint-main")
    base_conf = os.path.join(repaint_root, "confs", "test_inet256_thick.yml")

    # Step 1: 生成 mask（SAM）
    print("\n步骤1：mask 处理（SAM）")

    config = get_default_sam_config()
    model_type = config["model_type"]

    # 需要生成的 mask 列表（全流程断点续跑：meta + 文件存在双校验）
    missing_masks = []
    for p in imgs:
        out_mask = os.path.join(mask_dir, p.stem + ".png")
        sample_id = make_sample_id(p, real_dir)

        if args.overwrite:
            missing_masks.append(p)
            continue

        # meta 表示 done/skipped 且文件存在 -> 跳过
        if args.skip_existing and is_done_and_exists(index, sample_id, "mask", output_path=out_mask):
            continue

        # 兼容：文件存在但 meta 没记录 -> 补写一条 done，然后跳过
        if args.skip_existing and os.path.exists(out_mask) and os.path.getsize(out_mask) > 0:
            write_meta_event(
                sample_id=sample_id,
                stage="mask",
                status="done",
                gen_type="repaint",
                paths={"real": str(p), "mask": out_mask},
                params={"model_type": model_type},
            )
            _update_index(index, sample_id, "mask", "done", {"mask": out_mask})
            continue

        missing_masks.append(p)

    print(f"mask 待生成数量：{len(missing_masks)}")

    # 生成缺失的 mask
    if len(missing_masks) > 0:
        print(f"\n开始生成 {len(missing_masks)} 个缺失的 mask（SAM）")

        sam_checkpoints = {
            "vit_b": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_b_01ec64.pth"),
            "vit_h": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_h_4b8939.pth"),
            "vit_l": os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_l_0b3195.pth"),
        }

        ckpt = sam_checkpoints.get(model_type)
        if not ckpt or not os.path.exists(ckpt):
            print(f"[ERROR] 找不到 SAM 权重：{ckpt}")
            return

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
                write_meta_event(
                    sample_id=sample_id,
                    stage="mask",
                    status="done",
                    gen_type="repaint",
                    paths={"real": str(img_path), "mask": out_mask},
                    params={"model_type": model_type},
                )
                _update_index(index, sample_id, "mask", "done", {"mask": out_mask})
            else:
                fail_cnt += 1
                tqdm.write(f"生成mask失败：{img_path.name} | {msg}")
                write_failed_event(
                    sample_id=sample_id,
                    stage="mask",
                    reason=str(msg),
                    gen_type="repaint",
                    paths={"real": str(img_path), "mask_out": out_mask},
                    params={"model_type": model_type},
                    err_type="MASK_FAIL",
                )
                _update_index(index, sample_id, "mask", "failed", {"mask": out_mask})

        print(f"mask 生成完成：新生成 {ok_cnt}，失败 {fail_cnt}")
    else:
        print("所有 mask 已完成（断点续跑跳过）")

    # Step 1.5: Mask QA 评估（硬筛选）
    print("\n步骤1.5：Mask QA 评估（硬筛选）")
    
    qa_pass_cnt, qa_fail_cnt = 0, 0
    tier_a_cnt, tier_b_cnt, tier_c_cnt = 0, 0, 0
    valid_imgs = []  # 通过QA的图片列表
    
    # 收集失败样例（按原因分类，每类最多保留3个）
    fail_samples = {}  # {reason: [(name, details), ...]}
    
    for img_path in tqdm(imgs, desc="Mask QA评估", unit="张", ncols=80):
        mask_path = os.path.join(mask_dir, img_path.stem + ".png")
        sample_id = make_sample_id(img_path, real_dir)
        
        # 如果mask不存在，跳过QA
        if not os.path.exists(mask_path) or os.path.getsize(mask_path) == 0:
            continue
        
        # 执行Mask QA评估
        ok, qa_result = run_step_mask_qa(
            mask_path=mask_path,
            real_shape=None
        )
        
        if ok:
            hard_fail = qa_result.get("hard_fail", False)
            tier = qa_result.get("tier", "FAIL")
            area_ratio = qa_result.get("area_ratio", 0)
            reason = qa_result.get("reason", "unknown")
            
            # 记录QA结果到meta
            write_meta_event(
                sample_id=sample_id,
                stage="mask_qa",
                status="done" if not hard_fail else "failed",
                gen_type="repaint",
                paths={"real": str(img_path), "mask": mask_path},
                params={},
                qa={"mask_qa": qa_result},
            )
            
            if not hard_fail:
                qa_pass_cnt += 1
                valid_imgs.append(img_path)
                if tier == "A":
                    tier_a_cnt += 1
                elif tier == "B":
                    tier_b_cnt += 1
                elif tier == "C":
                    tier_c_cnt += 1
            else:
                qa_fail_cnt += 1
                # 收集失败样例（每类最多3个）
                if reason not in fail_samples:
                    fail_samples[reason] = []
                if len(fail_samples[reason]) < 3:
                    fail_samples[reason].append((img_path.name, f"area={area_ratio:.3f}"))
                
                # 记录为失败
                write_failed_event(
                    sample_id=sample_id,
                    stage="mask_qa",
                    reason=f"Hard fail: {reason}",
                    gen_type="repaint",
                    paths={"real": str(img_path), "mask": mask_path},
                    params={"area_ratio": area_ratio, "tier": tier},
                    err_type="MASK_QA_HARD_FAIL",
                )
        else:
            qa_fail_cnt += 1
            error = qa_result.get("error", "unknown")
            # 收集失败样例
            if error not in fail_samples:
                fail_samples[error] = []
            if len(fail_samples[error]) < 3:
                fail_samples[error].append((img_path.name, ""))
            
            write_failed_event(
                sample_id=sample_id,
                stage="mask_qa",
                reason=f"QA evaluation failed: {error}",
                gen_type="repaint",
                paths={"real": str(img_path), "mask": mask_path},
                params={},
                err_type="MASK_QA_ERROR",
            )
    
    # 打印汇总
    print(f"\n【Mask QA 汇总】")
    print(f"  通过: {qa_pass_cnt}")
    print(f"    - Tier A（优先，0.05-0.20）: {tier_a_cnt}")
    print(f"    - Tier B（可用，0.02-0.40）: {tier_b_cnt}")
    print(f"    - Tier C（保留但标记，0.005-0.02 & 0.40-0.50）: {tier_c_cnt}")
    print(f"  淘汰: {qa_fail_cnt}")
    
    # 打印失败样例
    if fail_samples:
        print(f"\n【失败样例】（每类最多显示3个）")
        for reason, samples in fail_samples.items():
            print(f"  原因: {reason} ({len(samples)}个样例)")
            for name, details in samples:
                if details:
                    print(f"    - {name} ({details})")
                else:
                    print(f"    - {name}")
    
    print(f"\n将继续处理 {len(valid_imgs)} 张通过QA的图片")
    
    # 更新imgs列表，只保留通过QA的图片
    imgs = valid_imgs
    
    if len(imgs) == 0:
        print("\n[WARN] 没有图片通过Mask QA评估，流程结束。")
        print("=" * 70)
        return

    # Step 2: 生成 corrupted（挖洞图）
    print("\n步骤2：生成 corrupted（挖洞图）")

    ok_cnt, skip_cnt, fail_cnt = 0, 0, 0
    repaint_todo = []

    for img_path in tqdm(imgs, desc="处理corrupted", unit="张"):
        name = img_path.name
        real_path = os.path.join(real_dir, name)
        mask_path = os.path.join(mask_dir, name)
        out_path = os.path.join(corrupted_dir, name)
        sample_id = make_sample_id(img_path, real_dir)

        # 断点续跑：corrupted 已完成 -> 跳过生成，但仍加入 repaint_todo（后面 fake 可能还没生成）
        if (not args.overwrite) and args.skip_existing and is_done_and_exists(index, sample_id, "corrupted", output_path=out_path):
            skip_cnt += 1
            repaint_todo.append({
                "sample_id": sample_id,
                "name": name,
                "real": real_path,
                "mask": mask_path,
                "corrupted": out_path,
            })
            continue

        # 兼容：corrupted 文件存在但 meta 没记录 -> 补写 done 并跳过生成
        if (not args.overwrite) and args.skip_existing and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            write_meta_event(
                sample_id=sample_id,
                stage="corrupted",
                status="done",
                gen_type="repaint",
                paths={"real": real_path, "mask": mask_path, "corrupted": out_path},
                params={},
            )
            _update_index(index, sample_id, "corrupted", "done", {"corrupted": out_path})
            skip_cnt += 1
            repaint_todo.append({
                "sample_id": sample_id,
                "name": name,
                "real": real_path,
                "mask": mask_path,
                "corrupted": out_path,
            })
            continue

        ok, msg = run_step_corrupted(
            real_path=real_path,
            mask_keep_path=mask_path,
            corrupted_path=out_path,
            overwrite=args.overwrite,
        )

        if ok:
            # step_corrupted 可能返回 exists_skip（如果你内部还保留了文件跳过）
            if msg == "exists_skip":
                skip_cnt += 1
                write_meta_event(
                    sample_id=sample_id,
                    stage="corrupted",
                    status="skipped",
                    gen_type="repaint",
                    paths={"real": real_path, "mask": mask_path, "corrupted": out_path},
                    params={},
                )
                _update_index(index, sample_id, "corrupted", "skipped", {"corrupted": out_path})
            else:
                ok_cnt += 1
                write_meta_event(
                    sample_id=sample_id,
                    stage="corrupted",
                    status="done",
                    gen_type="repaint",
                    paths={"real": real_path, "mask": mask_path, "corrupted": out_path},
                    params={},
                )
                _update_index(index, sample_id, "corrupted", "done", {"corrupted": out_path})

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                repaint_todo.append({
                    "sample_id": sample_id,
                    "name": name,
                    "real": real_path,
                    "mask": mask_path,
                    "corrupted": out_path,
                })
        else:
            fail_cnt += 1
            tqdm.write(f"生成corrupted失败：{name} | {msg}")
            write_failed_event(
                sample_id=sample_id,
                stage="corrupted",
                reason=str(msg),
                gen_type="repaint",
                paths={"real": real_path, "mask": mask_path, "corrupted_out": out_path},
                params={},
                err_type="CORRUPTED_FAIL",
            )
            _update_index(index, sample_id, "corrupted", "failed", {"corrupted": out_path})

    print(f"corrupted 完成：新生成 {ok_cnt}，跳过/已完成 {skip_cnt}，失败 {fail_cnt}")

    # Step 3: 调用 RePaint 批量生成 fake
    print("\n步骤3：调用 RePaint 批量生成 fake")

    import shutil

    if not os.path.exists(base_conf):
        print(f"[ERROR] 找不到 RePaint 配置文件：{base_conf}")
        return

    # 断点续跑：已经有 fake 的不要再提交 repaint
    filtered = []
    already_done = 0
    for item in repaint_todo:
        sid = item["sample_id"]
        fake_path = os.path.join(fake_dir, item["name"])

        if (not args.overwrite) and args.skip_existing and is_done_and_exists(index, sid, "repaint", output_path=fake_path):
            already_done += 1
            continue

        if (not args.overwrite) and args.skip_existing and os.path.exists(fake_path) and os.path.getsize(fake_path) > 0:
            write_meta_event(
                sample_id=sid,
                stage="repaint",
                status="done",
                gen_type="repaint",
                paths={"real": item["real"], "mask": item["mask"], "corrupted": item["corrupted"], "fake": fake_path},
                params={
                    "timestep_respacing": args.timestep_respacing,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len,
                },
            )
            _update_index(index, sid, "repaint", "done", {"fake": fake_path})
            already_done += 1
            continue

        filtered.append(item)

    repaint_todo = filtered
    print(f"repaint 待提交：{len(repaint_todo)}，已完成跳过：{already_done}")

    if len(repaint_todo) == 0:
        print("没有需要 RePaint 生成的样本，结束。")
        print(f"输出目录：{fake_dir}")
        print("=" * 70)
        return


    # -----------------------------
    # 准备 RePaint 临时输入目录（解决 gt/mask 不匹配 & 空目录）
    # -----------------------------
    tmp_root = os.path.join(PROJECT_ROOT, "data", "_repaint_tmp")
    tmp_gt_dir = os.path.join(tmp_root, "gt")      # 对应 yml 的 gt_path
    tmp_mask_dir = os.path.join(tmp_root, "mask")  # 对应 yml 的 mask_path
    os.makedirs(tmp_gt_dir, exist_ok=True)
    os.makedirs(tmp_mask_dir, exist_ok=True)

    # 清空旧内容（避免上次残留导致错配）
    for d in (tmp_gt_dir, tmp_mask_dir):
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            try:
                if os.path.isdir(fp):
                    shutil.rmtree(fp)
                else:
                    os.remove(fp)
            except Exception:
                pass

    # 拷贝本次要跑的 corrupted + mask 到临时目录
    copied = 0
    skipped_pair = 0
    for item in repaint_todo:
        src_corrupted = item["corrupted"]
        src_mask = item["mask"]

        if (not os.path.exists(src_corrupted)) or os.path.getsize(src_corrupted) == 0:
            skipped_pair += 1
            continue
        if (not os.path.exists(src_mask)) or os.path.getsize(src_mask) == 0:
            skipped_pair += 1
            continue

        shutil.copy2(src_corrupted, os.path.join(tmp_gt_dir, os.path.basename(src_corrupted)))
        shutil.copy2(src_mask, os.path.join(tmp_mask_dir, os.path.basename(src_mask)))
        copied += 1

    print(f"[RePaint] 临时输入准备完成：copied={copied}, skipped_pair={skipped_pair}")
    print(f"[RePaint] gt_path  -> {tmp_gt_dir}")
    print(f"[RePaint] mask_path-> {tmp_mask_dir}")

    if copied == 0:
        print("[ERROR] 临时输入目录为空：没有可用于 RePaint 的 (corrupted, mask) 对。请检查 corrupted/mask 是否生成成功。")
        return


    # 记录提交的任务
    for item in repaint_todo:
        expected_fake = os.path.join(fake_dir, item["name"])
        write_meta_event(
            sample_id=item["sample_id"],
            stage="repaint",
            status="submitted",
            gen_type="repaint",
            paths={"corrupted": item["corrupted"], "fake_expected": expected_fake},
            params={
                "timestep_respacing": args.timestep_respacing,
                "batch_size": args.batch_size,
                "max_len": args.max_len,
            },
        )
        _update_index(index, item["sample_id"], "repaint", "submitted", {"fake_expected": expected_fake})

    print(f"已提交 repaint 任务数：{len(repaint_todo)}")

    # 运行 RePaint
    ok, msg = run_repaint_batch(
        repaint_root=repaint_root,
        base_conf_path=base_conf,
        project_root=PROJECT_ROOT,
        timestep_respacing=args.timestep_respacing,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )

    if not ok:
        print(f"\n[ERROR] RePaint 运行失败：{msg}")
        for item in repaint_todo:
            expected_fake = os.path.join(fake_dir, item["name"])
            write_failed_event(
                sample_id=item["sample_id"],
                stage="repaint",
                reason=f"repaint_batch_failed: {msg}",
                gen_type="repaint",
                paths={"corrupted": item["corrupted"], "fake_expected": expected_fake},
                params={
                    "timestep_respacing": args.timestep_respacing,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len,
                },
                err_type="REPAINT_BATCH_FAIL",
            )
            _update_index(index, item["sample_id"], "repaint", "failed", {"fake_expected": expected_fake})
        return

    # 检查生成结果
    print("\nRePaint 运行成功，检查生成结果...")
    success_cnt, fail_cnt = 0, 0

    for item in repaint_todo:
        fake_path = os.path.join(fake_dir, item["name"])

        if os.path.exists(fake_path) and os.path.getsize(fake_path) > 0:
            success_cnt += 1
            write_meta_event(
                sample_id=item["sample_id"],
                stage="repaint",
                status="done",
                gen_type="repaint",
                paths={
                    "real": item["real"],
                    "mask": item["mask"],
                    "corrupted": item["corrupted"],
                    "fake": fake_path,
                },
                params={
                    "timestep_respacing": args.timestep_respacing,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len,
                },
            )
            _update_index(index, item["sample_id"], "repaint", "done", {"fake": fake_path})
        else:
            fail_cnt += 1
            write_failed_event(
                sample_id=item["sample_id"],
                stage="repaint",
                reason="fake missing or empty after repaint_batch",
                gen_type="repaint",
                paths={"corrupted": item["corrupted"], "fake_expected": fake_path},
                params={
                    "timestep_respacing": args.timestep_respacing,
                    "batch_size": args.batch_size,
                    "max_len": args.max_len,
                },
                err_type="REPAINT_OUTPUT_MISSING",
            )
            _update_index(index, item["sample_id"], "repaint", "failed", {"fake_expected": fake_path})

    print(f"\nRePaint 流程完成：成功 {success_cnt}，失败 {fail_cnt}")
    
    # Step 4: Fake QA 评估（灾难级筛选）
    print("\n步骤4：Fake QA 评估（灾难级筛选）")
    
    fake_qa_pass_cnt, fake_qa_fail_cnt = 0, 0
    
    # 收集失败样例和统计信息
    fail_samples = {}  # {reason: [(name, details), ...]}
    score_list = []  # 记录所有通过样本的评分
    
    for item in tqdm(repaint_todo, desc="Fake QA评估", unit="张", ncols=80):
        fake_path = os.path.join(fake_dir, item["name"])
        sample_id = item["sample_id"]
        
        # 只对成功生成的fake进行QA
        if not os.path.exists(fake_path) or os.path.getsize(fake_path) == 0:
            continue
        
        # 执行Fake QA评估
        ok, qa_result = run_step_sample_qa(
            real_path=item["real"],
            mask_path=item["mask"],
            fake_path=fake_path
        )
        
        if ok:
            hard_fail = qa_result.get("hard_fail", False)
            reason = qa_result.get("reason", "unknown")
            score = qa_result.get("score", 0)
            diff_in = qa_result.get("diff_in", 0)
            diff_out = qa_result.get("diff_out", 0)
            
            # 记录QA结果到meta
            write_meta_event(
                sample_id=sample_id,
                stage="fake_qa",
                status="done" if not hard_fail else "failed",
                gen_type="repaint",
                paths={
                    "real": item["real"],
                    "mask": item["mask"],
                    "fake": fake_path
                },
                params={},
                qa={"fake_qa": qa_result},
            )
            
            if not hard_fail:
                fake_qa_pass_cnt += 1
                score_list.append(score)
            else:
                fake_qa_fail_cnt += 1
                # 收集失败样例（每类最多3个）
                if reason not in fail_samples:
                    fail_samples[reason] = []
                if len(fail_samples[reason]) < 3:
                    fail_samples[reason].append((item['name'], f"in={diff_in:.1f}, out={diff_out:.1f}"))
                
                # 删除不合格的fake图像
                try:
                    os.remove(fake_path)
                except Exception:
                    pass
                
                # 记录为失败
                write_failed_event(
                    sample_id=sample_id,
                    stage="fake_qa",
                    reason=f"Hard fail: {reason}",
                    gen_type="repaint",
                    paths={
                        "real": item["real"],
                        "mask": item["mask"],
                        "fake": fake_path
                    },
                    params={"score": score, "diff_in": diff_in, "diff_out": diff_out},
                    err_type="FAKE_QA_HARD_FAIL",
                )
        else:
            fake_qa_fail_cnt += 1
            error = qa_result.get("error", "unknown")
            # 收集失败样例
            if error not in fail_samples:
                fail_samples[error] = []
            if len(fail_samples[error]) < 3:
                fail_samples[error].append((item['name'], ""))
    
    # 打印汇总
    print(f"\n【Fake QA 汇总】")
    print(f"  通过: {fake_qa_pass_cnt}")
    print(f"  灾难级淘汰: {fake_qa_fail_cnt}")
    
    if score_list:
        avg_score = sum(score_list) / len(score_list)
        min_score = min(score_list)
        max_score = max(score_list)
        print(f"  评分统计: 平均={avg_score:.1f}, 最低={min_score:.1f}, 最高={max_score:.1f}")
    
    # 打印失败样例
    if fail_samples:
        print(f"\n【失败样例】（每类最多显示3个）")
        for reason, samples in fail_samples.items():
            print(f"  原因: {reason} ({len(samples)}个样例)")
            for name, details in samples:
                if details:
                    print(f"    - {name} ({details})")
                else:
                    print(f"    - {name}")
    
    print(f"\n输出目录：{fake_dir}")
    print("=" * 70)


# ---------------------------------------------------------
# Copy-Move 流程
# ---------------------------------------------------------
def run_copymove_pipeline(args, index, imgs, real_dir, mask_dir, fake_dir):
    """
    Copy-Move 伪造流程：mask(内部可能生成) -> copy-move
    """
    print("\n" + "=" * 70)
    print("Copy-Move 流水线（mask -> copy-move）")
    print("=" * 70)

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "sam", "sam_vit_l_0b3195.pth")
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] SAM checkpoint not found: {checkpoint_path}")
        return

    print(f"SAM checkpoint: {checkpoint_path}")
    print(f"每张图片生成样本数: {args.samples_per_image}")

    success_cnt, fail_cnt, skip_cnt = 0, 0, 0

    for img_path in imgs:
        sample_id = make_sample_id(img_path, real_dir)

        # 断点续跑：如果已经有足够 fake，就跳过这张 real
        if (not args.overwrite) and args.skip_existing:
            existing = list(Path(fake_dir).glob(f"{img_path.stem}_fake_*.jpg"))
            if len(existing) >= args.samples_per_image:
                skip_cnt += 1
                write_meta_event(
                    sample_id=sample_id,
                    stage="copymove",
                    status="skipped",
                    gen_type="copymove",
                    paths={"real": str(img_path), "fake_dir": fake_dir},
                    params={"samples_per_image": args.samples_per_image, "existing": len(existing)},
                )
                _update_index(index, sample_id, "copymove", "skipped", {"fake_dir": fake_dir})
                continue

        print(f"\n处理图片: {img_path.name}")

        real_image = read_image_with_chinese_path(str(img_path))
        if real_image is None:
            print(f"[ERROR] 无法读取图像: {img_path}")
            fail_cnt += 1
            write_failed_event(
                sample_id=sample_id,
                stage="copymove",
                reason="read real image failed",
                gen_type="copymove",
                paths={"real": str(img_path)},
                params={"samples_per_image": args.samples_per_image},
                err_type="IMAGE_READ_FAIL",
            )
            _update_index(index, sample_id, "copymove", "failed", {"real": str(img_path)})
            continue

        try:
            fake_images = generate_fake_images_from_real(
                real_image,
                checkpoint_path,
                num_samples=args.samples_per_image,
                image_name=img_path.stem,
                mask_dir=mask_dir,
            )

            if len(fake_images) == 0:
                print(f"[WARN] 未生成任何伪造图像")
                fail_cnt += 1
                write_failed_event(
                    sample_id=sample_id,
                    stage="copymove",
                    reason="no fake generated",
                    gen_type="copymove",
                    paths={"real": str(img_path), "fake_dir": fake_dir},
                    params={"samples_per_image": args.samples_per_image},
                    err_type="NO_FAKE",
                )
                _update_index(index, sample_id, "copymove", "failed", {"fake_dir": fake_dir})
                continue

            print(f"  [保存] 保存伪造图像...")
            saved_cnt = 0
            saved_paths = []
            for idx, fake_image in enumerate(fake_images):
                output_name = f"{img_path.stem}_fake_{idx}.jpg"
                output_path = os.path.join(fake_dir, output_name)

                if (not args.overwrite) and args.skip_existing and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    # 单张已存在就跳过保存
                    saved_cnt += 1
                    saved_paths.append(output_path)
                    continue

                if save_image_with_chinese_path(fake_image, output_path):
                    saved_cnt += 1
                    saved_paths.append(output_path)
                else:
                    print(f"  [ERROR] 保存失败: {output_name}")

            print(f"  完成：保存 {saved_cnt}/{len(fake_images)} 张伪造图像")
            success_cnt += 1

            write_meta_event(
                sample_id=sample_id,
                stage="copymove",
                status="done",
                gen_type="copymove",
                paths={"real": str(img_path), "fake_dir": fake_dir},
                params={"samples_per_image": args.samples_per_image, "saved": saved_cnt, "fake_paths": saved_paths},
            )
            _update_index(index, sample_id, "copymove", "done", {"fake_dir": fake_dir})

        except Exception as e:
            print(f"[ERROR] 处理图片失败: {img_path.name}, 错误: {repr(e)}")
            fail_cnt += 1
            write_failed_event(
                sample_id=sample_id,
                stage="copymove",
                reason=repr(e),
                gen_type="copymove",
                paths={"real": str(img_path), "fake_dir": fake_dir},
                params={"samples_per_image": args.samples_per_image},
                err_type="COPYMOVE_EXCEPTION",
                trace="",
            )
            _update_index(index, sample_id, "copymove", "failed", {"fake_dir": fake_dir})
            continue

    print(f"\nCopy-Move 流程完成：成功 {success_cnt}，跳过 {skip_cnt}，失败 {fail_cnt}")
    print(f"输出目录：{fake_dir}")
    print("=" * 70)


# ---------------------------------------------------------
# 主函数
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="统一的伪造图像生成流水线（RePaint / Copy-Move）"
    )

    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["repaint", "copymove"],
        help="选择伪造流程：repaint 或 copymove",
    )

    # 默认开启断点续跑；关闭用 --no-skip-existing
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="关闭断点续跑：即使已有输出也重新跑（不推荐）")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已有输出（谨慎使用）")
    parser.add_argument("--max-len", type=int, default=0, help="最多处理多少张图片，0表示不限制")

    # RePaint 专用参数
    parser.add_argument("--timestep-respacing", type=str, default="250", help="[RePaint] 采样步数")
    parser.add_argument("--batch-size", type=int, default=4, help="[RePaint] 批次大小")

    # Copy-Move 专用参数
    parser.add_argument("--samples-per-image", type=int, default=1, help="[Copy-Move] 每张图片生成的伪造样本数")

    args = parser.parse_args()

    # 固定目录
    real_dir = os.path.join(PROJECT_ROOT, "data", "real")
    mask_dir = os.path.join(PROJECT_ROOT, "data", "mask")
    corrupted_dir = os.path.join(PROJECT_ROOT, "data", "corrupted")
    fake_dir_repaint = os.path.join(PROJECT_ROOT, "data", "fake", "repaint")
    fake_dir_copymove = os.path.join(PROJECT_ROOT, "data", "fake", "copy_move")

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(corrupted_dir, exist_ok=True)
    os.makedirs(fake_dir_repaint, exist_ok=True)
    os.makedirs(fake_dir_copymove, exist_ok=True)

    # 读取 meta 进度
    index = load_meta_index()
    print(f"已加载 meta 进度：{len(index)} 个 sample")

    # 打印配置
    print("=" * 70)
    print("伪造图像生成流水线")
    print("=" * 70)
    print(f"流程类型: {args.type}")
    print(f"项目路径: {PROJECT_ROOT}")
    print(f"真实图像: {real_dir}")
    print(f"掩码目录: {mask_dir}")
    print(f"断点续跑(跳过已完成): {args.skip_existing}")
    print(f"强制覆盖: {args.overwrite}")
    print(f"最大处理数: {args.max_len if args.max_len > 0 else '不限制'}")

    if args.type == "repaint":
        print(f"时间步数: {args.timestep_respacing}")
        print(f"批次大小: {args.batch_size}")
    else:
        print(f"每图样本数: {args.samples_per_image}")

    # 收集图片
    imgs = list_images(real_dir, exts=[".png"])
    if not imgs:
        print("\n[ERROR] real_dir 下没有 png 图片，退出。")
        return

    if args.max_len > 0:
        imgs = imgs[:args.max_len]

    print(f"\n待处理图片数量：{len(imgs)}")

    # 执行对应的流程
    if args.type == "repaint":
        run_repaint_pipeline(args, index, imgs, real_dir, mask_dir, corrupted_dir, fake_dir_repaint)
    elif args.type == "copymove":
        run_copymove_pipeline(args, index, imgs, real_dir, mask_dir, fake_dir_copymove)


if __name__ == "__main__":
    main()
