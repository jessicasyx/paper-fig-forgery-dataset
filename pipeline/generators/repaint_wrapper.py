import os
import time
import subprocess
from typing import Tuple, Dict, Any

import yaml
from tqdm import tqdm


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _to_relpath(target: str, base_dir: str) -> str:
    """
    把绝对路径 target 转成相对 base_dir 的相对路径，并统一成 '/'，避免 Windows 反斜杠坑
    """
    rel = os.path.relpath(target, base_dir)
    return rel.replace("\\", "/")


def run_repaint_batch(
    repaint_root: str,
    base_conf_path: str,
    project_root: str,
    timestep_respacing: str = "250",
    batch_size: int = 2,
    max_len: int = 0,
) -> Tuple[bool, str]:
    """
    稳定版（推荐）：
    - 根据 project_root 推导 real/mask/corrupted/fake/repaint
    - 写入临时 yml（相对 repaint_root，可复现）
    - subprocess.run 直接继承终端输出（可看到 RePaint 自带 tqdm 进度条）
    - 不捕获 stdout，避免 gbk 解码错误
    """

    test_py = os.path.join(repaint_root, "test.py")
    if not os.path.exists(test_py):
        return False, f"test.py 不存在: {test_py}"
    if not os.path.exists(base_conf_path):
        return False, f"配置文件不存在: {base_conf_path}"

    # 你的项目目录结构（固定约定）
    gt_abs = os.path.join(project_root, "data", "real")
    mask_abs = os.path.join(project_root, "data", "mask")
    lrs_abs = os.path.join(project_root, "data", "corrupted")
    srs_abs = os.path.join(project_root, "data", "fake", "repaint")

    # 转成相对 RePaint-main 的路径（保证可复现）
    gt_rel = _to_relpath(gt_abs, repaint_root)
    mask_rel = _to_relpath(mask_abs, repaint_root)
    lrs_rel = _to_relpath(lrs_abs, repaint_root)
    srs_rel = _to_relpath(srs_abs, repaint_root)

    # 读 yml，覆盖 key（你已经确认过这些 key）
    conf = _load_yaml(base_conf_path)

    conf["gt_path"] = gt_rel
    conf["mask_path"] = mask_rel

    conf.setdefault("paths", {})
    conf["paths"]["srs"] = srs_rel
    conf["paths"]["lrs"] = lrs_rel
    conf["paths"]["gts"] = gt_rel
    conf["paths"]["gt_keep_masks"] = mask_rel

    # 速度参数
    conf["timestep_respacing"] = str(timestep_respacing)
    conf["batch_size"] = int(batch_size)
    conf["max_len"] = int(max_len)

    # 建议：确保进度条开
    conf["show_progress"] = True

    # 写临时 yml
    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_conf = os.path.join(repaint_root, "confs", f"_tmp_pipeline_{ts}.yml")
    _dump_yaml(conf, tmp_conf)

    # 调用 test.py（cwd 非常关键）
    cmd = ["python", "test.py", "--conf_path", tmp_conf]
    env = os.environ.copy()

    # Windows 编码保险（防止中文/符号输出导致问题）
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print("\n" + "=" * 70)
    print("开始 RePaint 图像修复")
    print("=" * 70)
    print(f"配置文件: {tmp_conf}")
    print(f"输入目录: {gt_rel}")
    print(f"输出目录: {srs_rel}")
    print(f"批次大小: {batch_size}")
    print(f"时间步数: {timestep_respacing}")
    if max_len > 0:
        print(f"最大处理: {max_len} 张")
    print("=" * 70 + "\n")

    # 核心：不捕获输出，让 RePaint 自己 tqdm 打到终端（稳定 + 有进度条）
    p = subprocess.run(
        cmd,
        cwd=repaint_root,
        env=env,
        check=False,
    )

    # 你想保留临时 yml 方便复现实验：把这段删掉即可
    try:
        os.remove(tmp_conf)
    except Exception:
        pass

    if p.returncode != 0:
        return False, f"RePaint 运行失败，returncode={p.returncode}"

    print("\n" + "=" * 70)
    print("RePaint 处理完成")
    print("=" * 70 + "\n")

    return True, "ok"