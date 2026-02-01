import os
import json
import time
from typing import Dict, Any, Optional

from .jsonl import append_jsonl

META_PATH = os.path.join("meta", "meta.jsonl")
FAILED_PATH = os.path.join("meta", "failed.jsonl")


# ---------------------------
# 基础工具
# ---------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _file_ok(path: str) -> bool:
    """输出文件是否存在且非空（你也可以在这里加更强校验，比如能否被 cv2 读）"""
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


# ---------------------------
# 写入事件（统一 schema）
# ---------------------------
def write_meta_event(
    sample_id: str,
    stage: str,
    status: str,
    gen_type: str,
    paths: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    qa: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[int] = None,
):
    """
    stage:  "mask" | "corrupted" | "repaint" | "copymove" | "qa"  （你先沿用也行）
    status: "done" | "skipped" | "failed" | "submitted"
    type:   "repaint" | "copymove" | ...
    """
    _ensure_parent(META_PATH)

    obj = {
        "time": _now(),
        "sample_id": sample_id,
        "stage": stage,
        "status": status,          
        "type": gen_type,
        "paths": paths or {},
        "params": params or {},
        "qa": qa or {},
        "error": error or {},
        "duration_ms": duration_ms,
    }
    append_jsonl(META_PATH, obj)


def write_failed_event(
    sample_id: str,
    stage: str,
    reason: str,
    gen_type: str,
    paths: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    err_type: str = "ERROR",
    trace: str = "",
):
    """
    失败写两份：
    1) meta.jsonl 里写一条 status=failed（断点续跑只读 meta 也够）
    2) failed.jsonl 里再写一条（方便你单独统计失败）
    """
    error = {"reason": str(reason), "err_type": err_type, "trace": (trace or "")[:2000]}

    write_meta_event(
        sample_id=sample_id,
        stage=stage,
        status="failed",
        gen_type=gen_type,
        paths=paths,
        params=params or {},
        qa={},
        error=error,
        duration_ms=None,
    )

    _ensure_parent(FAILED_PATH)
    obj = {
        "time": _now(),
        "sample_id": sample_id,
        "stage": stage,
        "reason": str(reason),
        "type": gen_type,
        "paths": paths or {},
        "params": params or {},
    }
    append_jsonl(FAILED_PATH, obj)


# ---------------------------
# 断点续跑：读取进度 + 跳过判断
# ---------------------------
def load_meta_index(meta_path: str = META_PATH) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    返回 index[sample_id][stage] = 最新事件（最后一条为准）
    """
    index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not os.path.exists(meta_path):
        return index

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            sid = obj.get("sample_id")
            stage = obj.get("stage")
            if not sid or not stage:
                continue

            index.setdefault(sid, {})[stage] = obj

    return index


def is_done_and_exists(
    index: Dict[str, Dict[str, Dict[str, Any]]],
    sample_id: str,
    stage: str,
    *,
    output_path: Optional[str] = None,
    output_key: Optional[str] = None,
) -> bool:
    """
    只有满足：
    1) meta 里该 stage 最新 status 是 done 或 skipped
    2) 输出文件存在且非空
    才跳过
    """
    rec = index.get(sample_id, {}).get(stage)
    if not rec:
        return False

    status = (rec.get("status") or "").lower()
    if status not in {"done", "skipped"}:
        return False

    # 优先使用显式传入的 output_path；否则从 meta.paths 取
    p = output_path
    if not p and output_key:
        p = (rec.get("paths") or {}).get(output_key)

    return _file_ok(p)
