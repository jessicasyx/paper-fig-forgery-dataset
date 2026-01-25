import os, time
from .jsonl import append_jsonl

META_PATH = os.path.join("meta", "meta.jsonl")
FAILED_PATH = os.path.join("meta", "failed.jsonl")

def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def write_meta(sample_id, stage, status, type, paths, params=None, qa=None):
    obj = {
        "time": _now(),
        "sample_id": sample_id,
        "stage": stage,
        "status": status,
        "type": type,
        "paths": paths,
        "params": params or {},
        "qa": qa or {}
    }
    append_jsonl(META_PATH, obj)

def write_failed(sample_id, stage, reason, paths, params=None):
    obj = {
        "time": _now(),
        "sample_id": sample_id,
        "stage": stage,
        "reason": reason,
        "paths": paths,
        "params": params or {}
    }
    append_jsonl(FAILED_PATH, obj)
