import time
from typing import Dict, Any
from pipeline.utils.jsonl import append_jsonl


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_meta(meta_path: str, record: Dict[str, Any]) -> None:
    record.setdefault("time", now_str())
    append_jsonl(meta_path, record)


def write_failed(failed_path: str, record: Dict[str, Any]) -> None:
    record.setdefault("time", now_str())
    append_jsonl(failed_path, record)
