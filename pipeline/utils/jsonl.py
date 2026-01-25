import os
import json
from typing import Dict, Any


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """追加写入一行 jsonl（UTF-8，支持中文）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
