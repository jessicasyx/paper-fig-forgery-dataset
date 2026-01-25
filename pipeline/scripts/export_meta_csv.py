import json
import os
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise SystemExit("请先安装 pandas：pip install pandas")

# 获取脚本所在目录，然后计算项目根目录
SCRIPT_DIR = Path(__file__).parent.resolve()  # pipeline/scripts/
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # 往上退两级到项目根目录

META_JSONL = PROJECT_ROOT / "meta" / "meta.jsonl"
OUT_CSV = PROJECT_ROOT / "meta" / "meta.csv"

rows = []
if META_JSONL.exists():
    with META_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
else:
    raise SystemExit(f"找不到文件：{META_JSONL}")

df = pd.json_normalize(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"✅ 导出完成：{OUT_CSV} （共 {len(df)} 条记录）")
