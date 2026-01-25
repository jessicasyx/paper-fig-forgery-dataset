import hashlib
from pathlib import Path

def make_sample_id(real_path: str | Path, real_root: str | Path) -> str:
    """
    用 real_path 相对 real_root 的路径生成稳定 sample_id
    不随机器路径变化，适用于 Windows/Linux

    例:
      real_root = data/real
      real_path = data/real/其他/abc.png
      -> sample_id = "abc_1a2b3c4d"
    """
    real_path = Path(real_path)
    real_root = Path(real_root)

    rel = real_path.relative_to(real_root).as_posix()  # 统一成 / 分隔
    stem = real_path.stem  # abc（不含后缀）
    h = hashlib.md5(rel.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{h}"
