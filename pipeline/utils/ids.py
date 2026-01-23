import hashlib


def make_sample_id(filename: str) -> str:
    """
    用文件名生成稳定 sample_id
    例如: "abc.png" -> "abc_1a2b3c4d"
    """
    stem = filename.rsplit(".", 1)[0]
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{h}"
