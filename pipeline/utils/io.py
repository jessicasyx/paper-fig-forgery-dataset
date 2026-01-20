# pipeline/utils/io.py
import cv2
import numpy as np
from typing import Optional


def imread_cn(image_path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    读取图像（支持中文路径）
    """
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def imwrite_cn(image: np.ndarray, output_path: str, ext: str = ".png") -> bool:
    """
    保存图像（支持中文路径）
    """
    try:
        success, encoded = cv2.imencode(ext, image)
        if not success:
            return False
        encoded.tofile(output_path)
        return True
    except Exception:
        return False
