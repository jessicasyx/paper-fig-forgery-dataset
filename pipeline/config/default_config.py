"""
默认配置文件
集中管理项目的所有配置参数
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ============================================================
# 目录配置
# ============================================================
DIRS = {
    "real": PROJECT_ROOT / "data" / "real",
    "mask": PROJECT_ROOT / "data" / "mask",
    "corrupted": PROJECT_ROOT / "data" / "corrupted",
    "fake_repaint": PROJECT_ROOT / "data" / "fake" / "repaint",
    "fake_copymove": PROJECT_ROOT / "data" / "fake" / "copy_move",
    "checkpoints": PROJECT_ROOT / "checkpoints",
    "sam_checkpoints": PROJECT_ROOT / "checkpoints" / "sam",
    "meta": PROJECT_ROOT / "meta",
    "logs": PROJECT_ROOT / "logs",
    "repaint_root": PROJECT_ROOT / "external" / "RePaint-main",
}

# ============================================================
# SAM 配置
# ============================================================
SAM_CONFIG = {
    "model_type": "vit_b",  # vit_b, vit_l, vit_h
    "max_image_size": 800,
    "resize_enabled": True,
    "invert_mask": True,  # 黑=编辑区域，白=保留区域
    "sam_params": {
        "points_per_side": 16,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "crop_n_layers": 0,
        "min_mask_region_area": 500,
    }
}

# SAM 模型检查点路径
SAM_CHECKPOINTS = {
    "vit_b": DIRS["sam_checkpoints"] / "sam_vit_b_01ec64.pth",
    "vit_l": DIRS["sam_checkpoints"] / "sam_vit_l_0b3195.pth",
    "vit_h": DIRS["sam_checkpoints"] / "sam_vit_h_4b8939.pth",
}

# ============================================================
# RePaint 配置
# ============================================================
REPAINT_CONFIG = {
    "base_conf": "test_inet256_thick.yml",
    "timestep_respacing": "250",
    "batch_size": 4,
    "max_len": 0,  # 0 表示不限制
    "show_progress": True,
}

# ============================================================
# Copy-Move 配置
# ============================================================
COPYMOVE_CONFIG = {
    "samples_per_image": 4,
    "feather_width": 5,
    "jpeg_quality": 85,
    "brightness_alpha": 1.05,
    "brightness_beta": 10,
}

# ============================================================
# 质量评估配置
# ============================================================
QA_CONFIG = {
    "mask": {
        "min_ratio": 0.02,
        "max_ratio": 0.20,
        "max_edge": 0.10,
    },
    "fake": {
        "max_diff_mean": 20,
    }
}

# ============================================================
# 图像处理配置
# ============================================================
IMAGE_CONFIG = {
    "supported_formats": [".png", ".jpg", ".jpeg"],
    "repaint_format": ".png",
    "copymove_format": ".jpg",
    "jpeg_quality": 95,
}

# ============================================================
# 元数据配置
# ============================================================
META_CONFIG = {
    "meta_file": DIRS["meta"] / "meta.jsonl",
    "failed_file": DIRS["meta"] / "failed.jsonl",
    "csv_file": DIRS["meta"] / "meta.csv",
}

# ============================================================
# 日志配置
# ============================================================
LOG_CONFIG = {
    "log_dir": DIRS["logs"],
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# ============================================================
# 辅助函数
# ============================================================
def ensure_dirs():
    """确保所有必要的目录存在"""
    for dir_path in DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)


def get_sam_checkpoint(model_type: str = "vit_b") -> Path:
    """获取 SAM 检查点路径"""
    checkpoint = SAM_CHECKPOINTS.get(model_type)
    if checkpoint is None:
        raise ValueError(f"Unknown SAM model type: {model_type}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    return checkpoint


def validate_config():
    """验证配置的有效性"""
    errors = []
    
    # 检查 SAM 检查点
    for model_type, checkpoint in SAM_CHECKPOINTS.items():
        if not checkpoint.exists():
            errors.append(f"SAM checkpoint missing: {checkpoint}")
    
    # 检查 RePaint 目录
    if not DIRS["repaint_root"].exists():
        errors.append(f"RePaint directory not found: {DIRS['repaint_root']}")
    
    return errors


if __name__ == "__main__":
    # 测试配置
    print("=" * 70)
    print("项目配置验证")
    print("=" * 70)
    
    print("\n目录配置：")
    for name, path in DIRS.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name}: {path}")
    
    print("\nSAM 检查点：")
    for model_type, checkpoint in SAM_CHECKPOINTS.items():
        status = "✓" if checkpoint.exists() else "✗"
        size = f"{checkpoint.stat().st_size / 1024**2:.1f}MB" if checkpoint.exists() else "N/A"
        print(f"  {status} {model_type}: {checkpoint} ({size})")
    
    print("\n配置验证：")
    errors = validate_config()
    if errors:
        print("  发现以下问题：")
        for error in errors:
            print(f"    ✗ {error}")
    else:
        print("  ✓ 配置验证通过")
    
    print("\n" + "=" * 70)

