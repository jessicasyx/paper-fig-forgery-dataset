# 论文图像伪造数据集生成项目

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个用于生成论文图像伪造数据集的自动化流水线项目，支持多种伪造方式（RePaint 扩散模型、Copy-Move）和完整的质量评估系统。

---
 🎯 项目特点

- 🤖 **自动化流水线**：从真实图像到伪造图像的全自动生成
- 🎨 **多种伪造方式**：支持 RePaint（扩散模型）和 Copy-Move（复制粘贴）
- 🔍 **智能掩码生成**：基于 SAM (Segment Anything Model) 的自动掩码生成
- ✅ **质量评估系统**：自动过滤低质量样本
- 📊 **完整元数据追踪**：记录每个样本的完整生命周期
- 🔄 **断点续跑**：支持跳过已生成的样本，避免重复计算
- 🌏 **中文路径支持**：完美支持包含中文的文件路径

---

 📁 项目结构

```
paper-fig-forgery-dataset/
├── checkpoints/          # 模型权重文件
│   └── sam/             # SAM 模型检查点
├── data/                # 数据目录
│   ├── real/           # 原始真实图像（输入）
│   ├── mask/           # SAM 生成的掩码
│   ├── corrupted/      # 挖洞图（中间产物）
│   └── fake/           # 生成的伪造图像（输出）
│       ├── repaint/    # RePaint 生成
│       └── copy_move/  # Copy-Move 生成
├── pipeline/            # 核心流水线代码
│   ├── generators/     # 生成器模块
│   ├── steps/          # 流水线步骤
│   ├── utils/          # 工具函数
│   └── run_pipeline.py # 主执行脚本
├── meta/               # 元数据目录
│   ├── meta.jsonl     # 主元数据文件
│   └── failed.jsonl   # 失败记录
└── docs/              # 文档
    └── PROJECT_STRUCTURE.md  # 详细项目结构文档
```

---

## 🚀 快速开始

### 1. 环境配置

#### 安装 PyTorch（根据您的 CUDA 版本）

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 安装项目依赖

```bash
pip install -r requirements.txt
```

#### 验证安装

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
python -c "import cv2; import segment_anything; print('Dependencies OK')"
```

### 2. 准备数据

将原始图像（PNG 格式）放入 `data/real/` 目录：

```bash
data/real/
├── image1.png
├── image2.png
└── ...
```

### 3. 下载模型权重

#### SAM 模型权重

下载并放入 `checkpoints/sam/` 目录：

- [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) (375MB)
- [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) (1.2GB)
- [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (2.4GB)

#### RePaint 预训练模型

参考 `external/RePaint-main/README.md` 下载预训练模型。

### 4. 运行流水线

#### RePaint 流程（扩散模型修复）

```bash
python pipeline/run_pipeline.py --type repaint --max-len 10 --batch-size 2 --skip-existing
```

**参数说明：**
- `--type repaint`：使用 RePaint 流水线
- `--max-len 10`：最多处理 10 张图像（0=不限制）
- `--batch-size 2`：RePaint 批处理大小
- `--skip-existing`：跳过已存在的输出（推荐）
- `--timestep-respacing 250`：RePaint 时间步数（默认 250）

#### Copy-Move 流程（复制粘贴伪造）

```bash
python pipeline/run_pipeline.py --type copymove --max-len 10 --samples-per-image 5
```

**参数说明：**
- `--type copymove`：使用 Copy-Move 流水线
- `--samples-per-image 4`：每张图片生成 4 个伪造样本

### 5. 查看结果

生成的文件：
- `data/mask/*.png` - 掩码图像
- `data/corrupted/*.png` - 挖洞图（仅 RePaint）
- `data/fake/repaint/*.png` - RePaint 生成的伪造图
- `data/fake/copy_move/*.jpg` - Copy-Move 生成的伪造图
- `meta/meta.jsonl` - 元数据记录
- `meta/failed.jsonl` - 失败记录

---

## 📊 流水线流程

### RePaint 流程

```
输入：data/real/*.png
  ↓
步骤1：生成掩码 (SAM)
  → 输出：data/mask/*.png
  ↓
步骤2：生成挖洞图
  → 输出：data/corrupted/*.png
  ↓
步骤3：RePaint 修复生成伪造图
  → 输出：data/fake/repaint/*.png
  ↓
输出：meta/meta.jsonl（元数据）
```

### Copy-Move 流程

```
输入：data/real/*.png
  ↓
步骤1：生成掩码 (SAM)
  → 输出：data/mask/*.png
  ↓
步骤2：复制-粘贴伪造
  → 输出：data/fake/copy_move/*.jpg
  ↓
输出：meta/meta.jsonl（元数据）
```

---

## 🔧 配置说明

### SAM 配置

默认配置位于 `pipeline/generators/sam_masker.py`：

```python
{
    "model_type": "vit_b",      # 模型类型：vit_b, vit_l, vit_h
    "max_image_size": 800,      # 缩放最大边长
    "resize_enabled": True,     # 是否启用缩放
    "invert_mask": True,        # 黑=编辑区域，白=保留区域
    "sam_params": {
        "points_per_side": 16,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "crop_n_layers": 0,
        "min_mask_region_area": 500,
    }
}
```

### RePaint 配置

配置文件位于 `external/RePaint-main/confs/`，默认使用 `test_inet256_thick.yml`。

---

## 📈 质量评估

### Mask QA

评估指标：
- ✅ **pass**：是否通过质量检查
- 📏 **mask_ratio**：掩码占比（2%-20% 为优）
- 🔢 **num_regions**：区域数量
- 📐 **edge_diff_mean**：边界连续性

### Fake QA

评估指标：
- ✅ **pass**：是否通过质量检查
- 📊 **diff_mean**：平均差异
- 🎨 **diff_ratio**：差异像素比例
- 🌈 **similarity**：相似度

---

## 📝 依赖项

| 依赖 | 版本 | 用途 |
|------|------|------|
| PyTorch | ≥1.13 | 深度学习框架 |
| OpenCV | ≥4.8.0 | 图像处理 |
| Pillow | ≥10.0.0 | 图像读写 |
| Segment Anything | latest | SAM 模型 |
| NumPy | ≥1.24.0 | 数值计算 |
| PyYAML | ≥6.0 | 配置文件解析 |
| tqdm | ≥4.65.0 | 进度条显示 |

完整依赖列表见 `requirements.txt`。

---


## ⚠️ 注意事项

1. **显存要求**：RePaint 需要较大显存，建议 ≥8GB
2. **磁盘空间**：生成的数据集可能占用大量空间
3. **图像格式**：RePaint 流程仅支持 PNG 格式
4. **Windows 路径**：项目完美支持中文路径

---

**最后更新**：2026-01-31

