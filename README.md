# BirdCLEF · 鸟类鸣声智能识别系统

面向《语音识别》课程设计的鸟类鸣声分类项目。系统覆盖**数据预处理、多维声学特征提取、传统机器学习与深度学习建模、Stacking 二级集成、模块缝合网络（StitchedFusionNet）** 以及 **PySide6 图形界面推理**，支持 10 类与 37 类闭集物种识别。

完整实验分析与课程知识对照见语音识别.pdf。

---

## 主要特性

- **多模态声学特征**：MFCC、Mel 谱图、能量/音色/基频/节奏等 588 维手工描述子
- **多模型对比**：KNN、SVM、随机森林、CNN、CRNN、Transformer、Mamba、Hybrid、Stitch
- **两种改进路线**
  - **Stacking 集成**：深度模型概率 + SVM 概率 → Logistic Regression 元学习器
  - **StitchedFusionNet**：CNN 全局支路 + CRNN 注意力支路 + Mamba 时序支路，门控自适应融合
- **可视化与 GUI**：实验对比图、混淆矩阵、训练曲线；支持本地音频文件识别与麦克风录音
- **可复现实验**：YAML 配置驱动，结果写入 `comparison_results.json`

---

## 实验结果（验证集）

| 设定 | 最优方案 | Top-1 准确率 | Top-5 准确率 |
|------|----------|-------------|-------------|
| 10 类快速实验 | Stacking（CNN + SVM） | **85.5%** | **98.8%** |
| 37 类完整实验 | Stacking（Hybrid + SVM） | **82.7%** | **97.4%** |

> 权重文件与图表未纳入 Git 仓库，需本地训练或自行导出部署包后使用 GUI。

---

## 目录结构

本项目代码位于 `bird_recognition/`，**数据与训练产物默认放在其上一级工作区目录**（脚本内部以 `bird_recognition` 的父目录作为 `project_root`）。

```
语音识别大作业/                  # 工作区根目录（project_root）
├── train.csv                    # 训练元数据（需自行准备）
├── taxonomy.csv                 # 物种分类信息（需自行准备）
├── train_audio/                 # 音频文件目录（需自行准备）
├── outputs/                     # 训练产物（自动生成，已被 .gitignore 排除）
│   ├── cache/                   # Mel + 手工特征缓存
│   ├── features/                # ML 特征矩阵
│   ├── figures/                 # 实验图表
│   ├── models/                  # 权重、指标、部署包
│   └── comparison_results.json
└── bird_recognition/            # 本仓库
    ├── configs/                 # 实验配置
    ├── docs/                    # 项目报告
    ├── gui/                     # PySide6 图形界面
    ├── scripts/                 # 训练、对比、导出、可视化脚本
    ├── src/                     # 核心源码
    ├── requirements.txt
    └── README.md
```

### 核心源码说明

| 路径 | 说明 |
|------|------|
| `src/features/extractor.py` | 声学特征提取（MFCC、Mel、F0、节奏等） |
| `src/models/architectures.py` | CNN / CRNN / Transformer / Mamba / Hybrid / Stitch |
| `src/training/ensemble.py` | Stacking 元学习器 |
| `src/inference/stacking_predictor.py` | Stacking 推理接口 |
| `gui/main_window.py` | GUI 主窗口 |

---

## 环境配置

### 1. 创建环境

```bash
conda create -n bird-asr python=3.10 -y
conda activate bird-asr
```

### 2. 安装依赖

```bash
cd bird_recognition
pip install -r requirements.txt
```

主要依赖：`torch`、`librosa`、`scikit-learn`、`pandas`、`matplotlib`、`PySide6`、`sounddevice`。

### 3. 准备数据

将 BirdCLEF 风格数据集放入**工作区根目录**（与 `bird_recognition/` 同级）：

- `train.csv`
- `taxonomy.csv`
- `train_audio/`

若希望数据放在其他位置，可修改 `configs/*.yaml` 中的 `paths.data_root` 及相关路径。

---

## 快速开始

以下命令均在**工作区根目录**执行（即 `bird_recognition` 的上一级）：

```bash
cd /d/语音识别大作业
conda activate bird-asr
```

### 1. 快速对比实验（10 类，约 30 分钟）

```bash
python bird_recognition/scripts/compare_all.py \
  --config bird_recognition/configs/compare_fast.yaml \
  --skip-cv
```

### 2. 完整对比实验（37 类）

```bash
python bird_recognition/scripts/compare_all.py \
  --config bird_recognition/configs/compare.yaml \
  --skip-cv
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--skip-cv` | 跳过交叉验证，直接训练（更快） |
| `--skip-dl` | 仅运行传统 ML |
| `--skip-ml` | 仅运行深度学习 |
| `--dl-only cnn` | 只训练指定 DL 模型 |
| `--stacking-only` | 仅运行 Stacking（需基模型已训练） |

### 3. 单独训练模型

```bash
# 深度学习
python bird_recognition/scripts/train_dl.py \
  --config bird_recognition/configs/compare.yaml \
  --model hybrid

# 传统机器学习
python bird_recognition/scripts/train_ml.py \
  --config bird_recognition/configs/compare.yaml
```

### 4. 导出推理部署包

训练完成并生成 `outputs/comparison_results.json` 后：

```bash
# Stacking 部署包（推荐，GUI 默认使用）
python bird_recognition/scripts/export_stacking_bundle.py \
  --config bird_recognition/configs/compare.yaml

# Stitch 部署包
python bird_recognition/scripts/export_stitch_bundle.py \
  --config bird_recognition/configs/compare.yaml
```

部署包输出路径：`outputs/models/deploy/stacking_top37/`（含 `hybrid.pt`、`svm.joblib`、`stacking.joblib`、`manifest.json`）。

### 5. 结果可视化

```bash
python bird_recognition/scripts/visualize_results.py \
  --config bird_recognition/configs/compare.yaml
```

图表保存至 `outputs/figures/`。

### 6. 启动 GUI

```bash
python bird_recognition/scripts/run_gui.py
```

可选指定部署包：

```bash
# 37 类 Stacking（默认）
python bird_recognition/scripts/run_gui.py --bundle stacking_top37

# 37 类 Stitch
python bird_recognition/scripts/run_gui.py --bundle stitch_top37

# 10 类 Stacking
python bird_recognition/scripts/run_gui.py --bundle stacking_top10
```

GUI 功能：拖拽/选择音频、麦克风录音、Top-5 识别结果展示、Mel 谱图可视化。

---

## 配置文件

| 文件 | 用途 |
|------|------|
| `configs/compare_fast.yaml` | **推荐入门**：10 类、快速训练 |
| `configs/compare.yaml` | 完整实验：37 类、50 epoch、完整增强 |
| `configs/default.yaml` | 基础训练默认参数 |

可在 YAML 中调整：`top_n_species`（类别数）、`epochs`、`batch_size`、`min_samples_per_class` 等。

---

## 模型架构概览

```
音频 (32 kHz, 5 s)
    │
    ├─► 手工特征 (588 维) ──► KNN / SVM / RF
    │                              │
    └─► Mel 谱图 (128 × T) ──► CNN / CRNN / Transformer / Mamba
                                      │
                               Hybrid / Stitch
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            Stacking (DL prob + ML prob)        StitchedFusionNet
                    │                          (三支路门控缝合)
                    ▼
              物种分类输出
```

---

## 注意事项

1. **GPU 推荐**：深度学习训练建议使用 CUDA；无 GPU 时可在配置中将 `device` 改为 `cpu`。
2. **首次运行较慢**：需预计算特征缓存（`outputs/cache/`），37 类完整实验首次约 15–20 分钟。
3. **Git 仓库不含大文件**：模型权重（`.pt`）、音频数据、`outputs/` 等已通过 `.gitignore` 排除，克隆后需自行训练或导入。
4. **Windows 中文路径**：项目已包含 `scripts/setup_console.py` 处理控制台 UTF-8 输出。

---

## 许可证与引用

本项目为课程大作业，数据来源于 BirdCLEF 风格鸟鸣录音集合。若使用本项目代码或思路，请注明出处。

---

## 作者

骆俊杰 · 语音识别课程大作业
