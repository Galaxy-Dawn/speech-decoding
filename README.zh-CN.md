<div align="center">

# Towards Unified Brain-to-Text Decoding Across Speech Production and Perception

<p>
  <a href="https://github.com/Galaxy-Dawn/speech-decoding/stargazers"><img src="https://img.shields.io/github/stars/Galaxy-Dawn/speech-decoding?style=flat-square&color=yellow" alt="Stars"/></a>
  <a href="https://github.com/Galaxy-Dawn/speech-decoding/network/members"><img src="https://img.shields.io/github/forks/Galaxy-Dawn/speech-decoding?style=flat-square" alt="Forks"/></a>
  <img src="https://img.shields.io/github/last-commit/Galaxy-Dawn/speech-decoding?style=flat-square" alt="Last Commit"/>
  <img src="https://img.shields.io/github/discussions/Galaxy-Dawn/speech-decoding?style=flat-square" alt="Discussions"/>
</p>

<strong>Language</strong>: <a href="README.md">English</a> | <a href="README.zh-CN.md">中文</a>

</div>

## News

- **2026-03-13**：我们将论文 **Towards Unified Brain-to-Text Decoding Across Speech Production and Perception** 的代码开源了。

## 项目简介

本仓库是论文 **Towards Unified Brain-to-Text Decoding Across Speech Production and Perception** 的代码实现。
它组织了从神经数据预处理到句子级解码与分析的完整实验流程，并在 speech production 与 speech perception 两种场景下复用统一的基础设施。

仓库主要包括：

- **数据预处理**：面向被试级 speech-decoding 数据集的处理流程
- **通道贡献分析**：用于筛选信息量更高的电极通道
- **四种脑解码模型**：`NeuroSketch`、`ModernTCN`、`MedFormer`、`MultiResGRU`
- **基于 LLM 的 syllable-to-sentence 重建**：依赖仓库内置的 `LLaMA-Factory` 快照
- **后处理分析脚本**：包括 rank analysis 与 response latency analysis

这份 README 以“代码仓库使用说明”为主，方便你复现、理解和扩展该项目。

## 项目亮点

- **统一支持 speech production 与 speech perception 的工作流**
- **端到端流程**：从原始神经数据预处理到句子级解码
- **多种 decoder backbone**，便于做受控比较实验
- **基于 Hydra 的配置管理**，统一训练、推理与分析入口
- **完整的 LLM 训练/推理流程**，位于 `run/pipeline/llm/`
- **附带 demo 文本语料**：`data/demo_corpus_for_post_training.json`
- **公开版本中已使用匿名化被试编号**：`S1`-`S12`

## 快速导航

| 章节 | 说明 |
| --- | --- |
| [仓库结构](#仓库结构) | 主要目录及职责 |
| [运行要求](#运行要求) | 软件与硬件要求 |
| [安装](#安装) | 基于 `uv` 的环境配置 |
| [运行前准备](#运行前准备) | 本地路径与目录假设 |
| [快速开始](#快速开始) | 最小可运行入口 |
| [完整流程](#完整流程) | 端到端实验流程 |
| [配置说明](#配置说明) | Hydra 配置与覆盖方式 |
| [数据说明](#数据说明) | 仓库包含与不包含的数据 |

## 仓库结构

```text
.
├── data/
│   └── demo_corpus_for_post_training.json   # LLM 阶段使用的 demo 文本语料
├── run/
│   ├── analyze/                             # 分析脚本
│   ├── conf/                                # Hydra 配置
│   ├── pipeline/                            # bash/python 流水线入口
│   │   ├── brain_decoder/
│   │   ├── channel_contribution/
│   │   └── llm/
│   ├── prepare_data/                        # 预处理脚本
│   └── train.py                             # 主训练/推理入口
├── src/
│   ├── data_module/                         # dataset、collator、metrics、augmentation
│   ├── model_module/brain_decoder/          # 模型实现
│   ├── utils/                               # 训练与日志辅助工具
│   └── llm/LLaMA-Factory/                   # vendored LLaMA-Factory 快照
├── pyproject.toml
└── uv.lock
```

## 运行要求

### 软件要求

- Python **>= 3.11**
- PyTorch **2.4.1**
- Hydra **1.3.2**
- 使用 `uv` 进行依赖管理

依赖版本以 `pyproject.toml` 与 `uv.lock` 为准。

### 硬件要求

推荐环境：

- Linux（项目更接近 Ubuntu 22.04 风格环境）
- 支持 CUDA 的 GPU
- LLM 训练通常建议 **40 GB 及以上显存**
- 部分批量脚本默认假设多卡环境

## 安装

```bash
git clone https://github.com/Galaxy-Dawn/speech-decoding.git
cd speech-decoding
uv sync
source .venv/bin/activate
```

## 运行前准备

### 1. 配置本地路径

编辑：

- `run/conf/dir/local.yaml`

至少需要根据本地环境设置这些目录：

- `data_dir`
- `processed_dir`
- `logging_dir`
- `model_save_dir`
- `llm_save_dir`
- `llm_utils_dir`
- `llm_inference_result_dir`
- `analyze_result_dir`

### 2. 检查项目根目录假设

多个 shell 脚本里当前写死了：

```bash
PROJECT_DIR='Speech_Decoding'
```

如果你的本地仓库目录名**不是** `Speech_Decoding`，请二选一：

- 直接把本地文件夹改名为 `Speech_Decoding`，或
- 修改 `run/pipeline/**` 下脚本里的 `PROJECT_DIR`，并同步调整 `run/conf/dir/local.yaml`

否则即便 Python 环境没问题，bash 入口脚本也可能因为路径假设失败。

## 快速开始

### 方案 A：准备单个被试的数据

```bash
bash run/pipeline/prepare_data.sh subject_id=S1
```

该命令会为指定被试生成数据配置，并执行对应的预处理流程。

### 方案 B：运行 LLM demo 流程

```bash
bash run/pipeline/llm/training/training_llm_demo.sh
```

这个 demo 会使用仓库自带的文本侧 demo 语料，并执行：

1. translation/listwise 训练数据生成
2. translation 阶段训练
3. listwise 阶段训练
4. correction 数据生成
5. correction 阶段训练

### 方案 C：直接运行单条 Hydra 命令

```bash
python run/train.py \
  training.do_train=True \
  training.do_predict=True \
  model=NeuroSketch \
  dataset=speech_decoding_S1 \
  dataset.id=S1 \
  dataset.task=speaking_initial
```

在调试单个实验时，这通常比直接运行大批量 bash 流程更方便。

## 完整流程

### 1. 数据预处理

```bash
bash run/pipeline/prepare_data.sh
# 或
bash run/pipeline/prepare_data.sh subject_id=S1,S2
```

### 2. 通道贡献分析

训练单通道模型：

```bash
bash run/pipeline/channel_contribution/training_all_channel_contribution.sh
```

推理并汇总通道分数：

```bash
bash run/pipeline/channel_contribution/inference_all_channel_contribution.sh
python run/pipeline/channel_contribution/summarize_channel_results.py
```

### 3. 脑解码模型训练与推理

使用筛选后的通道训练四种 brain decoder：

```bash
bash run/pipeline/brain_decoder/training_brain_decoder.sh
bash run/pipeline/brain_decoder/inference_brain_decoder.sh
```

支持的模型包括：

- `NeuroSketch`
- `ModernTCN`
- `MedFormer`
- `MultiResGRU`

### 4. LLM 后训练

```bash
bash run/pipeline/llm/training/training_llm.sh
```

### 5. LLM 推理

```bash
bash run/pipeline/llm/inference/inference_brain_stage.sh
bash run/pipeline/llm/inference/inference_brain_stage_with_tone.sh
bash run/pipeline/llm/inference/inference_baseline.sh
bash run/pipeline/llm/inference/inference_llm_stage.sh
```

### 6. 分析

```bash
python run/analyze/analyze_rank.py
python run/analyze/response_latency_analysis.py
```

## 配置说明

本项目使用 **Hydra** 管理实验配置。

关键配置目录：

- `run/conf/train.yaml`
- `run/conf/analyze.yaml`
- `run/conf/prepare_data.yaml`
- `run/conf/llm_training.yaml`
- `run/conf/llm_inference.yaml`
- `run/conf/brain_decoder/*.yaml`
- `run/conf/dataset/*.yaml`
- `run/conf/dir/local.yaml`

常见 override 示例：

```bash
python run/train.py model=ModernTCN dataset=speech_decoding_S3 dataset.id=S3
python run/train.py dataset.task=listening_final training.num_train_epochs=100
```

## 数据说明

- **原始 sEEG 数据并未包含在本仓库中**
- 仓库中只提供了一个小型 **demo 文本语料**：`data/demo_corpus_for_post_training.json`
- 公开代码中的被试编号已经匿名化为 `S1`-`S12`
- `src/llm/LLaMA-Factory/outputs/` 下的生成结果已加入版本控制忽略

## 致谢

本仓库构建或参考了以下项目：

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [NeuroSketch](https://github.com/Galaxy-Dawn/NeuroSketch)
- [ModernTCN](https://github.com/luodhhh/ModernTCN)
- [MedFormer](https://github.com/DL4mHealth/Medformer)
- [MultiResGRU](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/writeups/zinxira-4th-place-solution-a-multilayer-bidirectio)

## 引用

如果这个仓库对你的研究有帮助，请在论文或项目页面公开后引用对应成果。
