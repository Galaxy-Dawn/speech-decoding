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

## Overview

This repository contains the codebase for **Towards Unified Brain-to-Text Decoding Across Speech Production and Perception**.
It organizes the full experimental workflow from neural data preprocessing to sentence-level decoding and analysis, with shared infrastructure across production and perception settings.

The repository includes:

- **Data preparation** for subject-specific speech-decoding datasets
- **Channel contribution analysis** for selecting informative electrodes
- **Four brain decoders**: `NeuroSketch`, `ModernTCN`, `MedFormer`, and `MultiResGRU`
- **LLM-based syllable-to-sentence reconstruction** built on a vendored copy of `LLaMA-Factory`
- **Post-hoc analysis scripts** for rank analysis and response-latency analysis

This README is written as a code-first guide for reproducing, understanding, and extending the repository.

## Highlights

- **Unified workflow across speech production and perception**
- **End-to-end pipeline** from raw neural data preprocessing to sentence-level decoding
- **Multiple decoder backbones** for controlled model comparison
- **Hydra-based configuration** for training, inference, and analysis
- **LLM training/inference pipeline** under `run/pipeline/llm/`
- **Included demo corpus** for the text-side post-training workflow: `data/demo_corpus_for_post_training.json`
- **Anonymized subject IDs** (`S1`-`S12`) in the released code

## Quick Navigation

| Section | What it covers |
| --- | --- |
| [Repository Layout](#repository-layout) | Main folders and responsibilities |
| [Requirements](#requirements) | Software and hardware assumptions |
| [Installation](#installation) | Environment setup with `uv` |
| [Before You Run Anything](#before-you-run-anything) | Path assumptions and local config |
| [Quick Start](#quick-start) | Minimal runnable entry points |
| [Full Pipeline](#full-pipeline) | End-to-end workflow |
| [Configuration](#configuration) | Hydra overrides and config locations |
| [Data Notes](#data-notes) | What is and is not included |

## Repository Layout

```text
.
├── data/
│   └── demo_corpus_for_post_training.json   # demo text corpus for the LLM stage
├── run/
│   ├── analyze/                             # analysis scripts
│   ├── conf/                                # Hydra configs
│   ├── pipeline/                            # bash/python pipeline entrypoints
│   │   ├── brain_decoder/
│   │   ├── channel_contribution/
│   │   └── llm/
│   ├── prepare_data/                        # preprocessing scripts
│   └── train.py                             # main training/inference entrypoint
├── src/
│   ├── data_module/                         # dataset, collator, metrics, augmentation
│   ├── model_module/brain_decoder/          # model implementations
│   ├── utils/                               # training and logging helpers
│   └── llm/LLaMA-Factory/                   # vendored LLaMA-Factory snapshot
├── pyproject.toml
└── uv.lock
```

## Requirements

### Software

- Python **>= 3.11**
- PyTorch **2.4.1**
- Hydra **1.3.2**
- `uv` for dependency management

Dependencies are pinned in `pyproject.toml` and `uv.lock`.

### Hardware

Recommended environment:

- Linux (tested on Ubuntu 22.04 style environments)
- CUDA-capable GPU for model training/inference
- LLM training typically benefits from **40 GB+ GPU memory**
- Multi-GPU setups are assumed by several batch scripts

## Installation

```bash
git clone https://github.com/Galaxy-Dawn/speech-decoding.git
cd speech-decoding
uv sync
source .venv/bin/activate
```

## Before You Run Anything

### 1. Configure local paths

Edit:

- `run/conf/dir/local.yaml`

At minimum, set the directories for your local environment:

- `data_dir`
- `processed_dir`
- `logging_dir`
- `model_save_dir`
- `llm_save_dir`
- `llm_utils_dir`
- `llm_inference_result_dir`
- `analyze_result_dir`

### 2. Check the project-root assumption

Several shell scripts currently hardcode:

```bash
PROJECT_DIR='Speech_Decoding'
```

If your checkout directory is **not** named `Speech_Decoding`, choose one of the following:

- rename the local folder to `Speech_Decoding`, or
- edit the `PROJECT_DIR` variable in the scripts under `run/pipeline/**`, and align `run/conf/dir/local.yaml`

Without this step, the bash entrypoints may fail even if the Python environment is correct.

## Quick Start

### Option A: Prepare one subject

```bash
bash run/pipeline/prepare_data.sh subject_id=S1
```

This generates subject-specific dataset config and processed data for the selected subject.

### Option B: Run the LLM demo workflow

```bash
bash run/pipeline/llm/training/training_llm_demo.sh
```

This demo uses the included text-side demo corpus and runs:

1. translation/listwise training-data generation
2. translation-stage training
3. listwise-stage training
4. correction-data generation
5. correction-stage training

### Option C: Run a single Hydra command directly

```bash
python run/train.py \
  training.do_train=True \
  training.do_predict=True \
  model=NeuroSketch \
  dataset=speech_decoding_S1 \
  dataset.id=S1 \
  dataset.task=speaking_initial
```

This is often the easiest way to debug one experiment before launching the larger bash pipelines.

## Full Pipeline

### 1. Data preparation

```bash
bash run/pipeline/prepare_data.sh
# or
bash run/pipeline/prepare_data.sh subject_id=S1,S2
```

### 2. Channel contribution analysis

Train per-channel models:

```bash
bash run/pipeline/channel_contribution/training_all_channel_contribution.sh
```

Run inference and summarize channel scores:

```bash
bash run/pipeline/channel_contribution/inference_all_channel_contribution.sh
python run/pipeline/channel_contribution/summarize_channel_results.py
```

### 3. Brain decoder training and inference

Train the four brain decoders using selected channels:

```bash
bash run/pipeline/brain_decoder/training_brain_decoder.sh
bash run/pipeline/brain_decoder/inference_brain_decoder.sh
```

Supported decoders:

- `NeuroSketch`
- `ModernTCN`
- `MedFormer`
- `MultiResGRU`

### 4. LLM post-training

```bash
bash run/pipeline/llm/training/training_llm.sh
```

### 5. LLM inference

```bash
bash run/pipeline/llm/inference/inference_brain_stage.sh
bash run/pipeline/llm/inference/inference_brain_stage_with_tone.sh
bash run/pipeline/llm/inference/inference_baseline.sh
bash run/pipeline/llm/inference/inference_llm_stage.sh
```

### 6. Analysis

```bash
python run/analyze/analyze_rank.py
python run/analyze/response_latency_analysis.py
```

## Configuration

This project uses **Hydra** for experiment management.

Key config directories:

- `run/conf/train.yaml`
- `run/conf/analyze.yaml`
- `run/conf/prepare_data.yaml`
- `run/conf/llm_training.yaml`
- `run/conf/llm_inference.yaml`
- `run/conf/brain_decoder/*.yaml`
- `run/conf/dataset/*.yaml`
- `run/conf/dir/local.yaml`

Example override patterns:

```bash
python run/train.py model=ModernTCN dataset=speech_decoding_S3 dataset.id=S3
python run/train.py dataset.task=listening_final training.num_train_epochs=100
```

## Data Notes

- **Raw sEEG data is not bundled** in this repository.
- The repository includes only a small **demo text corpus** at `data/demo_corpus_for_post_training.json`.
- Subject identifiers in the released code are anonymized as `S1`-`S12`.
- Generated experiment outputs under `src/llm/LLaMA-Factory/outputs/` are ignored from version control.

## Acknowledgements

This repository builds on or references the following projects:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [NeuroSketch](https://github.com/Galaxy-Dawn/NeuroSketch)
- [ModernTCN](https://github.com/luodhhh/ModernTCN)
- [MedFormer](https://github.com/DL4mHealth/Medformer)
- [MultiResGRU](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/writeups/zinxira-4th-place-solution-a-multilayer-bidirectio)

## Citation

If this repository is useful in your research, please cite the corresponding paper or project page when it is publicly available.
