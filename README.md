# README of the source data

Our source data includes two parts: 1) the numerical results in the manuscript, which are compiled into an Excel file (Supplementary Tables.xlsx); 2) the code implementation of our framework proposed in the manuscript, which is organized into the directory `./Source Code`.

## Demonstration of supplementary tables

We provide all relevant numerical results from the manuscript compiled into an Excel file (Supplementary Tables.xlsx). Below, we describe the contents of each sheet in detail.

* Chance level: In the contribution analysis, we compute the relative improvement of the decoding performance over chance level.  We estimated the chance level of the classification task by randomly sampling labels according to the class distribution as weights and repeating the process 5000 times. In this sheet, we provide the results of F1 score chance level of the validation set and the accuracy chance level of the test set for each participant.
* Channel contribution (speaking)/Channel contribution (listening): These two sheets provide the detailed results of all channels of each participant in the speaking and listening tasks, respectively. For each task, classification target, participant, and channel, we performed inference 10 times using 10 different random seeds.
* Response latency: This sheet provides the results of the correlation and response latency analysis in the manuscript. For each channel with stable latency, we reported the maximum correlation and corresponding time lags on each syllable.
* Initial-final classification: This sheet provides the initial and final classification results of the four brain decoders (NeuroSketch, ModernTCN, Medformer and MultiResGRU) in the speaking and listening tasks on all participants. We reported the accuracy and F1 score. We performed inference 10 times using 10 different random seeds.
* Epsilon rank ratio: This sheet provides the epsilon rank ratios of each layer of the four brain decoders in the speaking task.
* OOD syllable: This sheet provides the accuracy of in-domain and out-of-domain syllables in participants S11 and S12, whose experimental corpus contained out-of-domain syllables. We performed inference 10 times using 10 different random seeds.
* Beam search: This sheet provides the beam search candidates quality measurements of the four brain decoders in the speaking and listening tasks on all participants. We reported the exact match probability (EMP) and the distribution of the proportion of the syllable error rates (SER). We performed inference 10 times using 10 different random seeds.
* Syllable-to-sentence (ours): This sheet provides the sentence decoding results of our LLM using three-stage post-training and two-stage inference based on Qwen2.5-7B-Instruct on all participants. We reported the CER. We performed inference 10 times using 10 different random seeds.
* Syllable-to-sentence (others): This sheet provides the sentence decoding results of the tested LLMs in the manuscript, including small- and medium-sized LLMs (Qwen2.5-7B-Instruct and Qwen2.5-72B-Instruct) and large commercial LLMs (Qwen-3 Max, Deepseek-v3.2-exp, Doubao-1.6, GPT-5-chat-latest, grok-4-fast and Llama 3.1). We reported the CER. We performed inference 10 times using 10 different random seeds.
* Syllable-to-sentence (ablation): This sheet provides the sentence decoding results of the methods in the ablation study. We reported the CER. We performed inference 10 times using 10 different random seeds.
* Decoding with tone: This sheet provides the results of the tone decoding (accuracy and F1 score) and the beam search candidates quality (EMR and the distribution of the proportion of the SER) using tone information in the speaking and listening tasks. We performed inference 10 times using 10 different random seeds.

## Demonstration of source code

We provide code implementation in directory `./Source Code`. Below, we describe these source files in detail.

## 1. System Requirements

### 1.1 Software Dependencies and Version Requirements

| Dependency                  | Version Requirement    | Purpose                                    |
|-----------------------------|------------------------|--------------------------------------------|
| Python                      | $\geq$3.11             | Runtime Environment                        |
| accelerate                  | $\geq$1.12.0           | LLM Training Acceleration                  |
| einops                      | $\geq$0.8.1            | Tensor Operations                          |
| fire                        | $\geq$0.7.1            | Command Line Tool                          |
| hydra-core                  | $\geq$1.3.2            | Configuration Management                   |
| jieba                       | $\geq$0.42.1           | Chinese Word Segmentation                  |
| matplotlib                  | $\geq$3.10.8           | Visualization                              |
| nltk                        | $\geq$3.9.2            | Natural Language Processing                |
| numpy                       | $\geq$2.4.0            | Numerical Computation                      |
| omegaconf                   | $\geq$2.3.0            | Configuration Processing                   |
| opencc-python-reimplemented | $\geq$0.1.7            | Chinese Simplified-Traditional Conversion  |
| pandas                      | $\geq$2.3.3            | Data Processing                            |
| psutil                      | $\geq$7.2.0            | System Information                         |
| pypinyin                    | $\geq$0.55.0           | Chinese Pinyin Conversion                  |
| python-box                  | $\geq$7.3.2            | Dictionary Operations                      |
| python-levenshtein          | $\geq$0.27.3           | String Similarity                          |
| regex                       | $\geq$2025.11.3        | Regular Expressions                        |
| rouge-chinese               | $\geq$1.0.3            | Chinese ROUGE Score                        |
| scikit-learn                | $\geq$1.8.0            | Machine Learning Algorithms                |
| speechbrain                 | $\geq$1.0.3            | Speech Processing                          |
| timm                        | $\geq$1.0.22           | Vision Model Library                       |
| torch                       | ==2.4.1                | Deep Learning Framework                    |
| torchaudio                  | ==2.4.1                | Audio Processing                           |
| tqdm                        | $\geq$4.67.1           | Progress Bar                               |
| transformers                | $\geq$4.57.3           | Pre-trained Model Library                  |
| pinyin2hanzi                | $\geq$0.1.1            | Chinese Pinyin to Hanzi Conversion         |

### 1.2 Operating System Requirements

- **Recommended Operating System**: Linux
- **Tested Versions**: Ubuntu 22.04 LTS
- **Other Requirements**: 
  - Supports CUDA 11.8+ (for GPU acceleration)
  - Minimum 40GB GPU memory, 80GB+ recommended
  - Minimum 50GB disk space

## 2. Installation Guide

### 2.1 Installation Steps

#### Step 1: Install uv Dependency Management Tool

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (if not already added)
export PATH="$HOME/.cargo/bin:$PATH"
```

#### Step 2: Clone the Repository

#### Step 3: Configure Environment with uv

```bash
# Create virtual environment and install dependencies based on uv.lock
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

#### Step 4: Install LLaMA-Factory Framework

```bash
# Clone LLaMA-Factory to the specified directory
mkdir -p src/llm
cd src/llm
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd ../..

# Install LLaMA-Factory dependencies
uv add -r src/llm/LLaMA-Factory/requirements.txt
```

### 2.2 Installation Duration

On a standard desktop computer (Macbook Air with M4 chip), the typical duration to complete the entire installation process is approximately:

- uv installation: ~1 minute
- Dependency installation: ~10 minutes
- LLaMA-Factory installation: ~5 minutes

Total: ~16 minutes

## 3. Demo

### 3.1 Demo Operation Steps

#### Step 1: Ensure the Environment is Activated

```bash
source .venv/bin/activate
```

#### Step 2: Supplement the folder name in run/conf/dir/local.yaml

#### Step 3: Execute the Demo Script

```bash
# Execute the demo script
bash run/pipeline/llm/training/training_llm_demo.sh
```

### 3.2 Demo Script Description

The `training_llm_demo.sh` script executes the following workflow:

1. Generate translation and listwise training data using demo data
2. Train the translation model
3. Train the listwise model
4. Generate correction training data
5. Train the correction model

### 3.3 Expected Output Results

| Stage                      | Expected Metrics                                             | Result Format                                                |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Data Generation            | Generated training data volume                               | Console output of data counts, data files                    |
| Translation Model Training | Training loss, validation loss, CER| Log files and model checkpoints |
| Listwise Model Training    | Training loss, validation loss, CER | Log files and model checkpoints |
| Correction Model Training  | Training loss, validation loss, CER | Log files and model checkpoints |

### 3.4 Demo Runtime Duration
We utilized an AMD EPYC 7663 56-core processor and a NVIDIA A100 GPU with 80GB of memory. The expected runtime for the demo on this configured server is approximately:

- Data generation: ~2 minutes
- Translation model training: ~30 minutes
- Listwise model training: ~30 minutes
- Correction model training: ~30 minutes

Total: ~92 minutes

## 4. Complete Workflow

1. Data preparation and configuration

  - Supplement the folder name in run/conf/dir/local.yaml.
  - bash `run/pipeline/prepare_data.sh`

2. Calculation of channel contribution (train → infer → summarize)

  - bash `run/pipeline/channel_contribution/training_all_channel_contribution.sh`
  - bash `run/pipeline/channel_contribution/inference_all_channel_contribution.sh`
  - python `run/pipeline/channel_contribution/summarize/channel_results.py`

3. Brain decoders with aggregated high-contribution channels

  - bash `run/pipeline/brain_decoder/training_brain_decoder.sh`
  - bash `run/pipeline/brain_decoder/inference_brain_decoder.sh`

4. LLM three-stage fine-tuning (use LLaMA-Factory[https://github.com/hiyouga/LLaMA-Factory])

  - **Clone [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to: src/llm/LLaMA-Factory**
  - Download the training data in [Google drive](https://drive.google.com/drive/folders/1VJ1h7zHHlWWonCYWWApovxkOxMlX5FVN?usp=sharing).
  - Use the `run/pipeline/llm/training/training_llm.sh` for training LLM models (translation, listwise, correction).

5. Inference and evaluation (two-stage)

  - bash `run/pipeline/llm/inference/inference_brain_stage.sh`          (no-tone classification + beam search)
  - bash `run/pipeline/llm/inference/inference_brain_stage_with_tone.sh` (tone classification + beam search)
  - bash `run/pipeline/llm/inference/inference_baseline.sh`       (IME-style baseline, computes sentence-level CER)
  - bash `run/pipeline/llm/inference/inference_llm_stage.sh`             (two-stage inference, computes sentence-level CER)

6. Analysis
   Rank analysis of the four brain decoders and correlation between speaking/listening time delays: `run/analyze`

Notes:

- All scripts load configurations via Hydra (run/conf). You can override keys from the command line, e.g., dataset=speech_decoding_S1.
- Run scripts from the repository root to ensure relative paths resolve correctly.

## 5. Directory Structure

```
├── run/
│   ├── pipeline/            # Main workflow scripts
│   ├── analyze/             # Analysis scripts
│   ├── conf/                # Hydra configuration files
│   └── prepare_data/        # Data preparation scripts
├── src/
│   ├── data_module/         # Data module
│   ├── model_module/        # Brain decoder model implementations
│   ├── utils/               # Utility functions
│   └── llm/                 # LLM-related code
│       └── LLaMA-Factory/   # LLaMA-Factory framework
├── data/                    # Data directory
├── pyproject.toml           # Project configuration
├── uv.lock                  # Dependency lock file
```

This is a brief overview. For a more detailed file description, please see below.

- run/
  - pipeline/
    - train.py: the code for training brain decoders
    - prepare_data.sh: data configuration and preprocessing
    - channel_contribution/
      - training_all_channel_contribution.sh: train single-channel models
      - inference_all_channel_contribution.sh: run inference for single-channel models
      - summarize_channel_results.py: compute average F1 and write high-contribution channels back to the dataset configuration
    - brain_decoder/
      - training_brain_decoder.sh: train four brain decoders with aggregated high-contribution channels
      - inference_brain_decoder.sh: inference for the brain decoders
    - llm/
      - training/:
        - training_llm.sh: a script to run all procedures
        - data_preprocessing.py: preprocess nlpcc and sighan2015 data for LLM training
        - get_translation_and_listwise_training_data.py: prepare translation and listwise training data
        - get_correction_training_data.py: prepare correction training data
        - training_translation.sh: train translation model with LLaMA-Factory
        - training_listwise.sh: train listwise model with LLaMA-Factory
        - training_correction.sh: train correction model with LLaMA-Factory
      - inference/
        - inference_brain_stage.sh: no-tone classification + beam search
        - inference_brain_stage_with_tone.sh: tone classification + beam search
        - inference_llm_stage.sh: two-stage inference, outputs sentence-level CER
        - inference_baseline.sh: IME-style baseline, outputs sentence-level CER
  - analyze/: rank analysis for the four brain decoders; correlation analysis of speaking/listening time delays
  - conf/: Hydra configurations for project structure, preprocessing, training, inference, and analysis
  - prepare_data/
    - generate_yaml.py: generate dataset configuration (subject ID, number of channels, electrode names, etc.) from raw data and save to run/conf/dataset
    - prepare_data.py: preprocess raw data to be used for training and inference

- src/
  - data_module/: The data_module provides reusable dataset components—train, eval, and test splits—along with a data_collator for batch collation that aligns inputs with model requirements, and a compute_metrics utility for evaluation (e.g., F1, accuracy), offering a standardized interface for training, validation, and testing workflows.
  - model_module/: Implementations of the four brain decoders
  - utils/: Utilities and training helpers
    - aux_func.py: It reports trainable parameter counts, normalizes metric keys by replacing the eval_ prefix with test_, verifies third‑party package availability and versions, and supports batch dynamic imports to automatically register components.
    - summarize_channel_results.py: summarize channel contributions (F1 improvement%), export Excel details and channel list
    - get_callback.py: SWACallback, EMACallback, EarlyStoppingCallback, AveragingCheckpointCallback
    - get_act.py: ReLU, GELU, SiLU, Mish, HardSwish
    - get_optimizer.py: Muon, Adan, AdamW
    - get_scheduler.py: learning rate schedulers (e.g., linear warmup + cosine decay)
    - log.py: experiment logging helpers
    - get_checkpoint_aggregation.py: average multiple checkpoints
    - get_sentence_inference_results.py: aggregate results and compute CER
  - llm/
    - LLaMA-Factory/: clone the LLaMA-Factory repository here

## 6. Configuration Management

All scripts load configurations via Hydra (located in `run/conf/` directory). You can override configuration items from the command line, for example:

```bash
bash run/pipeline/brain_decoder/training_brain_decoder.sh dataset=speech_decoding_S1
```

## 7. Notes

1. All scripts should be run from the repository root to ensure relative paths resolve correctly
2. Ensure all dependencies are installed before first run
3. LLM training requires a significant amount of GPU memory, it is recommended to use a GPU with at least 40GB

## 8. Acknowledgements

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): LLM fine-tuning training framework
- [NeuroSketch](https://github.com/Galaxy-Dawn/NeuroSketch): 2D CNN-based neural decoding model
- [ModernTCN](https://github.com/luodhhh/ModernTCN): 1D CNN-based temporal convolutional network
- [MedFormer](https://github.com/DL4mHealth/Medformer): Attention-based Transformer for medical time-series analysis
- [MultiResGRU](https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/writeups/zinxira-4th-place-solution-a-multilayer-bidirectio): RNN-based GRU architecture

## 9. License

[MIT License](LICENSE)
