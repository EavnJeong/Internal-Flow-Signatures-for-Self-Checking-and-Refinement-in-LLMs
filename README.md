# Internal-Flow-Signatures-for-Self-Checking-and-Refinement-in-LLMs
This repository extracts depthwise internal flow signatures from autoregressive LLM generation, trains a lightweight validator to predict hallucination, and applies a localized refinement that reduces hallucination by intervening once during regeneration.

## What this codebase does
1. Extract flow features from a base LLM while it generates answers on a dataset.
2. Train a validator on the extracted flow to classify hallucination vs non hallucination.
3. Analyze validator performance and task wise behavior.
4. Optionally run refinement baselines and flow guided refinement to reduce hallucination.

## Repository structure
### Entry points
- `main.py`  
  End to end runner for generation, flow extraction, and optional refinement depending on arguments.

- `extract_flow.py`  
  Extracts flow signatures for a dataset and saves per sample traces and feature tensors.

- `train_validator.py`  
  Trains a flow based hallucination validator.

- `analyze_validator.py`  
  Evaluates trained validators, reports metrics, and produces analysis artifacts.

### Core library
- `llm_signature/`
  - `pipeline.py` orchestrates dataset loading, generation, tracing, signature extraction, and saving.
  - `models.py` loads base LLMs and configures inference settings.
  - `generation.py` implements decoding utilities and caching behavior.
  - `tracer.py` hooks the model at a fixed monitored boundary and records depthwise states.
  - `signatures.py` computes flow signatures from traced states.
  - `subspace_accum.py` builds moving readout aligned frames and handles window alignment.
  - `prompt.py` holds prompt templates and task formatting.
  - `utils.py` shared helpers.

### Validator
- `validators/`
  - `dataset.py` loads extracted flow features and labels for training and evaluation.
  - `model.py` defines the validator architecture.
  - `train.py` training loop, logging, checkpointing.
  - `utils.py` metrics, thresholding, helpers.

### Refinement
- `refine/`
  - `find_culprit_events.py` localizes depth and token positions that trigger validator evidence.
  - `intervene_generate.py` performs regeneration with a single localized intervention.
  - `stopping.py` stopping rules and regeneration control.
  - `utils.py` shared refinement utilities.
  - `debug_alignment.py` diagnostics for transported frame alignment and sanity checks.

### Dataset
- `dataset/`
  - `getter.py` dataset factory and routing.
  - `halueval.py` HaluEval loader and formatting.

### Config
- `config/data.json`  
  Dataset paths, split info, and dataset specific settings.

## Quickstart
### 1. Environment
Recommended Python version: 3.10 to 3.12

Install dependencies with your preferred workflow. You will need at least:
- PyTorch
- Transformers
- Accelerate
- numpy
- pandas
- tqdm

If you use external judging, you also need the corresponding API client.

### 2. Configure data
Edit `config/data.json` to point to your dataset locations and any dataset specific options.

### 3. Extract flow
`extract_flow.py` generates answers, optionally obtains hallucination labels from an external judge, traces depthwise states under teacher forcing, builds moving subspaces from top competitors, computes flow signatures, and saves per batch tensors to disk.

The output directory is set automatically to:
`flow/{dataset}/{task}/{model}/`

#### OpenAI API key
When judging is enabled, the script expects an OpenAI API key to be available as an environment variable.

Export the API key before running:
```bash
export OPENAI_API_KEY=your_api_key_here

python extract_flow.py \
  --dataset halueval \
  --task qa \
  --model qwen25 \
  --batch_size 4 \
  --max_new_tokens 128 \
  --device_map auto
```

#### 4. Train a validator
`train_validator.py` trains `FlowGRUValidator` on extracted flow tensors under `flow/{dataset}/{task}/{model}/`. It automatically estimates the positive class weight from the training set and optimizes a weighted `BCEWithLogitsLoss`. The best checkpoint is tracked by test AUROC and saved to `validator_ckpt_{wandb_run_name}.pt`.

Basic usage:
```bash
python train_validator.py \
  --flow_root flow \
  --dataset halueval \
  --task qa \
  --model qwen25 \
  --device cuda:0 \
  --batch_size 16 \
  --epochs 300 \
  --lr 3e-5 \
  --weight_decay 1e-2 \
  --grad_clip 1.0 \
  --pool max \
  --wandb_run_name qwen25_qa
```

#### 5. Analyze validator statistics
`analyze_validator_stats.py` loads a trained `FlowGRUValidator` checkpoint, runs inference on a chosen split, and summarizes which feature groups contribute to predictions. It writes a `report.json` and saves diagnostic figures under `--out_dir`.

This script supports:
- Confusion mode breakdown (tp, fp, tn, fn)
- Grad x Input group mass statistics
- Optional group occlusion deltas on logits
- Depth shape curves and heatmaps
- Hotspot depth and token histograms

Basic usage:
```bash
python analyze_validator_stats.py \
  --ckpt_path validator_ckpt_qwen25_qa.pt \
  --flow_root flow \
  --dataset halueval \
  --task qa \
  --model qwen25 \
  --split test \
  --batch_size 8 \
  --device cuda:0 \
  --out_dir validator_report
```

