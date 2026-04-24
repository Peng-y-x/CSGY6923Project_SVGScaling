# CS-GY 6923 Optional Project (Spring 2026)

This repository is the working codebase for the SVG language model scaling project.

## 1. Project Scope

Goal: train decoder-only Transformer LMs on SVG code, study scaling laws, compare standard parameterization vs. muP, then generate and evaluate SVG samples.

Course project deliverables:
- PDF report (recommended 6-10 pages, excluding references/appendix)
- Code repository with scripts + README

## 2. Repository Layout

```text
configs/        # experiment configs (data, model sizes, LR sweep, muP, generation)
scripts/        # runnable entrypoints
src/
  data/         # download, cleaning, validation, split, push to HF
  tokenization/ # tokenizer training + encoding
  models/       # transformer definitions (standard + muP)
  train/        # trainer, optimizer/scheduler, metrics
  eval/         # perplexity + XML/render/structural validity
  generation/   # unconditional and prefix-conditioned generation
  analysis/     # scaling fit + plots
notebooks/      # optional analysis notebooks (non-critical path)
outputs/        # figures/tables/samples generated during experiments
report/         # report source and report figures
```

## 3. Environment

Recommended: Python 3.10+ and PyTorch with CUDA for A100.

Install:

```bash
pip install -r requirements.txt
```

If you use Colab Pro:
- Runtime: GPU (A100)
- Clone this repo in Colab
- Install dependencies
- Authenticate Hugging Face when pushing/loading datasets or models

Recommended startup commands for every fresh Colab session:

```bash
# 1) System packages required by CairoSVG render checks
apt-get update -y
apt-get install -y libcairo2 libcairo2-dev libffi-dev

# 2) Python dependencies
pip install -U pip
pip install -r requirements.txt

# 3) (Optional) login if you need HF push
huggingface-cli login

# 4) Quick sanity check for render validation
python - <<'PY'
from src.data.validate_svg import validate_render
svg = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24"><circle cx="12" cy="12" r="6"/></svg>'
ok, err = validate_render(svg)
print("render_check_ok:", ok)
print("render_check_err:", err)
PY
```

If your Hugging Face token is stored in Colab Keys, scripts can auto-load it.
Default config:
- `hf_auth.auto_load_colab_key: true`
- `hf_auth.colab_key_name: HF_TOKEN`

## 4. Data Strategy (Recommended)

Use Hugging Face Datasets as the storage backend for cleaned/split datasets.

Why:
- Stable versioning for reproducibility
- Fast loading in Colab via `load_dataset(...)`
- Shared canonical dataset across LR sweep and all model-size runs

Recommended version tags:
- `v1-clean-rawsplit`: cleaned SVG text with train/val/test split
- `v2-tokenized`: tokenized dataset for training

Keep all dataset construction scripts in this repo so results can be reproduced from source datasets.

## 5. End-to-End Execution Plan

Run in this order:

1. Data preprocessing
```bash
python scripts/run_preprocess.py --config configs/data.yaml
```
This step enforces `targets.min_train_tokens_estimate` from config. If train token estimate is below target, preprocessing exits with an error so you can increase sampled data.
This step also writes `manifest.json` and reuses existing processed outputs when the config hash matches (use `--force` to rebuild).

2. Tokenizer training + encoding
```bash
python scripts/run_tokenizer.py --config configs/data.yaml
```

2.1 Push tokenizer artifacts to HF model repo (optional but recommended)
```bash
python scripts/push_tokenizer_to_hf.py --config configs/data.yaml
```

2.2 Push tokenized dataset to HF dataset repo (for direct training input_ids)
```bash
python scripts/push_tokenized_dataset_to_hf.py --config configs/data.yaml
```

3. Part 2: smallest-model LR sweep (standard parameterization)
```bash
python scripts/run_lr_sweep.py --config configs/sweep_lr.yaml
```

4. Part 2: 5 model scales, 1 epoch each
```bash
python scripts/run_train.py --config configs/train_tiny.yaml
python scripts/run_train.py --config configs/train_small.yaml
python scripts/run_train.py --config configs/train_medium.yaml
python scripts/run_train.py --config configs/train_large.yaml
python scripts/run_train.py --config configs/train_xl.yaml
```

5. Part 2 scaling-law fit and plots
```bash
python scripts/fit_scaling_law.py
```

6. Part 3: muP LR sweep + muP scaling runs
```bash
python scripts/run_mup_train.py --config configs/mup_tiny.yaml
```

7. Part 4: best-model generation + evaluation
```bash
python scripts/run_generate.py --config configs/generation.yaml
python scripts/run_eval.py
```

## 6. Expected Artifacts

- `outputs/figures/`
  - scaling curve (standard)
  - scaling curve (standard vs. muP)
  - LR sweep plots
  - training loss curves
  - generated SVG render grids
- `outputs/tables/`
  - architecture table
  - training throughput/memory/time table
  - evaluation metrics table
- `outputs/samples/`
  - raw generated SVG files
  - rendered PNGs

## 7. Minimum Report Checklist

- Data pipeline and stats (token counts, split sizes, length histogram, filtering before/after)
- LR sweep setup/results (smallest model)
- Scaling plot + power-law fit (`L = a * N^{-alpha} + c`)
- Standard vs. muP comparison
- 10x parameter extrapolation with uncertainty discussion
- Best-model sample generation and quantitative validity metrics
- Design decisions, failures, limitations, and next steps

## 8. Notes for Current Status

Implemented now:
1. `scripts/run_preprocess.py` end-to-end pipeline:
- dataset download
- SVG cleaning
- XML validation
- render validation (CairoSVG)
- deduplication by SVG hash
- file-aware split (`by_file`)
- split export (`train/validation/test` JSONL)
- stats + histograms + complexity examples (`.svg` and `.png`)
2. `scripts/run_tokenizer.py`:
- BPE tokenizer training
- split encoding
- vocab size + token totals + sequence length histograms
3. HF publishing helpers:
- `scripts/push_tokenizer_to_hf.py`
- `scripts/push_tokenized_dataset_to_hf.py`

Still pending:
1. model training pipeline in `src/train/*` + `scripts/run_train.py`
2. LR sweep automation and result logging
3. scaling-law fit scripts wired to actual run outputs
