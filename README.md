# SVG Scaling Laws Optional Project

This repository contains the code for training decoder-only Transformer language models on SVG code, fitting scaling laws, comparing standard parameterization with muP, and evaluating generated SVG samples.

## Code Structure

```text
configs/
  data.yaml                              # dataset cleaning, tokenizer, HF repo settings
  sweep_lr.yaml                          # Part 2 standard Tiny LR sweep
  train_{tiny,small,medium,large,xl}.yaml # Part 2 standard model configs
  mup_sweep_lr.yaml                      # Part 3 muP Tiny LR sweep
  mup_{tiny,small,medium,large,xl}.yaml  # Part 3 muP model configs
  part4_best.yaml                        # Part 4 continued best-model training
  part4_generation_temperature_groups.yaml # final Part 4 generation rerun

scripts/
  run_preprocess.py                      # download, clean, validate, split SVGs
  run_tokenizer.py                       # train BPE tokenizer and encode splits
  push_tokenizer_to_hf.py                # upload tokenizer artifacts
  push_tokenized_dataset_to_hf.py        # upload tokenized dataset
  run_lr_sweep.py / run_part2_all.py     # Part 2 LR sweep and standard scaling runs
  run_mup_lr_sweep.py / run_part3_mup_all.py # Part 3 muP sweep and scaling runs
  fit_scaling_law.py / fit_part3_scaling.py  # scaling-law fits and extrapolation
  run_part4_train.py                     # continue training the best model
  run_generate.py / run_eval.py          # generation and XML/render/structural evaluation
  check_tokenizer_alignment.py           # tokenizer sanity check for generation

src/
  data/          # SVG cleaning, validation, split, HF utilities
  tokenization/  # BPE tokenizer training and dataset encoding
  models/        # standard Transformer and muP Transformer
  train/         # trainers, optimizer, scheduler, metrics, checkpoints
  eval/          # perplexity, XML validity, SVG structural/render checks
  generation/    # sampling and rendering helpers
  analysis/      # scaling-law fitting utilities

colab_train_part*.ipynb                  # Colab execution notebooks used for the experiments
```

## Environment Setup

Python 3.10+ is recommended. For full training, use a CUDA GPU; the experiments were run on Colab A100 with `bf16` autocast.

```bash
pip install -U pip
pip install -r requirements.txt
```

For Colab or Linux environments using CairoSVG render validation:

```bash
apt-get update -y
apt-get install -y libcairo2 libcairo2-dev libffi-dev
pip install -r requirements.txt
```

If pushing datasets or tokenizer artifacts to Hugging Face:

```bash
huggingface-cli login
```

## Data and Tokenizer

Build the cleaned SVG dataset and train/encode the tokenizer:

```bash
python scripts/run_preprocess.py --config configs/data.yaml
python scripts/run_tokenizer.py --config configs/data.yaml
```

Optional upload commands:

```bash
python scripts/push_tokenizer_to_hf.py --config configs/data.yaml
python scripts/push_tokenized_dataset_to_hf.py --config configs/data.yaml
```

The training configs can load the tokenized Hugging Face dataset directly, so if the dataset is already available on Hugging Face, later training runs do not need to rebuild local preprocessing artifacts.

## Reproduce Experiments

### Part 2: Standard Parameterization Scaling

Run the Tiny learning-rate sweep:

```bash
python scripts/run_lr_sweep.py --config configs/sweep_lr.yaml
```

Train all five model sizes with the selected Tiny LR:

```bash
python scripts/run_part2_all.py --best-lr-json outputs/part2_lr_sweep_v2/best_lr.json
```

Fit the standard scaling curve:

```bash
python scripts/fit_scaling_law.py
```

### Part 3: muP Scaling and Extrapolation

Run the muP Tiny learning-rate sweep:

```bash
python scripts/run_mup_lr_sweep.py --config configs/mup_sweep_lr.yaml
```

Train all five muP model sizes with the selected Tiny LR:

```bash
python scripts/run_part3_mup_all.py --best-lr-json outputs/part3_mup_lr_sweep/best_lr.json
```

Fit SP vs. muP scaling curves and compute the 10x extrapolation:

```bash
python scripts/fit_part3_scaling.py ^
  --sp-runs-dir outputs/part2_v2 ^
  --mup-runs-dir outputs/part3_mup ^
  --sp-sweep-json outputs/part2_lr_sweep_v2/sweep_results.json ^
  --mup-sweep-json outputs/part3_mup_lr_sweep/sweep_results.json ^
  --output-dir outputs/part3_analysis
```

On Linux/macOS, replace `^` with `\` for line continuation.

### Part 4: Best Model Training and Generation

Continue training the best model:

```bash
python scripts/run_part4_train.py --config configs/part4_best.yaml
```

Run final temperature-group generation:

```bash
python scripts/run_generate.py ^
  --config configs/part4_generation_temperature_groups.yaml ^
  --checkpoint-path outputs/part4_best/best_mup_xl/checkpoints/best.pt ^
  --tokenizer-path data/processed/v1-clean-rawsplit/tokenizer/tokenizer.json ^
  --output-dir outputs/part4_generation_temperature_groups
```

Evaluate generated samples:

```bash
python scripts/run_eval.py ^
  --train-config configs/part4_best.yaml ^
  --checkpoint-path outputs/part4_best/best_mup_xl/checkpoints/best.pt ^
  --samples-jsonl outputs/part4_generation_temperature_groups/samples.jsonl ^
  --output-dir outputs/part4_generation_temperature_groups_eval
```

Before generation, verify tokenizer alignment if needed:

```bash
python scripts/check_tokenizer_alignment.py ^
  --tokenizer-path data/processed/v1-clean-rawsplit/tokenizer/tokenizer.json ^
  --dataset-repo Zala0429/svg-scaling-v2-tokenized
```

## Colab / Drive Notes

The YAML configs include `drive_output_dir` fields for checkpoint syncing. In Colab, mount Drive first and rerun interrupted cells; trainers resume from `latest.pt` when available.

The notebooks used for the submitted experiments are:

```text
colab_train_part1.ipynb
colab_train_part2.ipynb
colab_train_part3.ipynb
colab_train_part4.ipynb
```

## Generated Outputs

Experiment outputs are not required to be committed. The scripts write run artifacts under `outputs/` by default, including:

```text
outputs/part2_lr_sweep_v2/
outputs/part2_v2/
outputs/part3_mup_lr_sweep/
outputs/part3_mup/
outputs/part3_analysis/
outputs/part4_best/
outputs/part4_generation_temperature_groups/
outputs/part4_generation_temperature_groups_eval/
```

In Colab, the same artifacts can be synced to Google Drive through the `drive_output_dir` values in the YAML configs.
