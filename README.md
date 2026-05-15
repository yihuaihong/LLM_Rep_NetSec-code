# LLM_Rep_NetSec — Code

Code and SLURM submission scripts for the paper **"When 99% Probe Accuracy Lies: LLMs Have No Generalizable Network-Attack Concept"** (NeurIPS 2026 submission).

The paper runs a four-test battery (baseline equivalence, decomposition, causal effect, cross-dataset transfer) on LLM hidden-state probe directions for a network-flow attack-classification task across 4 LLMs and 5 datasets.

## Layout

```
scripts/      experiment runners and figure/table builders (~40 .py files)
utils/        shared helpers: model loading, probe fitting, data loaders, eval
sbatch/       SLURM submission templates for cluster runs
configs/      default.yaml — paths, hyperparameters, holdout splits
environment.yml / requirements.txt   reproducible Python environment
```

## Quick start

```bash
# 1. Environment
conda env create -f environment.yml          # creates `netsec_rep`
conda activate netsec_rep

# 2. Edit configs/default.yaml to point at your local
#    - model cache_dir (HuggingFace snapshots)
#    - dataset paths (CICIDS2017/2018, UNSW-NB15, IoT-23, CTU-13)

# 3. Extract hidden-state representations for one (model, dataset)
PYTHONPATH=. python scripts/exp_phase1_directions.py \
    --model Qwen3-8B --dataset cicids2017

# 4. Run the four-test battery from cached representations
PYTHONPATH=. python scripts/exp_phase1_tfidf_baseline.py        # baseline
PYTHONPATH=. python scripts/exp_phase1_multi_axis.py            # decomposition
PYTHONPATH=. python scripts/exp_a1_direct_classifier.py         # direct LLM
PYTHONPATH=. python scripts/exp_phase1_steering.py              # causal steering
PYTHONPATH=. python scripts/exp_phase2_zero_shot_transfer.py    # transfer
```

For cluster runs, use the `sbatch/` templates:
```bash
MODEL=Qwen3-8B DATASET=cicids2017 sbatch sbatch/run_model_dataset.sbatch
```

## Models evaluated

Headline (instruction-tuned): Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3,
Qwen3-8B-Instruct, Gemma-2-9B-it. Causal-steering / format / adversarial
ablations additionally use Qwen3-8B-Base (to avoid chat-template confound on
the JBB harmful direction). A scale ablation uses Qwen2.5-32B-Instruct.

## Datasets

| Dataset      | Schema                | Role             |
|--------------|-----------------------|------------------|
| CICIDS2017   | CICFlowMeter (18 num) | main + train     |
| UNSW-NB15    | Argus-style           | main             |
| CICIDS2018   | CICFlowMeter (same)   | transfer         |
| IoT-23       | Zeek `conn.log`       | transfer         |
| CTU-13       | Argus binetflow       | transfer         |

Holdout split for zero-day evaluation: CICIDS = {Bot, Heartbleed, Infiltration};
UNSW = {Backdoor, Shellcode, Worms}.

## Notes on reproducibility

- All experiments are deterministic given a fixed random seed (we used 42, 7, 99
  for the three-seed stability check); see `exp_a4_multi_seed.py`.
- Hidden states are extracted once per `(model, dataset)` and cached as `.pt`
  files under `results/representations/` (not included in this repo — too large).
- Per-experiment metrics JSONs are written to `results/metrics/`; these are
  consumed by `build_paper_figures.py` and `build_paper_tables.py`.

## Citation

If you use this code, please cite the paper (BibTeX entry to be added once the
paper is finalized).
