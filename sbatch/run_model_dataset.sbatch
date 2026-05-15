#!/bin/bash
#SBATCH --job-name=netsec_rep
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -e
cd /home/yh6210/Research_Projects/LLM_Rep_NetSec

source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh
conda activate netsec_rep

MODEL=${MODEL:?set MODEL=...}
DATASET=${DATASET:?set DATASET=...}
echo "Running model=$MODEL dataset=$DATASET"
nvidia-smi | head -20
PYTHONPATH=. python scripts/run_pipeline.py --model "$MODEL" --dataset "$DATASET"
