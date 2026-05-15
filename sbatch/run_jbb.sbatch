#!/bin/bash
#SBATCH --job-name=jbb_extract
#SBATCH --account=torch_pr_520_tandon_priority
#SBATCH --partition=a100_tandon,h100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -e
cd /home/yh6210/Research_Projects/LLM_Rep_NetSec

source /scratch/yh6210/miniconda3/etc/profile.d/conda.sh
conda activate netsec_rep

MODEL=${MODEL:?set MODEL=...}
EXTRA_ARGS=${EXTRA_ARGS:-}
echo "Extracting JBB reps for $MODEL  EXTRA_ARGS=$EXTRA_ARGS"
nvidia-smi | head -20
PYTHONPATH=. python scripts/extract_harmful_harmless_multimodel.py --model "$MODEL" $EXTRA_ARGS
