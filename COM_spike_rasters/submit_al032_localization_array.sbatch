#!/bin/bash
#SBATCH --job-name=al032_loc_array
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --output=/scratch/am15577/UnitMatch/spike_rosters_raw/slurm_al032_loc_array_%A_%a.out
#SBATCH --error=/scratch/am15577/UnitMatch/spike_rosters_raw/slurm_al032_loc_array_%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail

cd /scratch/am15577/UnitMatch
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate torchgpu

MANIFEST=/scratch/am15577/UnitMatch/spike_rosters_raw/al032_session_manifest.csv

PYTHONUNBUFFERED=1 python /scratch/am15577/UnitMatch/spike_rosters_raw/run_localization_from_manifest.py \
  --manifest "${MANIFEST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --chunk-duration 5s \
  --n-jobs 1 \
  --highlight-count 10
