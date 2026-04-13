#!/bin/bash
#SBATCH --job-name=al032_dredge_f5
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --output=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/logs/slurm_dredge_f5_%j.out
#SBATCH --error=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/logs/slurm_dredge_f5_%j.err
#SBATCH --open-mode=append

set -euo pipefail

cd /scratch/am15577/UnitMatch
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate torchgpu
export MPLCONFIGDIR=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"

/scratch/am15577/conda/envs/torchgpu/bin/python /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/build_corrected_unit_features.py
/scratch/am15577/conda/envs/torchgpu/bin/python /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/prepare_unitmatch_dredge_run.py
