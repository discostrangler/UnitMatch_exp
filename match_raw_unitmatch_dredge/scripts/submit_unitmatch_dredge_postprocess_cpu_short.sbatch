#!/bin/bash
#SBATCH --job-name=al032_dredge_post
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --output=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/logs/slurm_unitmatch_dredge_post_%j.out
#SBATCH --error=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/logs/slurm_unitmatch_dredge_post_%j.err
#SBATCH --open-mode=append

set -euo pipefail

cd /scratch/am15577/UnitMatch
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate torchgpu
export MPLCONFIGDIR=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"

/usr/bin/bash /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/postprocess_unitmatch_dredge.sh
