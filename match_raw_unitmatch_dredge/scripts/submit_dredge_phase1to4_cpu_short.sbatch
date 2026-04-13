#!/usr/bin/bash
#SBATCH --job-name=al032_dredge_p14
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/logs/slurm_al032_dredge_p14_%j.out
#SBATCH --error=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/logs/slurm_al032_dredge_p14_%j.err

set -euo pipefail

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/am15577/conda/envs/torchgpu
export MPLCONFIGDIR=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/tmp/mplconfig
export PYTHONUNBUFFERED=1

python -u /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/freeze_baseline_reference.py
python -u /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/build_dredge_input_manifest.py
python -u /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/run_dredge_per_session.py
python -u /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/audit_dredge_outputs.py
