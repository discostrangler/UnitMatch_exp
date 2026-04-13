#!/usr/bin/bash
#SBATCH --job-name=al032_dredge_xy20
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/logs/slurm_exact_shank_xy_dredge_20units_%j.out
#SBATCH --error=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/logs/slurm_exact_shank_xy_dredge_20units_%j.err

set -euo pipefail

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/am15577/conda/envs/torchgpu
export MPLCONFIGDIR=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/tmp/mplconfig

export EXACT_SHANK_OUTPUT_DIR=exact_shank_xy_amplitude_movie_20units_dredge
export EXACT_SHANK_MOVIE_BASENAME=al032_exact_shank_xy_amplitude_movie_20units_dredge.mp4
export EXACT_SHANK_TARGET_OUTLINE_UNITS=20
export EXACT_SHANK_MAX_OUTLINE_UNITS_PER_SHANK=8
export EXACT_SHANK_MIN_MEAN_PROBABILITY=0.6

python /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/spike_video/build_exact_shank_xy_movie_dredge.py
