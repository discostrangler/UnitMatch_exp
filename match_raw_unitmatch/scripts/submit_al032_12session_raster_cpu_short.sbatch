#!/bin/bash
#SBATCH --job-name=al032_12raster
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --output=/scratch/am15577/UnitMatch/match_raw_unitmatch/logs/slurm_12session_raster_%j.out
#SBATCH --error=/scratch/am15577/UnitMatch/match_raw_unitmatch/logs/slurm_12session_raster_%j.err
#SBATCH --open-mode=append

set -euo pipefail

cd /scratch/am15577/UnitMatch
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate torchgpu
export MPLCONFIGDIR=/scratch/am15577/UnitMatch/match_raw_unitmatch/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"

/scratch/am15577/conda/envs/torchgpu/bin/python \
  /scratch/am15577/UnitMatch/match_raw_unitmatch/scripts/build_al032_12session_raster.py \
  --config /scratch/am15577/UnitMatch/match_raw_unitmatch/configs/unitmatch_run_config.json \
  --coverage-summary-csv /scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/tracked_tables/tracked_unit_coverage_summary.csv \
  --selected-tracked-units-csv /scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/tracked_tables/selected_tracked_units.csv \
  --session-manifest-csv /scratch/am15577/UnitMatch/match_raw_unitmatch/manifests/unitmatch_input_manifest.csv \
  --overlay-output-csv-gz /scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/figures/al032_12session_selected_overlay_spikes.csv.gz \
  --output-png /scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/figures/al032_12session_raster.png \
  --output-summary-json /scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/figures/al032_12session_raster_summary.json
