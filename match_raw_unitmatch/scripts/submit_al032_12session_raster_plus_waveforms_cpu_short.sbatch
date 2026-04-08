#!/bin/bash
#SBATCH --job-name=al032_12wf
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --output=/scratch/am15577/UnitMatch/match_raw_unitmatch/logs/slurm_12session_raster_wf_%j.out
#SBATCH --error=/scratch/am15577/UnitMatch/match_raw_unitmatch/logs/slurm_12session_raster_wf_%j.err
#SBATCH --open-mode=append

set -euo pipefail

cd /scratch/am15577/UnitMatch
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate torchgpu
export MPLCONFIGDIR=/scratch/am15577/UnitMatch/match_raw_unitmatch/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"

/scratch/am15577/conda/envs/torchgpu/bin/python \
  /scratch/am15577/UnitMatch/match_raw_unitmatch/scripts/build_al032_12session_raster_plus_waveforms.py \
  --config /scratch/am15577/UnitMatch/match_raw_unitmatch/configs/unitmatch_run_config.json \
  --selected-tracked-units-csv /scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/tracked_tables/selected_tracked_units.csv \
  --session-manifest-csv /scratch/am15577/UnitMatch/match_raw_unitmatch/manifests/unitmatch_input_manifest.csv \
  --output-dir /scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/figures \
  --candidate-select-count 8 \
  --final-select-count 6 \
  --min-waveform-sessions 10 \
  --waveform-min-good-spikes 5000 \
  --waveform-min-candidate-spikes 1000 \
  --waveform-sample-count 250 \
  --waveform-oversample-count 400 \
  --waveform-min-aligned-spikes 180 \
  --waveform-display-traces-per-session 40 \
  --waveform-read-window-seconds 10.0
