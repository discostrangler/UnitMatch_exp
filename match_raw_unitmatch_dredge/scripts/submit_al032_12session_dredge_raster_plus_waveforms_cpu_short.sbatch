#!/bin/bash
#SBATCH --job-name=al032_dredge_wf
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --output=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/logs/slurm_12session_dredge_raster_wf_%j.out
#SBATCH --error=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/logs/slurm_12session_dredge_raster_wf_%j.err
#SBATCH --open-mode=append

set -euo pipefail

cd /scratch/am15577/UnitMatch
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate torchgpu
export MPLCONFIGDIR=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"

/scratch/am15577/conda/envs/torchgpu/bin/python -u \
  /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/build_al032_12session_dredge_raster_plus_waveforms.py \
  --config /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/configs/unitmatch_dredge_run_config.json \
  --selected-tracked-units-csv /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/outputs/tracked_tables/selected_tracked_units.csv \
  --session-manifest-csv /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/manifests/al032_dredge_input_manifest.csv \
  --output-dir /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/outputs/figures \
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
