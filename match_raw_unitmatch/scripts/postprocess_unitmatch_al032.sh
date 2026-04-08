#!/bin/bash
set -euo pipefail

CONFIG=/scratch/am15577/UnitMatch/match_raw_unitmatch/configs/unitmatch_run_config.json
PY=/scratch/am15577/conda/envs/torchgpu/bin/python

$PY /scratch/am15577/UnitMatch/match_raw_unitmatch/scripts/build_tracked_unit_tables.py --config "$CONFIG"
$PY /scratch/am15577/UnitMatch/match_raw_unitmatch/scripts/attach_tracked_ids_to_localized_spikes.py --config "$CONFIG"
