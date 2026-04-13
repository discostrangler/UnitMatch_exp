#!/bin/bash
set -euo pipefail

CONFIG=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/configs/unitmatch_dredge_run_config.json
PY=/scratch/am15577/conda/envs/torchgpu/bin/python
TRACKED_ROOT=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/outputs/tracked_tables
QC_DIR=/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/outputs/tracked_tables/dredge_qc

$PY /scratch/am15577/UnitMatch/match_raw_unitmatch/scripts/build_tracked_unit_tables.py --config "$CONFIG"
$PY /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/attach_dredge_tracked_ids_to_corrected_spikes.py --config "$CONFIG"
$PY /scratch/am15577/UnitMatch/match_raw_unitmatch/scripts/build_tracked_unit_coverage_summary.py --config "$CONFIG"
$PY /scratch/am15577/UnitMatch/match_raw_unitmatch/scripts/plot_unitmatch_qc_figures.py --config "$CONFIG" --output-dir "$QC_DIR"
$PY /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/identify_shared_high_confidence_units.py --config "$CONFIG"
$PY /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/estimate_session_alignment_offsets.py --config "$CONFIG"
$PY /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/apply_session_alignment_offsets.py --config "$CONFIG"
$PY /scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/scripts/build_al032_12session_dredge_aligned_raster.py --config "$CONFIG"

cp "$QC_DIR/session_pair_tracking_metrics.csv" "$TRACKED_ROOT/pairwise_session_tracking_metrics.csv"
cp "$QC_DIR/tracked_unit_lifespan_counts.csv" "$TRACKED_ROOT/tracked_unit_lifespan_counts.csv"
