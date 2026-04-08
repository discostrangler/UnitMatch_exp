# match_raw_unitmatch

This folder contains the raw-session UnitMatch pipeline for AL032 across 12 sessions.

Included here:

- `scripts/`: manifest building, validation, UnitMatch execution, tracked-ID attachment, raster building, and UnitMatch QC plotting
- `configs/`: frozen run and plot configuration used for the AL032 run
- `manifests/`: 12-session input manifests and validation reports
- `outputs/unitmatch_raw_12session/outputs_unitmatch/`: lightweight UnitMatch QC figures and summary tables

Not included here:

- large generated outputs such as attached spike tables, raw UnitMatch matrices, or raster waveform sample archives
- raw data
- the upstream `UnitMatch-main-2` repository

Local source paths used by this pipeline:

- raw extracted sessions: `/scratch/am15577/UnitMatch/raw_data/extracted`
- official UnitMatch code: `/scratch/am15577/UnitMatch/UnitMatch-main-2`

Main entry points:

- `scripts/run_unitmatch_al032.py`
- `scripts/build_tracked_unit_tables.py`
- `scripts/attach_tracked_ids_to_localized_spikes.py`
- `scripts/build_tracked_unit_coverage_summary.py`
- `scripts/build_al032_12session_raster.py`
- `scripts/build_al032_12session_raster_plus_waveforms.py`
- `scripts/plot_unitmatch_qc_figures.py`
