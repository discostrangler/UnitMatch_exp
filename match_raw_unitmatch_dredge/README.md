# match_raw_unitmatch_dredge

This folder contains the DREDge-augmented AL032 raw-session UnitMatch pipeline.

Included here:

- `scripts/`: DREDge per-session correction, corrected-feature building, DREDge UnitMatch execution, tracked-ID attachment, session-offset alignment, raster building, feature/report generation, and Slurm wrappers
- `spike_video/`: exact-shank movie builders and Slurm wrappers for the DREDge-aligned spike videos

Not included here:

- generated outputs
- logs
- temporary caches
- raw data
- the upstream `UnitMatch-main-2` repository

Main entry points:

- `scripts/run_dredge_per_session.py`
- `scripts/build_corrected_unit_features.py`
- `scripts/run_unitmatch_dredge_al032.py`
- `scripts/attach_dredge_tracked_ids_to_corrected_spikes.py`
- `scripts/identify_shared_high_confidence_units.py`
- `scripts/estimate_session_alignment_offsets.py`
- `scripts/apply_session_alignment_offsets.py`
- `scripts/build_al032_12session_dredge_aligned_raster.py`
- `scripts/build_al032_12session_dredge_raster_plus_waveforms.py`
- `spike_video/build_exact_shank_xy_movie_dredge.py`
- `spike_video/build_exact_shank_xy_rgb_movie_dredge.py`
