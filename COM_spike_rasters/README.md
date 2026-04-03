# COM_spike_rasters

Center-of-mass spike-raster pipeline and AL032 manifest/audit files used to generate the COM roster set.

## What is included

- `build_localized_single_session_raster.py`: per-session raw-localization and raster builder
- `run_localization_from_manifest.py`: manifest-driven single-session runner
- `build_al032_session_manifest.py`: AL032 session manifest generator
- `audit_al032_phase4_outputs.py`: post-run audit, schema check, and completion manifest writer
- `submit_al032_localization_array.sbatch`: Slurm array submission wrapper
- `al032_session_manifest.csv` and `al032_session_manifest.json`: runtime session manifest
- `al032_phase4_manifest.csv` and `al032_phase4_manifest.json`: mouse-level completion manifest
- `al032_phase4_schema_report.json`: schema and quality audit summary
- `al032_phase4_contract.json`: frozen output contract
- `al032_phase4_completion_checklist.md`: completion checklist

## What is not included

The heavy per-session output data under `spike_rosters_raw/al032_session_outputs/` is not bundled here. That output tree is about 1.9 GB and contains generated tables and figures rather than source code.

## Notes

- The pipeline localizes spikes with SpikeInterface `center_of_mass`.
- `AL032_2019-11-21` is handled in partial mode because the source raw file contains a known corrupted compressed chunk range.
- Paths in the bundled manifests are the original HPC paths used for the AL032 run.
