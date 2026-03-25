# local_runs_1

Clean full-data UnitMatch package for this checkout.

## What is here

- `run_all.py`: single entrypoint that builds the report package
- `data/`: compact CSV and JSON outputs copied or derived from the local full-data runs
- `report/`: paper-style Markdown, HTML, PDF, and the requested figure families
- `manifest.json`: output index

## Usage

Run from the repo root:

```bash
python local_runs_1/run_all.py
```

Useful flags:

- `--force-refresh`: delete generated outputs inside `local_runs_1` and rebuild them
- `--force-source`: re-run the shared source generation scripts under `local_runs/*` before packaging

## Notes

- This package keeps only compact report artifacts inside `local_runs_1` so it stays easier to review and push.
- It reuses the heavier cached outputs already generated under `local_runs/` when possible.
- The bundled checkout supports waveform-based matching, ISI validation, and a spike-time-derived reference-population analog.
- Natural-image-response validation from the paper is not reproducible from the files available in this checkout.
