#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AL032 localization session from a manifest row.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_session_manifest.csv"),
        help="Path to the AL032 manifest CSV.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Zero-based manifest row index, intended for Slurm array tasks.",
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Session name to run instead of using a task index.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=str,
        default="5s",
        help="Chunk duration to pass to the localization script.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker count to pass to the localization script.",
    )
    parser.add_argument(
        "--highlight-count",
        type=int,
        default=10,
        help="Highlight count for the raster output.",
    )
    parser.add_argument(
        "--localization-method",
        type=str,
        default="center_of_mass",
        choices=["center_of_mass", "monopolar_triangulation"],
        help="SpikeInterface localization method to pass to the session-localization script.",
    )
    parser.add_argument(
        "--localization-feature",
        type=str,
        default="ptp",
        help="Feature parameter to pass to the localization method.",
    )
    parser.add_argument(
        "--monopolar-max-distance-um",
        type=float,
        default=150.0,
        help="Maximum distance parameter for monopolar_triangulation.",
    )
    parser.add_argument(
        "--coordinate-margin-um",
        type=float,
        default=200.0,
        help="Allowed localization margin beyond the probe y-range before spikes are flagged as out of range.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even if the validation report already exists.",
    )
    return parser.parse_args()


def load_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_row(rows: list[dict[str, str]], task_index: int | None, session_name: str | None) -> dict[str, str]:
    if session_name is not None:
        for row in rows:
            if row["session_name"] == session_name:
                return row
        raise KeyError(f"Session {session_name} not found in manifest")

    if task_index is None:
        raise ValueError("Either --task-index or --session-name is required")
    if task_index < 0 or task_index >= len(rows):
        raise IndexError(f"Task index {task_index} is out of range for {len(rows)} sessions")
    return rows[task_index]


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest.resolve())
    row = resolve_row(rows, args.task_index, args.session_name)

    output_dir = Path(row["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report_path = Path(row["validation_report_path"]).resolve()
    if validation_report_path.exists() and not args.force:
        print(f"Outputs already exist at {validation_report_path}; exiting.")
        return

    script_path = Path("/scratch/am15577/UnitMatch/spike_rosters_raw/build_localized_single_session_raster.py")
    command = [
        sys.executable,
        str(script_path),
        "--session",
        row["session_path"],
        "--output-dir",
        str(output_dir),
        "--highlight-count",
        str(args.highlight_count),
        "--chunk-duration",
        args.chunk_duration,
        "--n-jobs",
        str(args.n_jobs),
        "--localization-method",
        args.localization_method,
        "--localization-feature",
        args.localization_feature,
        "--monopolar-max-distance-um",
        str(args.monopolar_max_distance_um),
        "--coordinate-margin-um",
        str(args.coordinate_margin_um),
    ]

    if row.get("partial_mode", "").lower() == "true":
        command.extend(
            [
                "--partial-mode",
                "--corrupted-chunk-start",
                row["corrupted_chunk_start"],
                "--corrupted-chunk-end",
                row["corrupted_chunk_end"],
            ]
        )

    print(f"[runner] session={row['session_name']} output_dir={output_dir}")
    print("[runner] command=" + " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
