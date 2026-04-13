#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from _pipeline_utils import dump_json, now_iso


ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge")
UPSTREAM_ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch")
OUTPUTS = UPSTREAM_ROOT / "outputs"
BASELINE_ROOT = ROOT / "outputs" / "comparisons" / "baseline_reference"


BASELINE_FILES = [
    ("pairwise_overlap/pairwise_session_table.csv", OUTPUTS / "unitmatch_raw_12session" / "eval" / "pairwise_session_table.csv"),
    (
        "pairwise_overlap/ordered_session_tracking_table.csv",
        OUTPUTS / "unitmatch_raw_12session" / "eval" / "ordered_session_tracking_table.csv",
    ),
    (
        "pairwise_overlap/replication_metrics_summary.json",
        OUTPUTS / "unitmatch_raw_12session" / "eval" / "replication_metrics_summary.json",
    ),
    (
        "persistence/tracked_unit_evaluation_table.csv",
        OUTPUTS / "unitmatch_raw_12session" / "eval" / "tracked_unit_evaluation_table.csv",
    ),
    (
        "persistence/tracked_unit_lifespan_counts.csv",
        OUTPUTS / "unitmatch_raw_12session" / "outputs_unitmatch" / "tracked_unit_lifespan_counts.csv",
    ),
    (
        "persistence/unitmatch_tracked_unit_lifespan_histogram.png",
        OUTPUTS / "unitmatch_raw_12session" / "outputs_unitmatch" / "unitmatch_tracked_unit_lifespan_histogram.png",
    ),
    ("selected_units/selected_tracked_units.csv", OUTPUTS / "tracked_tables" / "selected_tracked_units.csv"),
    ("selected_units/tracked_unit_summary.csv", OUTPUTS / "tracked_tables" / "tracked_unit_summary.csv"),
    (
        "selected_units/tracked_unit_coverage_summary.csv",
        OUTPUTS / "tracked_tables" / "tracked_unit_coverage_summary.csv",
    ),
    ("figures/al032_12session_raster.png", OUTPUTS / "figures" / "al032_12session_raster.png"),
    (
        "figures/al032_12session_raster_summary.json",
        OUTPUTS / "figures" / "al032_12session_raster_summary.json",
    ),
    (
        "figures/al032_12session_raster_plus_waveforms.png",
        OUTPUTS / "figures" / "al032_12session_raster_plus_waveforms.png",
    ),
    (
        "figures/al032_12session_raster_plus_waveforms_summary.json",
        OUTPUTS / "figures" / "al032_12session_raster_plus_waveforms_summary.json",
    ),
    (
        "figures/al032_12session_selected_waveform_tracked_units.csv",
        OUTPUTS / "figures" / "al032_12session_selected_waveform_tracked_units.csv",
    ),
]


def ensure_symlink(target: Path, link_path: Path) -> str:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    os.symlink(target, link_path)
    return str(link_path)


def main() -> None:
    BASELINE_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    missing = []

    for rel_dest, src in BASELINE_FILES:
        dest = BASELINE_ROOT / rel_dest
        if not src.exists():
            missing.append(str(src))
            continue
        ensure_symlink(src, dest)
        records.append(
            {
                "category": rel_dest.split("/", 1)[0],
                "label": Path(rel_dest).name,
                "source_path": str(src),
                "baseline_reference_path": str(dest),
                "mode": "symlink",
            }
        )

    payload = {
        "created_at": now_iso(),
        "baseline_reference_root": str(BASELINE_ROOT),
        "linked_file_count": len(records),
        "missing_source_count": len(missing),
        "missing_sources": missing,
        "files": records,
    }
    dump_json(BASELINE_ROOT / "baseline_reference_manifest.json", payload)


if __name__ == "__main__":
    main()
