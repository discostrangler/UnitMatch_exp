#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


KNOWN_CORRUPTION = {
    "AL032_2019-11-21": {
        "corruption_known": True,
        "partial_mode": True,
        "corrupted_chunk_start": 1767,
        "corrupted_chunk_end": 1772,
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AL032 per-session localization manifest.")
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/raw_data/extracted"),
        help="Root directory containing extracted session folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_session_outputs"),
        help="Root directory where per-session localization outputs will be written.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_session_manifest.csv"),
        help="CSV manifest output path.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_session_manifest.json"),
        help="JSON manifest output path.",
    )
    parser.add_argument(
        "--localization-method",
        type=str,
        default="center_of_mass",
        choices=["center_of_mass", "monopolar_triangulation"],
        help="Localization method label to embed in the manifest rows.",
    )
    return parser.parse_args()


def parse_meta(meta_path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in meta_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key] = value
    return meta


def read_cluster_group_counts(cluster_group_path: Path) -> tuple[int, int]:
    cluster_count = 0
    good_cluster_count = 0
    with cluster_group_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            cluster_count += 1
            if row.get("group") == "good":
                good_cluster_count += 1
    return cluster_count, good_cluster_count


def derive_status(validation_report_path: Path) -> str:
    if not validation_report_path.exists():
        return "pending"

    report = json.loads(validation_report_path.read_text())
    if report.get("run_mode") == "partial" or report.get("skipped_corrupted_spikes", 0) > 0 or report.get("failed_window_count", 0) > 0:
        return "partial"
    if report.get("join_integrity", {}).get("localized_plus_skipped_equals_total"):
        return "complete"
    return "failed"


def build_row(session_dir: Path, output_root: Path, localization_method: str) -> dict[str, object]:
    session_name = session_dir.name
    ks_dir = session_dir / "ks"
    meta_path = next(session_dir.glob("*.ap.meta"))
    cbin_path = next(session_dir.glob("*.ap.cbin"))
    ch_path = next(session_dir.glob("*.ap.ch"))
    meta = parse_meta(meta_path)
    sample_rate_hz = float(meta.get("imSampRate", "30000"))
    n_saved_chans = int(meta.get("nSavedChans", "0"))

    spike_times_path = ks_dir / "spike_times.npy"
    spike_clusters_path = ks_dir / "spike_clusters.npy"
    spike_templates_path = ks_dir / "spike_templates.npy"
    templates_path = ks_dir / "templates.npy"
    channel_positions_path = ks_dir / "channel_positions.npy"
    cluster_group_path = ks_dir / "cluster_group.tsv"

    required_paths = [
        cbin_path,
        ch_path,
        meta_path,
        spike_times_path,
        spike_clusters_path,
        spike_templates_path,
        templates_path,
        channel_positions_path,
        cluster_group_path,
    ]
    raw_localization_possible = all(path.exists() for path in required_paths)

    spike_times = np.load(spike_times_path, mmap_mode="r")
    spike_clusters = np.load(spike_clusters_path, mmap_mode="r")
    cluster_count, good_cluster_count = read_cluster_group_counts(cluster_group_path)
    good_cluster_ids: list[int] = []
    with cluster_group_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("group") == "good":
                good_cluster_ids.append(int(row["cluster_id"]))
    duration_s = float(int(spike_times[-1]) / sample_rate_hz) if len(spike_times) else 0.0

    output_dir = output_root / session_name
    validation_report_path = output_dir / f"{session_name}_validation_report.json"

    corruption_info = KNOWN_CORRUPTION.get(
        session_name,
        {
            "corruption_known": False,
            "partial_mode": False,
            "corrupted_chunk_start": "",
            "corrupted_chunk_end": "",
        },
    )

    row = {
        "session_name": session_name,
        "localization_method": localization_method,
        "session_path": str(session_dir),
        "raw_cbin_path": str(cbin_path),
        "raw_ch_path": str(ch_path),
        "raw_meta_path": str(meta_path),
        "ks_dir": str(ks_dir),
        "spike_times_path": str(spike_times_path),
        "spike_clusters_path": str(spike_clusters_path),
        "spike_templates_path": str(spike_templates_path),
        "templates_path": str(templates_path),
        "channel_positions_path": str(channel_positions_path),
        "cluster_group_path": str(cluster_group_path),
        "sample_rate_hz": sample_rate_hz,
        "duration_s": duration_s,
        "n_saved_chans": n_saved_chans,
        "total_spikes": int(spike_times.shape[0]),
        "cluster_count": int(cluster_count),
        "good_cluster_count": int(good_cluster_count),
        "good_cluster_spike_count": int(np.count_nonzero(np.isin(spike_clusters, np.asarray(good_cluster_ids, dtype=np.int32)))),
        "raw_localization_possible": raw_localization_possible,
        "corruption_known": corruption_info["corruption_known"],
        "partial_mode": corruption_info["partial_mode"],
        "corrupted_chunk_start": corruption_info["corrupted_chunk_start"],
        "corrupted_chunk_end": corruption_info["corrupted_chunk_end"],
        "output_dir": str(output_dir),
        "validation_report_path": str(validation_report_path),
        "status": derive_status(validation_report_path),
    }
    return row


def main() -> None:
    args = parse_args()
    extracted_root = args.extracted_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted(path for path in extracted_root.glob("AL032_*") if path.is_dir())
    rows = [build_row(session_dir, output_root, args.localization_method) for session_dir in session_dirs]

    fieldnames = list(rows[0].keys()) if rows else []
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(rows, indent=2))

    print(f"Wrote {len(rows)} AL032 sessions to {args.manifest_csv}")
    print(f"Wrote JSON manifest to {args.manifest_json}")


if __name__ == "__main__":
    main()
