#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_utils import csv_write, dump_json, load_json, now_iso


ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge")
UPSTREAM_ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch")
MANIFEST_PATH = UPSTREAM_ROOT / "manifests" / "al032_12session_manifest.csv"
OUTPUT_CSV = ROOT / "manifests" / "al032_dredge_input_manifest.csv"
OUTPUT_JSON = ROOT / "manifests" / "al032_dredge_input_manifest.json"


def count_localized_spikes(path: Path) -> dict[str, int]:
    counts = {
        "n_localized_spikes_finite": 0,
        "n_good_localized_spikes_finite": 0,
    }
    usecols = ["is_good_cluster", "localization_success", "spike_time_s", "y_um", "amplitude"]
    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=500_000):
        finite_mask = (
            chunk["localization_success"].astype(bool).to_numpy()
            & np.isfinite(chunk["spike_time_s"].to_numpy(dtype=np.float64))
            & np.isfinite(chunk["y_um"].to_numpy(dtype=np.float64))
            & np.isfinite(chunk["amplitude"].to_numpy(dtype=np.float64))
        )
        counts["n_localized_spikes_finite"] += int(finite_mask.sum())
        good_mask = finite_mask & chunk["is_good_cluster"].astype(bool).to_numpy()
        counts["n_good_localized_spikes_finite"] += int(good_mask.sum())
    return counts


def summarize_corrupted_intervals(localized_spike_table_path: Path) -> dict[str, object]:
    session_name = localized_spike_table_path.name.replace("_localized_spike_table.csv.gz", "")
    summary_path = localized_spike_table_path.parent / f"{session_name}_corrupted_region_summary.json"
    report = load_json(summary_path)
    if not isinstance(report, dict):
        return {
            "corrupted_region_summary_path": None,
            "known_corrupted_interval_count": 0,
            "known_corrupted_interval_seconds": 0.0,
            "known_corrupted_interval_info": json.dumps([]),
        }
    regions = report.get("corrupted_regions", [])
    intervals = []
    total_seconds = 0.0
    for region in regions:
        start_s = float(region.get("skip_time_start_s", region.get("time_start_s", 0.0)))
        end_s = float(region.get("skip_time_end_s", region.get("time_end_s", 0.0)))
        total_seconds += max(0.0, end_s - start_s)
        intervals.append(
            {
                "time_start_s": start_s,
                "time_end_s": end_s,
                "spike_count_skipped_region": int(region.get("spike_count_skipped_region", 0)),
                "good_cluster_spike_count_skipped_region": int(region.get("good_cluster_spike_count_skipped_region", 0)),
            }
        )
    return {
        "corrupted_region_summary_path": str(summary_path),
        "known_corrupted_interval_count": len(intervals),
        "known_corrupted_interval_seconds": total_seconds,
        "known_corrupted_interval_info": json.dumps(intervals),
    }


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH).sort_values("session_order")
    rows = []
    for row in manifest.itertuples(index=False):
        localized_spike_table_path = Path(row.localized_spike_table_path)
        localized_counts = count_localized_spikes(localized_spike_table_path)
        corrupted = summarize_corrupted_intervals(localized_spike_table_path)
        rows.append(
            {
                "session_name": row.session_name,
                "session_date": row.session_date,
                "session_order": int(row.session_order),
                "localized_spike_table_path": str(localized_spike_table_path),
                "ks_path": row.ks_path,
                "duration_s": float(row.duration_s),
                "partial_or_exact": row.partial_or_exact,
                "corrupted_region_summary_path": corrupted["corrupted_region_summary_path"],
                "known_corrupted_interval_count": int(corrupted["known_corrupted_interval_count"]),
                "known_corrupted_interval_seconds": float(corrupted["known_corrupted_interval_seconds"]),
                "known_corrupted_interval_info": corrupted["known_corrupted_interval_info"],
                "n_localized_spikes_finite": int(localized_counts["n_localized_spikes_finite"]),
                "n_good_localized_spikes_finite": int(localized_counts["n_good_localized_spikes_finite"]),
                "localization_method": row.localization_method,
                "sample_rate_hz": float(row.sample_rate_hz),
                "channel_positions_path": row.channel_positions_path,
            }
        )

    out = pd.DataFrame(rows)
    csv_write(out, OUTPUT_CSV)
    dump_json(
        OUTPUT_JSON,
        {
            "created_at": now_iso(),
            "manifest_csv": str(OUTPUT_CSV),
            "session_count": int(len(out)),
            "sessions": out.to_dict(orient="records"),
        },
    )


if __name__ == "__main__":
    main()
