#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_utils import csv_write, dump_json, now_iso, read_cluster_group


def check_file(path_str: str) -> tuple[bool, str]:
    path = Path(path_str)
    return path.exists(), str(path)


def validate_session(row: pd.Series) -> dict:
    issues: list[str] = []
    warnings: list[str] = []

    required_file_fields = [
        "ks_path",
        "spike_times_path",
        "spike_clusters_path",
        "templates_path",
        "channel_positions_path",
        "cluster_group_path",
        "params_path",
        "raw_waveforms_path",
    ]
    exists = {}
    for field in required_file_fields:
        path = Path(row[field])
        exists[field] = path.exists()
        if not path.exists():
            issues.append(f"missing:{field}")

    spike_times = np.load(row["spike_times_path"], mmap_mode="r") if exists["spike_times_path"] else None
    spike_clusters = np.load(row["spike_clusters_path"], mmap_mode="r") if exists["spike_clusters_path"] else None
    templates = np.load(row["templates_path"], mmap_mode="r") if exists["templates_path"] else None
    channel_positions = np.load(row["channel_positions_path"], mmap_mode="r") if exists["channel_positions_path"] else None
    good_cluster_ids = []
    if exists["cluster_group_path"]:
        good_cluster_ids = read_cluster_group(Path(row["cluster_group_path"])).query("group.str.lower() == 'good'", engine="python")["cluster_id"].astype(int).tolist()

    spike_arrays_aligned = False
    if spike_times is not None and spike_clusters is not None:
        spike_arrays_aligned = spike_times.shape[0] == spike_clusters.shape[0]
        if not spike_arrays_aligned:
            issues.append("shape_mismatch:spike_times_vs_spike_clusters")

    waveform_file_count = 0
    good_waveform_hit_count = 0
    waveform_shape = None
    waveform_cv_dim_two = False
    waveform_channels_match = False
    waveform_timepoints = None
    waveform_dir = Path(row["raw_waveforms_path"])
    if waveform_dir.exists():
        files = sorted(waveform_dir.glob("Unit*_RawSpikes.npy"))
        waveform_file_count = len(files)
        preview_files = files[: min(3, len(files))]
        if preview_files:
            arr = np.load(preview_files[0], mmap_mode="r")
            waveform_shape = list(arr.shape)
            waveform_timepoints = int(arr.shape[0]) if arr.ndim >= 1 else None
            waveform_cv_dim_two = bool(arr.ndim == 3 and arr.shape[2] == 2)
            if not waveform_cv_dim_two:
                issues.append("invalid_waveform_shape")
            if channel_positions is not None and arr.ndim >= 2:
                waveform_channels_match = int(arr.shape[1]) == int(channel_positions.shape[0])
                if not waveform_channels_match:
                    warnings.append("channel_positions_do_not_match_waveform_channels")
            expected = {f"Unit{cluster_id}_RawSpikes.npy" for cluster_id in good_cluster_ids}
            found = {file_path.name for file_path in files}
            good_waveform_hit_count = len(expected.intersection(found))
            if good_cluster_ids and good_waveform_hit_count != len(good_cluster_ids):
                missing_count = len(good_cluster_ids) - good_waveform_hit_count
                issues.append(f"missing_good_unit_waveforms:{missing_count}")
        else:
            issues.append("missing_waveform_files")

    template_channels_match = False
    if templates is not None and channel_positions is not None:
        template_channels_match = int(templates.shape[2]) == int(channel_positions.shape[0])
        if not template_channels_match:
            issues.append("shape_mismatch:templates_vs_channel_positions")

    sample_rate_present = bool(row["sample_rate_hz"] and row["sample_rate_hz"] > 0)
    if not sample_rate_present:
        issues.append("missing_sample_rate")

    unitmatch_ready = len(issues) == 0

    return {
        "session_name": row["session_name"],
        "session_order": int(row["session_order"]),
        "ks_path": row["ks_path"],
        "raw_waveforms_path": row["raw_waveforms_path"],
        "unit_label_path": row["cluster_group_path"],
        "prepared_data_path": row["prepared_data_path"],
        "localization_spike_table_path": row["localized_spike_table_path"],
        "localization_method": row["localization_method"],
        "localization_mode": row["localization_mode"],
        "sample_rate_hz": float(row["sample_rate_hz"]),
        "duration_s": float(row["duration_s"]),
        "good_cluster_count": int(row["good_cluster_count"]),
        "required_files_exist": all(exists.values()),
        "spike_arrays_aligned": spike_arrays_aligned,
        "waveform_file_count": waveform_file_count,
        "good_waveform_hit_count": good_waveform_hit_count,
        "waveform_shape": "" if waveform_shape is None else "x".join(map(str, waveform_shape)),
        "waveform_timepoints": waveform_timepoints if waveform_timepoints is not None else "",
        "waveform_cv_dim_two": waveform_cv_dim_two,
        "waveform_channels_match_channel_positions": waveform_channels_match,
        "template_channels_match_channel_positions": template_channels_match,
        "sample_rate_present": sample_rate_present,
        "unitmatch_ready": unitmatch_ready,
        "issues": ";".join(issues),
        "warnings": ";".join(warnings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/manifests/al032_12session_manifest.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/manifests/unitmatch_input_manifest.csv",
    )
    parser.add_argument(
        "--output-json",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/manifests/session_validation_report.json",
    )
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.manifest_csv)
    rows = [validate_session(row) for _, row in manifest_df.iterrows()]
    out_df = pd.DataFrame(rows).sort_values("session_order")
    csv_write(out_df, Path(args.output_csv))
    payload = {
        "created_at": now_iso(),
        "manifest_csv": args.manifest_csv,
        "session_count": int(out_df.shape[0]),
        "ready_session_count": int(out_df["unitmatch_ready"].sum()),
        "not_ready_sessions": out_df.loc[~out_df["unitmatch_ready"], "session_name"].tolist(),
        "sessions": out_df.to_dict(orient="records"),
    }
    dump_json(Path(args.output_json), payload)


if __name__ == "__main__":
    main()
