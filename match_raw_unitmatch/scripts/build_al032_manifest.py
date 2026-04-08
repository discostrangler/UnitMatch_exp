#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_utils import (
    count_unique_clusters,
    csv_write,
    discover_al032_sessions,
    dump_json,
    find_raw_waveforms_path,
    get_good_cluster_ids,
    get_raw_waveform_unit_ids,
    maybe_float,
    now_iso,
    parse_params_py,
    parse_session_date,
    parse_spikeglx_meta,
    read_session_validation_report,
    to_builtin,
)


def build_row(session_path: Path, localization_root: Path) -> dict:
    session_name = session_path.name
    session_date = parse_session_date(session_name)
    ks_path = session_path / "ks"
    params_path = ks_path / "params.py"
    meta_path = next(iter(sorted(session_path.glob("*.ap.meta"))), None)
    meta = parse_spikeglx_meta(meta_path) if meta_path else {}
    params = parse_params_py(params_path)
    cluster_group_path = ks_path / "cluster_group.tsv"
    good_cluster_ids = get_good_cluster_ids(cluster_group_path)
    raw_waveforms_path = find_raw_waveforms_path(ks_path)
    raw_waveform_unit_ids = get_raw_waveform_unit_ids(raw_waveforms_path) if raw_waveforms_path else []
    spike_times_path = ks_path / "spike_times.npy"
    spike_clusters_path = ks_path / "spike_clusters.npy"
    spike_times = np.load(spike_times_path, mmap_mode="r")
    sample_rate = maybe_float(params.get("sample_rate")) or maybe_float(meta.get("imSampRate")) or 30000.0
    duration_s = maybe_float(meta.get("fileTimeSecs"))
    if duration_s is None:
        duration_s = float(spike_times[-1]) / float(sample_rate) if spike_times.shape[0] else 0.0
    report = read_session_validation_report(localization_root, session_name)
    outputs = report.get("outputs", {}) if report else {}
    row = {
        "session_name": session_name,
        "session_date": session_date,
        "session_order": None,
        "session_path": str(session_path),
        "ks_path": str(ks_path),
        "spike_times_path": str(spike_times_path),
        "spike_clusters_path": str(spike_clusters_path),
        "templates_path": str(ks_path / "templates.npy"),
        "channel_positions_path": str(ks_path / "channel_positions.npy"),
        "cluster_group_path": str(cluster_group_path),
        "prepared_data_path": str(ks_path / "PreparedData.mat") if (ks_path / "PreparedData.mat").exists() else "",
        "params_path": str(params_path) if params_path.exists() else "",
        "raw_waveforms_path": str(raw_waveforms_path) if raw_waveforms_path else "",
        "sample_rate_hz": sample_rate,
        "duration_s": duration_s,
        "total_spike_count": int(spike_times.shape[0]),
        "cluster_count_from_spike_clusters": count_unique_clusters(spike_clusters_path),
        "cluster_count_from_cluster_group": len(pd.read_csv(cluster_group_path, sep="\t")) if cluster_group_path.exists() else 0,
        "good_cluster_count": len(good_cluster_ids),
        "good_cluster_ids_preview": ",".join(map(str, good_cluster_ids[:10])),
        "raw_waveform_file_count": len(raw_waveform_unit_ids),
        "good_cluster_waveform_coverage_count": len(set(good_cluster_ids).intersection(raw_waveform_unit_ids)),
        "raw_ap_cbin_exists": any(session_path.glob("*.ap.cbin")),
        "raw_ap_ch_exists": any(session_path.glob("*.ap.ch")),
        "raw_ap_meta_exists": meta_path is not None,
        "raw_availability": bool(any(session_path.glob("*.ap.cbin")) and any(session_path.glob("*.ap.ch")) and meta_path is not None),
        "localized_outputs_exist": bool(report),
        "localized_spike_table_path": outputs.get("spike_table_csv_gz", ""),
        "localized_good_cluster_summary_path": outputs.get("good_cluster_summary_csv", ""),
        "localized_validation_report_path": str(localization_root / session_name / f"{session_name}_validation_report.json") if report else "",
        "localization_method": report.get("localization_method", ""),
        "localization_mode": report.get("run_mode", ""),
        "localization_fraction_overall": report.get("localized_fraction_overall"),
        "localization_fraction_good_clusters": report.get("localized_fraction_good_clusters"),
        "partial_or_exact": report.get("run_mode", ""),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/scratch/am15577/UnitMatch/raw_data/extracted")
    parser.add_argument(
        "--localization-root",
        default="/scratch/am15577/UnitMatch/spike_rosters_raw/al032_rosters_monopolar_triangulation",
    )
    parser.add_argument(
        "--output-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/manifests/al032_12session_manifest.csv",
    )
    parser.add_argument(
        "--output-json",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/manifests/al032_12session_manifest.json",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    localization_root = Path(args.localization_root)
    sessions = discover_al032_sessions(raw_root)
    rows = [build_row(session_path, localization_root) for session_path in sessions]
    for index, row in enumerate(rows, start=1):
        row["session_order"] = index

    df = pd.DataFrame(rows)
    csv_write(df, Path(args.output_csv))
    payload = {
        "created_at": now_iso(),
        "raw_root": str(raw_root),
        "localization_root": str(localization_root),
        "session_count": len(rows),
        "sessions": [to_builtin(row) for row in rows],
    }
    dump_json(Path(args.output_json), payload)


if __name__ == "__main__":
    main()
