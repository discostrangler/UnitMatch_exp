#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_utils import csv_write, dump_json, now_iso


ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge")
UPSTREAM_ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch")
DREDGE_ROOT = ROOT / "outputs" / "dredge_per_session"
FEATURE_ROOT = ROOT / "outputs" / "corrected_unit_features"
MANIFEST_PATH = ROOT / "manifests" / "al032_dredge_input_manifest.csv"
BASELINE_CONFIG_PATH = UPSTREAM_ROOT / "configs" / "unitmatch_run_config.json"
BASELINE_WAVEFORMINFO_PATH = UPSTREAM_ROOT / "outputs" / "unitmatch_raw_12session" / "WaveformInfo.npz"
BASELINE_CLUSINFO_PATH = UPSTREAM_ROOT / "outputs" / "unitmatch_raw_12session" / "ClusInfo.pickle"


def robust_spread(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def build_unit_index_table() -> pd.DataFrame:
    with open(BASELINE_CLUSINFO_PATH, "rb") as handle:
        clus_info = pickle.load(handle)

    session_names = [str(v) for v in clus_info["session_names"].tolist()]
    session_id = np.asarray(clus_info["session_id"], dtype=np.int64)
    original_ids = np.asarray(clus_info["original_ids"], dtype=object)

    rows = []
    for unit_index in range(session_id.shape[0]):
        cluster_value = original_ids[unit_index]
        if isinstance(cluster_value, np.ndarray):
            cluster_id = int(np.asarray(cluster_value).reshape(-1)[0])
        elif isinstance(cluster_value, (list, tuple)):
            cluster_id = int(cluster_value[0])
        else:
            cluster_id = int(cluster_value)
        session_index = int(session_id[unit_index])
        rows.append(
            {
                "unitmatch_unit_index": int(unit_index),
                "session_index": session_index,
                "session_name": session_names[session_index],
                "cluster_id": cluster_id,
            }
        )
    return pd.DataFrame(rows).sort_values(["session_index", "cluster_id"]).reset_index(drop=True)


def summarize_corrected_spikes(session_name: str, corrected_csv_gz: Path) -> pd.DataFrame:
    usecols = [
        "cluster_id",
        "is_good_cluster",
        "localization_success",
        "y_um",
        "y_um_dredge",
        "dredge_peak_displacement_um",
        "amplitude",
    ]
    by_cluster: dict[int, dict[str, list[np.ndarray]]] = {}
    for chunk in pd.read_csv(
        corrected_csv_gz,
        compression="gzip",
        usecols=usecols,
        chunksize=250_000,
        low_memory=False,
    ):
        good_mask = (
            chunk["is_good_cluster"].astype(bool).to_numpy()
            & chunk["localization_success"].astype(bool).to_numpy()
            & np.isfinite(chunk["y_um"].to_numpy(dtype=np.float64))
            & np.isfinite(chunk["y_um_dredge"].to_numpy(dtype=np.float64))
        )
        if not np.any(good_mask):
            continue
        good = chunk.loc[good_mask].copy()
        good["y_shift_um"] = (
            good["y_um_dredge"].to_numpy(dtype=np.float64) - good["y_um"].to_numpy(dtype=np.float64)
        )
        for cluster_id, group in good.groupby("cluster_id", sort=False):
            cluster_id = int(cluster_id)
            payload = by_cluster.setdefault(
                cluster_id,
                {"y_before": [], "y_after": [], "y_shift": [], "amplitude": []},
            )
            payload["y_before"].append(group["y_um"].to_numpy(dtype=np.float32))
            payload["y_after"].append(group["y_um_dredge"].to_numpy(dtype=np.float32))
            payload["y_shift"].append(group["y_shift_um"].to_numpy(dtype=np.float32))
            payload["amplitude"].append(group["amplitude"].to_numpy(dtype=np.float32))

    rows = []
    for cluster_id in sorted(by_cluster):
        payload = by_cluster[cluster_id]
        y_before = np.concatenate(payload["y_before"]).astype(np.float64, copy=False)
        y_after = np.concatenate(payload["y_after"]).astype(np.float64, copy=False)
        y_shift = np.concatenate(payload["y_shift"]).astype(np.float64, copy=False)
        amplitude = np.concatenate(payload["amplitude"]).astype(np.float64, copy=False)
        spread_before = robust_spread(y_before)
        spread_after = robust_spread(y_after)
        rows.append(
            {
                "session_name": session_name,
                "cluster_id": int(cluster_id),
                "n_good_localized_spikes": int(y_before.size),
                "median_y_um_before": float(np.median(y_before)),
                "median_y_um_after": float(np.median(y_after)),
                "median_y_shift_um": float(np.median(y_shift)),
                "mean_y_shift_um": float(np.mean(y_shift)),
                "std_y_shift_um": float(np.std(y_shift, ddof=1)) if y_shift.size > 1 else 0.0,
                "spread_before_um": float(spread_before),
                "spread_after_um": float(spread_after),
                "spread_ratio_after_over_before": float(spread_after / spread_before) if spread_before > 0 else np.nan,
                "q05_y_shift_um": float(np.quantile(y_shift, 0.05)),
                "q95_y_shift_um": float(np.quantile(y_shift, 0.95)),
                "median_amplitude": float(np.median(amplitude)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH).sort_values("session_order")
    unit_index_df = build_unit_index_table()
    waveform_info = np.load(BASELINE_WAVEFORMINFO_PATH, allow_pickle=True)
    avg_centroid = np.asarray(waveform_info["avg_centroid"], dtype=np.float64)
    avg_waveform_per_tp = np.asarray(waveform_info["avg_waveform_per_tp"], dtype=np.float64)
    max_site = np.asarray(waveform_info["max_site"], dtype=np.int64)

    per_cluster_frames = []
    for row in manifest.itertuples(index=False):
        corrected_csv_gz = DREDGE_ROOT / row.session_name / f"{row.session_name}_corrected_spike_table.csv.gz"
        per_cluster_frames.append(summarize_corrected_spikes(row.session_name, corrected_csv_gz))
    cluster_summary = pd.concat(per_cluster_frames, ignore_index=True)

    merged = unit_index_df.merge(cluster_summary, on=["session_name", "cluster_id"], how="left")

    centroid_x_cv0 = avg_centroid[0, :, 0]
    centroid_x_cv1 = avg_centroid[0, :, 1]
    centroid_y_cv0 = avg_centroid[1, :, 0]
    centroid_y_cv1 = avg_centroid[1, :, 1]
    centroid_z_cv0 = avg_centroid[2, :, 0]
    centroid_z_cv1 = avg_centroid[2, :, 1]

    y_shift = merged["median_y_shift_um"].fillna(0.0).to_numpy(dtype=np.float64)
    corrected_y_cv0 = centroid_y_cv0 + y_shift
    corrected_y_cv1 = centroid_y_cv1 + y_shift

    peak_tp = int(avg_waveform_per_tp.shape[2] // 2)
    traj_y_peak_cv0 = avg_waveform_per_tp[1, :, peak_tp, 0]
    traj_y_peak_cv1 = avg_waveform_per_tp[1, :, peak_tp, 1]
    corrected_traj_y_peak_cv0 = traj_y_peak_cv0 + y_shift
    corrected_traj_y_peak_cv1 = traj_y_peak_cv1 + y_shift

    uncorrected = merged.copy()
    uncorrected["feature_source"] = "uncorrected"
    uncorrected["centroid_x_cv0_um"] = centroid_x_cv0
    uncorrected["centroid_x_cv1_um"] = centroid_x_cv1
    uncorrected["centroid_y_cv0_um"] = centroid_y_cv0
    uncorrected["centroid_y_cv1_um"] = centroid_y_cv1
    uncorrected["centroid_z_cv0_um"] = centroid_z_cv0
    uncorrected["centroid_z_cv1_um"] = centroid_z_cv1
    uncorrected["centroid_y_mean_um"] = np.nanmean(np.stack([centroid_y_cv0, centroid_y_cv1], axis=1), axis=1)
    uncorrected["centroid_cv_disagreement_um"] = np.abs(centroid_y_cv0 - centroid_y_cv1)
    uncorrected["trajectory_y_peak_cv0_um"] = traj_y_peak_cv0
    uncorrected["trajectory_y_peak_cv1_um"] = traj_y_peak_cv1
    uncorrected["trajectory_y_peak_mean_um"] = np.nanmean(np.stack([traj_y_peak_cv0, traj_y_peak_cv1], axis=1), axis=1)
    uncorrected["max_site_cv0"] = max_site[:, 0]
    uncorrected["max_site_cv1"] = max_site[:, 1]

    corrected = merged.copy()
    corrected["feature_source"] = "dredge_corrected"
    corrected["centroid_x_cv0_um"] = centroid_x_cv0
    corrected["centroid_x_cv1_um"] = centroid_x_cv1
    corrected["centroid_y_cv0_um"] = corrected_y_cv0
    corrected["centroid_y_cv1_um"] = corrected_y_cv1
    corrected["centroid_z_cv0_um"] = centroid_z_cv0
    corrected["centroid_z_cv1_um"] = centroid_z_cv1
    corrected["centroid_y_mean_um"] = np.nanmean(np.stack([corrected_y_cv0, corrected_y_cv1], axis=1), axis=1)
    corrected["centroid_cv_disagreement_um"] = np.abs(corrected_y_cv0 - corrected_y_cv1)
    corrected["trajectory_y_peak_cv0_um"] = corrected_traj_y_peak_cv0
    corrected["trajectory_y_peak_cv1_um"] = corrected_traj_y_peak_cv1
    corrected["trajectory_y_peak_mean_um"] = np.nanmean(
        np.stack([corrected_traj_y_peak_cv0, corrected_traj_y_peak_cv1], axis=1),
        axis=1,
    )
    corrected["max_site_cv0"] = max_site[:, 0]
    corrected["max_site_cv1"] = max_site[:, 1]

    corrected_path = FEATURE_ROOT / "al032_corrected_unit_feature_table.csv"
    uncorrected_path = FEATURE_ROOT / "al032_uncorrected_unit_feature_table.csv"
    shift_path = FEATURE_ROOT / "al032_geometry_shift_by_unit.csv"

    csv_write(corrected, corrected_path)
    csv_write(uncorrected, uncorrected_path)
    csv_write(
        merged[
            [
                "unitmatch_unit_index",
                "session_index",
                "session_name",
                "cluster_id",
                "n_good_localized_spikes",
                "median_y_um_before",
                "median_y_um_after",
                "median_y_shift_um",
                "mean_y_shift_um",
                "std_y_shift_um",
                "spread_before_um",
                "spread_after_um",
                "spread_ratio_after_over_before",
                "q05_y_shift_um",
                "q95_y_shift_um",
                "median_amplitude",
            ]
        ],
        shift_path,
    )

    dump_json(
        FEATURE_ROOT / "corrected_unit_feature_summary.json",
        {
            "created_at": now_iso(),
            "corrected_feature_table_csv": str(corrected_path),
            "uncorrected_feature_table_csv": str(uncorrected_path),
            "geometry_shift_by_unit_csv": str(shift_path),
            "n_units": int(corrected.shape[0]),
            "n_units_with_finite_shift": int(np.count_nonzero(np.isfinite(merged["median_y_shift_um"].to_numpy(dtype=np.float64)))),
            "median_abs_shift_um": float(np.nanmedian(np.abs(merged["median_y_shift_um"].to_numpy(dtype=np.float64)))),
        },
    )


if __name__ == "__main__":
    main()
