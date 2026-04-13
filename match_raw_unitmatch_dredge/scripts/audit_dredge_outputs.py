#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_utils import csv_write, dump_json, now_iso


ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge")
UPSTREAM_ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch")
MANIFEST_PATH = ROOT / "manifests" / "al032_dredge_input_manifest.csv"
DREDGE_ROOT = ROOT / "outputs" / "dredge_per_session"
ATTACHED_SPIKES_ROOT = UPSTREAM_ROOT / "outputs" / "attached_spikes"
SELECTED_UNITS_PATH = UPSTREAM_ROOT / "outputs" / "tracked_tables" / "selected_tracked_units.csv"


def robust_spread(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))


def load_tracking(session_name: str) -> pd.DataFrame:
    path = ATTACHED_SPIKES_ROOT / session_name / f"{session_name}_tracked_spikes.csv.gz"
    return pd.read_csv(
        path,
        compression="gzip",
        usecols=["source_spike_index", "cluster_id", "tracked_unit_id", "conflict_flag"],
        low_memory=False,
    )


def audit_session(session_name: str, selected_units_df: pd.DataFrame) -> dict[str, object]:
    session_root = DREDGE_ROOT / session_name
    corrected_path = session_root / f"{session_name}_corrected_spike_table.csv.gz"
    corrected_df = pd.read_csv(corrected_path, compression="gzip", low_memory=False)
    tracking_df = load_tracking(session_name)
    df = corrected_df.merge(tracking_df, on=["source_spike_index", "cluster_id"], how="left")
    stable_palette = selected_units_df.sort_values("selection_rank")

    valid_mask = (
        df["dredge_applied"].astype(bool).to_numpy()
        & df["is_good_cluster"].astype(bool).to_numpy()
        & df["tracked_unit_id"].notna().to_numpy()
        & np.isfinite(df["y_um"].to_numpy(dtype=np.float64))
        & np.isfinite(df["y_um_dredge"].to_numpy(dtype=np.float64))
    )
    eval_df = df.loc[valid_mask].copy()
    eval_df["tracked_unit_id"] = eval_df["tracked_unit_id"].astype(int)
    eval_df = eval_df[eval_df["tracked_unit_id"].isin(set(stable_palette["tracked_unit_id"].astype(int).tolist()))].copy()

    unit_rows = []
    for row in stable_palette.itertuples(index=False):
        unit_df = eval_df[eval_df["tracked_unit_id"] == int(row.tracked_unit_id)].copy()
        if len(unit_df) < 500:
            continue
        spread_before = robust_spread(unit_df["y_um"].to_numpy(dtype=np.float64))
        spread_after = robust_spread(unit_df["y_um_dredge"].to_numpy(dtype=np.float64))
        unit_rows.append(
            {
                "session_name": session_name,
                "tracked_unit_id": int(row.tracked_unit_id),
                "tracked_label": row.tracked_label,
                "color_hex": row.color_hex,
                "n_spikes": int(len(unit_df)),
                "spread_before_um": float(spread_before),
                "spread_after_um": float(spread_after),
                "spread_ratio_after_over_before": float(spread_after / spread_before) if spread_before > 0 else np.nan,
                "median_depth_before_um": float(np.median(unit_df["y_um"].to_numpy(dtype=np.float64))),
                "median_depth_after_um": float(np.median(unit_df["y_um_dredge"].to_numpy(dtype=np.float64))),
            }
        )

    unit_df = pd.DataFrame(unit_rows)
    csv_write(unit_df, session_root / f"{session_name}_stable_unit_spread_comparison.csv")

    displacement = df.loc[df["dredge_applied"].astype(bool), "dredge_peak_displacement_um"].to_numpy(dtype=np.float64)
    abs_disp = np.abs(displacement[np.isfinite(displacement)])
    if unit_df.empty:
        median_ratio = float("nan")
        improved_unit_count = 0
        worsened_unit_count = 0
    else:
        ratios = unit_df["spread_ratio_after_over_before"].to_numpy(dtype=np.float64)
        median_ratio = float(np.nanmedian(ratios))
        improved_unit_count = int(np.sum(ratios < 1.0))
        worsened_unit_count = int(np.sum(ratios > 1.0))

    qc = {
        "created_at": now_iso(),
        "session_name": session_name,
        "mean_abs_displacement_um": float(abs_disp.mean()) if abs_disp.size else np.nan,
        "max_abs_displacement_um": float(abs_disp.max()) if abs_disp.size else np.nan,
        "q95_abs_displacement_um": float(np.quantile(abs_disp, 0.95)) if abs_disp.size else np.nan,
        "stable_unit_count_evaluated": int(len(unit_df)),
        "stable_units_evaluated": unit_df[["tracked_unit_id", "tracked_label"]].to_dict(orient="records"),
        "median_spread_before_um": float(unit_df["spread_before_um"].median()) if not unit_df.empty else np.nan,
        "median_spread_after_um": float(unit_df["spread_after_um"].median()) if not unit_df.empty else np.nan,
        "median_spread_ratio_after_over_before": median_ratio,
        "improved_unit_count": improved_unit_count,
        "worsened_unit_count": worsened_unit_count,
        "raster_visibly_tightens_bands": bool(np.isfinite(median_ratio) and median_ratio < 0.98),
        "session_got_worse": bool(np.isfinite(median_ratio) and median_ratio > 1.02),
        "spread_comparison_csv": str(session_root / f"{session_name}_stable_unit_spread_comparison.csv"),
        "corrected_spike_table_csv_gz": str(session_root / f"{session_name}_corrected_spike_table.csv.gz"),
        "before_after_raster_png": str(session_root / f"{session_name}_before_after_dredge_raster.png"),
    }
    dump_json(session_root / f"{session_name}_dredge_qc_summary.json", qc)
    return qc


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH).sort_values("session_order")
    selected_units_df = pd.read_csv(SELECTED_UNITS_PATH).sort_values("selection_rank")
    rows = []
    for row in manifest.itertuples(index=False):
        rows.append(audit_session(row.session_name, selected_units_df))
    all_df = pd.DataFrame(rows)
    csv_write(all_df, DREDGE_ROOT / "al032_dredge_audit_summary.csv")
    payload = {
        "created_at": now_iso(),
        "session_count": int(len(all_df)),
        "sessions_with_visible_tightening": all_df.loc[
            all_df["raster_visibly_tightens_bands"].astype(bool), "session_name"
        ].tolist(),
        "sessions_got_worse": all_df.loc[all_df["session_got_worse"].astype(bool), "session_name"].tolist(),
        "median_of_session_median_spread_ratio": float(
            np.nanmedian(all_df["median_spread_ratio_after_over_before"].to_numpy(dtype=np.float64))
        ),
        "mean_of_session_mean_abs_displacement_um": float(
            np.nanmean(all_df["mean_abs_displacement_um"].to_numpy(dtype=np.float64))
        ),
        "max_session_abs_displacement_um": float(
            np.nanmax(all_df["max_abs_displacement_um"].to_numpy(dtype=np.float64))
        ),
        "per_session_rows": all_df.to_dict(orient="records"),
        "summary_csv": str(DREDGE_ROOT / "al032_dredge_audit_summary.csv"),
    }
    dump_json(DREDGE_ROOT / "al032_dredge_audit_summary.json", payload)


if __name__ == "__main__":
    main()
