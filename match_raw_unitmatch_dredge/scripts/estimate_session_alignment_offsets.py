#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_utils import csv_write, dump_json, now_iso


def series_to_bool_mask(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        values = numeric.to_numpy(dtype=np.float64, copy=False)
        mask = np.zeros(values.shape[0], dtype=bool)
        finite = np.isfinite(values)
        mask[finite] = values[finite] != 0.0
        return mask

    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "1", "t", "yes", "y"}).fillna(False).to_numpy(dtype=bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/configs/unitmatch_dredge_run_config.json",
    )
    parser.add_argument(
        "--alignment-units-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/outputs/tracked_tables/high_confidence_shared_units_for_alignment.csv",
    )
    parser.add_argument("--reference-session", default="")
    parser.add_argument("--min-shared-units", type=int, default=3)
    return parser.parse_args()


def load_session_unit_medians(session_csv_gz: Path, session_name: str, tracked_unit_ids: set[int]) -> pd.DataFrame:
    usecols = [
        "tracked_unit_id",
        "conflict_flag",
        "is_good_cluster",
        "localization_success",
        "y_um_dredge",
    ]
    y_chunks: dict[int, list[np.ndarray]] = {}
    for chunk in pd.read_csv(session_csv_gz, compression="gzip", usecols=usecols, chunksize=250_000, low_memory=False):
        mask = (
            chunk["tracked_unit_id"].notna().to_numpy()
            & chunk["is_good_cluster"].astype(bool).to_numpy()
            & chunk["localization_success"].astype(bool).to_numpy()
            & np.isfinite(chunk["y_um_dredge"].to_numpy(dtype=np.float64))
        )
        if "conflict_flag" in chunk.columns:
            mask &= ~series_to_bool_mask(chunk["conflict_flag"])
        if not np.any(mask):
            continue
        sub = chunk.loc[mask, ["tracked_unit_id", "y_um_dredge"]].copy()
        sub["tracked_unit_id"] = sub["tracked_unit_id"].astype(int)
        sub = sub[sub["tracked_unit_id"].isin(tracked_unit_ids)]
        if sub.empty:
            continue
        for tracked_unit_id, group in sub.groupby("tracked_unit_id", sort=False):
            y_chunks.setdefault(int(tracked_unit_id), []).append(group["y_um_dredge"].to_numpy(dtype=np.float32))

    rows = []
    for tracked_unit_id in sorted(y_chunks):
        values = np.concatenate(y_chunks[tracked_unit_id]).astype(np.float64, copy=False)
        rows.append(
            {
                "session_name": session_name,
                "tracked_unit_id": int(tracked_unit_id),
                "good_tracked_spike_count": int(values.size),
                "median_y_um_dredge": float(np.median(values)),
                "std_y_um_dredge": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    corrected_spikes_root = Path(config["corrected_spikes_root"])
    tracked_root = Path(config["tracked_tables_root"])
    alignment_units = pd.read_csv(args.alignment_units_csv)
    tracked_unit_ids = set(alignment_units["tracked_unit_id"].astype(int).tolist())
    if not tracked_unit_ids:
        raise SystemExit("No eligible alignment units found.")

    reference_session = args.reference_session or config["session_names"][0]
    session_frames = []
    for session_name in config["session_names"]:
        session_csv_gz = corrected_spikes_root / session_name / f"{session_name}_tracked_spikes.csv.gz"
        session_frames.append(load_session_unit_medians(session_csv_gz, session_name, tracked_unit_ids))
    unit_depths = pd.concat(session_frames, ignore_index=True)
    unit_depths_csv = corrected_spikes_root / "session_alignment_unit_depths.csv"
    csv_write(unit_depths, unit_depths_csv)

    ref = unit_depths.loc[unit_depths["session_name"] == reference_session, ["tracked_unit_id", "median_y_um_dredge"]].copy()
    ref = ref.rename(columns={"median_y_um_dredge": "reference_median_y_um_dredge"})

    offset_rows = []
    pair_rows = []
    for session_name in config["session_names"]:
        current = unit_depths.loc[unit_depths["session_name"] == session_name].copy()
        merged = current.merge(ref, on="tracked_unit_id", how="inner")
        merged["offset_to_reference_um"] = (
            merged["reference_median_y_um_dredge"].to_numpy(dtype=np.float64)
            - merged["median_y_um_dredge"].to_numpy(dtype=np.float64)
        )
        if session_name == reference_session:
            session_offset = 0.0
        elif merged.shape[0] >= int(args.min_shared_units):
            session_offset = float(np.median(merged["offset_to_reference_um"].to_numpy(dtype=np.float64)))
        else:
            session_offset = 0.0
        pair_rows.append(merged.assign(reference_session=reference_session, session_offset_applied_um=session_offset))
        offset_rows.append(
            {
                "session_name": session_name,
                "reference_session_name": reference_session,
                "shared_alignment_unit_count": int(merged.shape[0]),
                "session_alignment_offset_um": float(session_offset),
                "median_pairwise_offset_um": float(np.median(merged["offset_to_reference_um"])) if not merged.empty else np.nan,
                "mad_pairwise_offset_um": float(
                    1.4826 * np.median(np.abs(merged["offset_to_reference_um"] - np.median(merged["offset_to_reference_um"])))
                )
                if not merged.empty
                else np.nan,
                "min_pairwise_offset_um": float(np.min(merged["offset_to_reference_um"])) if not merged.empty else np.nan,
                "max_pairwise_offset_um": float(np.max(merged["offset_to_reference_um"])) if not merged.empty else np.nan,
            }
        )

    offsets = pd.DataFrame(offset_rows)
    pairs = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    offsets_csv = corrected_spikes_root / "session_alignment_offsets.csv"
    pairs_csv = corrected_spikes_root / "session_alignment_pairwise_offsets.csv"
    csv_write(offsets, offsets_csv)
    csv_write(pairs, pairs_csv)

    dump_json(
        corrected_spikes_root / "session_alignment_offsets_summary.json",
        {
            "created_at": now_iso(),
            "reference_session_name": reference_session,
            "min_shared_units": int(args.min_shared_units),
            "alignment_units_csv": str(args.alignment_units_csv),
            "unit_depths_csv": str(unit_depths_csv),
            "offsets_csv": str(offsets_csv),
            "pairwise_offsets_csv": str(pairs_csv),
            "session_offsets": offsets.to_dict(orient="records"),
        },
    )


if __name__ == "__main__":
    main()
