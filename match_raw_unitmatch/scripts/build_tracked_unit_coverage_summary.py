#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_utils import csv_write, dump_json, now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AL032 12-session tracked unit coverage summary.")
    parser.add_argument(
        "--config",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/configs/unitmatch_run_config.json",
    )
    parser.add_argument("--min-sessions-present", type=int, default=10)
    parser.add_argument("--min-good-tracked-spikes", type=int, default=1000)
    parser.add_argument("--min-mean-probability", type=float, default=0.60)
    parser.add_argument("--select-count", type=int, default=10)
    return parser.parse_args()


def cluster_color(cluster_id: int) -> tuple[int, int, int]:
    import colorsys

    hue = (cluster_id * 0.61803398875) % 1.0
    sat = 0.72
    val = 0.96
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


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


def compute_session_stats(
    session_name: str,
    session_index: int,
    attached_csv_gz: Path,
    cluster_map: pd.DataFrame,
) -> pd.DataFrame:
    tracked_counts: dict[int, int] = {}
    localized_counts: dict[int, int] = {}
    good_counts: dict[int, int] = {}
    y_chunks_by_unit: dict[int, list[np.ndarray]] = {}

    usecols = [
        "tracked_unit_id",
        "conflict_flag",
        "is_good_cluster",
        "localization_success",
        "y_um",
    ]
    dtypes = {
        "tracked_unit_id": "float64",
        "is_good_cluster": "bool",
        "localization_success": "bool",
        "y_um": "float64",
    }

    for chunk in pd.read_csv(
        attached_csv_gz,
        compression="gzip",
        usecols=usecols,
        dtype=dtypes,
        chunksize=250_000,
        low_memory=False,
    ):
        tracked_mask = chunk["tracked_unit_id"].notna().to_numpy()
        if "conflict_flag" in chunk.columns:
            conflict = series_to_bool_mask(chunk["conflict_flag"])
            tracked_mask &= ~conflict
        if not np.any(tracked_mask):
            continue

        tracked = chunk.loc[tracked_mask, ["tracked_unit_id", "is_good_cluster", "localization_success", "y_um"]].copy()
        tracked["tracked_unit_id"] = tracked["tracked_unit_id"].astype(np.int64)

        for tracked_unit_id, count in tracked.groupby("tracked_unit_id").size().items():
            tracked_counts[int(tracked_unit_id)] = tracked_counts.get(int(tracked_unit_id), 0) + int(count)

        localized_mask = tracked["localization_success"].to_numpy(dtype=bool) & np.isfinite(
            tracked["y_um"].to_numpy(dtype=np.float64)
        )
        if np.any(localized_mask):
            localized = tracked.loc[localized_mask].copy()
            for tracked_unit_id, count in localized.groupby("tracked_unit_id").size().items():
                localized_counts[int(tracked_unit_id)] = localized_counts.get(int(tracked_unit_id), 0) + int(count)

            good = localized.loc[localized["is_good_cluster"].to_numpy(dtype=bool)].copy()
            if not good.empty:
                for tracked_unit_id, count in good.groupby("tracked_unit_id").size().items():
                    good_counts[int(tracked_unit_id)] = good_counts.get(int(tracked_unit_id), 0) + int(count)
                for tracked_unit_id, group in good.groupby("tracked_unit_id", sort=False):
                    y_chunks_by_unit.setdefault(int(tracked_unit_id), []).append(
                        group["y_um"].to_numpy(dtype=np.float32)
                    )

    rows: list[dict[str, object]] = []
    cluster_lookup = {
        int(row.tracked_unit_id): int(row.cluster_id)
        for row in cluster_map.itertuples(index=False)
    }
    tracked_unit_ids = sorted(set(cluster_lookup) | set(tracked_counts) | set(localized_counts) | set(good_counts))
    for tracked_unit_id in tracked_unit_ids:
        y_values = (
            np.concatenate(y_chunks_by_unit.get(tracked_unit_id, []))
            if tracked_unit_id in y_chunks_by_unit
            else np.empty(0, dtype=np.float32)
        )
        rows.append(
            {
                "session_name": session_name,
                "session_index": session_index,
                "tracked_unit_id": int(tracked_unit_id),
                "raw_cluster_id": cluster_lookup.get(int(tracked_unit_id), np.nan),
                "tracked_spike_count": int(tracked_counts.get(int(tracked_unit_id), 0)),
                "localized_tracked_spike_count": int(localized_counts.get(int(tracked_unit_id), 0)),
                "good_tracked_spike_count": int(good_counts.get(int(tracked_unit_id), 0)),
                "good_depth_median_um": float(np.median(y_values)) if y_values.size else np.nan,
                "good_depth_std_um": float(np.std(y_values, ddof=1)) if y_values.size > 1 else (0.0 if y_values.size == 1 else np.nan),
                "good_depth_min_um": float(np.min(y_values)) if y_values.size else np.nan,
                "good_depth_max_um": float(np.max(y_values)) if y_values.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def json_map_from_group(group: pd.DataFrame, key_col: str, value_col: str) -> str:
    payload = {}
    for row in group.itertuples(index=False):
        value = getattr(row, value_col)
        if pd.isna(value):
            continue
        payload[str(getattr(row, key_col))] = float(value) if isinstance(value, (np.floating, float)) else int(value)
    return json.dumps(payload, sort_keys=True)


def build_coverage_summary(
    tracked_summary: pd.DataFrame,
    session_stats: pd.DataFrame,
    min_sessions_present: int,
    min_good_tracked_spikes: int,
    min_mean_probability: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    long_groups = {int(k): g.copy() for k, g in session_stats.groupby("tracked_unit_id", sort=False)}
    for row in tracked_summary.itertuples(index=False):
        tracked_unit_id = int(row.tracked_unit_id)
        group = long_groups.get(tracked_unit_id, pd.DataFrame(columns=session_stats.columns))

        good_counts = group["good_tracked_spike_count"].to_numpy(dtype=np.int64) if not group.empty else np.empty(0, dtype=np.int64)
        localized_counts = (
            group["localized_tracked_spike_count"].to_numpy(dtype=np.int64) if not group.empty else np.empty(0, dtype=np.int64)
        )
        tracked_counts = group["tracked_spike_count"].to_numpy(dtype=np.int64) if not group.empty else np.empty(0, dtype=np.int64)
        depth_medians = (
            group["good_depth_median_um"].dropna().to_numpy(dtype=np.float64) if not group.empty else np.empty(0, dtype=np.float64)
        )

        n_sessions_with_good = int(np.count_nonzero(good_counts > 0))
        min_good = int(good_counts.min()) if good_counts.size else 0
        total_good = int(good_counts.sum()) if good_counts.size else 0
        median_good = float(np.median(good_counts)) if good_counts.size else 0.0
        depth_center = float(np.median(depth_medians)) if depth_medians.size else np.nan
        depth_stability_std = float(np.std(depth_medians, ddof=1)) if depth_medians.size > 1 else (0.0 if depth_medians.size == 1 else np.nan)
        depth_stability_max_abs_diff = (
            float(np.max(depth_medians) - np.min(depth_medians)) if depth_medians.size else np.nan
        )

        eligible = (
            bool(row.conflict_free_validity_flag)
            and int(row.n_sessions_present) >= int(min_sessions_present)
            and n_sessions_with_good >= int(min_sessions_present)
            and min_good >= int(min_good_tracked_spikes)
            and float(row.mean_cross_session_probability) >= float(min_mean_probability)
        )

        exclusion_reasons: list[str] = []
        if not bool(row.conflict_free_validity_flag):
            exclusion_reasons.append("conflict")
        if int(row.n_sessions_present) < int(min_sessions_present):
            exclusion_reasons.append("too_few_sessions")
        if n_sessions_with_good < int(min_sessions_present):
            exclusion_reasons.append("too_few_good_sessions")
        if min_good < int(min_good_tracked_spikes):
            exclusion_reasons.append("low_good_spike_count")
        if float(row.mean_cross_session_probability) < float(min_mean_probability):
            exclusion_reasons.append("low_mean_probability")

        rows.append(
            {
                "tracked_unit_id": tracked_unit_id,
                "n_sessions_present": int(row.n_sessions_present),
                "sessions_present": row.sessions_present,
                "cluster_ids_by_session": row.cluster_ids_by_session,
                "conflict_free_validity_flag": bool(row.conflict_free_validity_flag),
                "mean_cross_session_probability": float(row.mean_cross_session_probability),
                "max_cross_session_probability": float(row.max_cross_session_probability),
                "n_sessions_with_good_tracked_spikes": n_sessions_with_good,
                "tracked_spike_counts_by_session": json_map_from_group(group, "session_name", "tracked_spike_count"),
                "localized_tracked_spike_counts_by_session": json_map_from_group(group, "session_name", "localized_tracked_spike_count"),
                "good_tracked_spike_counts_by_session": json_map_from_group(group, "session_name", "good_tracked_spike_count"),
                "median_localized_depth_um_by_session": json_map_from_group(group, "session_name", "good_depth_median_um"),
                "depth_std_um_by_session": json_map_from_group(group, "session_name", "good_depth_std_um"),
                "total_tracked_spikes": int(tracked_counts.sum()) if tracked_counts.size else 0,
                "total_localized_tracked_spikes": int(localized_counts.sum()) if localized_counts.size else 0,
                "total_good_tracked_spikes": total_good,
                "min_good_tracked_spikes": min_good,
                "median_good_tracked_spikes": median_good,
                "depth_center_um": depth_center,
                "depth_stability_std_um": depth_stability_std,
                "depth_stability_max_abs_diff_um": depth_stability_max_abs_diff,
                "selection_eligible": bool(eligible),
                "exclusion_reason": ";".join(exclusion_reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selection_eligible", "n_sessions_present", "min_good_tracked_spikes", "mean_cross_session_probability", "tracked_unit_id"],
        ascending=[False, False, False, False, True],
    )


def select_depth_diverse_units(coverage_summary: pd.DataFrame, select_count: int) -> pd.DataFrame:
    eligible = coverage_summary.loc[coverage_summary["selection_eligible"]].copy()
    if eligible.empty:
        return eligible

    selection_target = min(int(select_count), int(eligible.shape[0]))
    eligible["selection_score"] = eligible["min_good_tracked_spikes"]
    eligible["selection_tiebreak"] = eligible["total_good_tracked_spikes"]
    eligible["depth_alignment_score"] = -eligible["depth_stability_max_abs_diff_um"].fillna(0.0)

    chosen_ids: list[int] = []
    depth_values = eligible["depth_center_um"].to_numpy(dtype=np.float64)
    if selection_target == 1:
        best = eligible.sort_values(
            ["n_sessions_present", "selection_score", "selection_tiebreak", "mean_cross_session_probability", "depth_alignment_score"],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        chosen_ids = [int(best["tracked_unit_id"])]
    else:
        depth_min = float(np.nanmin(depth_values))
        depth_max = float(np.nanmax(depth_values))
        bin_edges = np.linspace(depth_min, depth_max, selection_target + 1)
        used: set[int] = set()
        for idx in range(selection_target):
            lo = bin_edges[idx]
            hi = bin_edges[idx + 1]
            if idx == selection_target - 1:
                mask = (eligible["depth_center_um"] >= lo) & (eligible["depth_center_um"] <= hi)
            else:
                mask = (eligible["depth_center_um"] >= lo) & (eligible["depth_center_um"] < hi)
            candidates = eligible.loc[mask & ~eligible["tracked_unit_id"].isin(used)].copy()
            if candidates.empty:
                continue
            best = candidates.sort_values(
                ["n_sessions_present", "selection_score", "selection_tiebreak", "mean_cross_session_probability", "depth_alignment_score"],
                ascending=[False, False, False, False, False],
            ).iloc[0]
            tracked_unit_id = int(best["tracked_unit_id"])
            chosen_ids.append(tracked_unit_id)
            used.add(tracked_unit_id)

        if len(chosen_ids) < selection_target:
            remaining = eligible.loc[~eligible["tracked_unit_id"].isin(chosen_ids)].copy()
            remaining = remaining.sort_values(
                ["n_sessions_present", "selection_score", "selection_tiebreak", "mean_cross_session_probability", "depth_alignment_score"],
                ascending=[False, False, False, False, False],
            )
            chosen_ids.extend(int(value) for value in remaining["tracked_unit_id"].head(selection_target - len(chosen_ids)))

    selected = eligible.loc[eligible["tracked_unit_id"].isin(chosen_ids)].copy()
    selected = selected.sort_values("depth_center_um").reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)

    color_hex: list[str] = []
    color_r: list[int] = []
    color_g: list[int] = []
    color_b: list[int] = []
    tracked_label: list[str] = []
    for tracked_unit_id in selected["tracked_unit_id"].to_numpy(dtype=np.int64):
        r, g, b = cluster_color(int(tracked_unit_id))
        color_hex.append(f"#{r:02x}{g:02x}{b:02x}")
        color_r.append(r)
        color_g.append(g)
        color_b.append(b)
        tracked_label.append(f"T{int(tracked_unit_id)}")
    selected["tracked_label"] = tracked_label
    selected["color_hex"] = color_hex
    selected["color_r"] = color_r
    selected["color_g"] = color_g
    selected["color_b"] = color_b
    return selected


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    tracked_root = Path(config["tracked_tables_root"])
    attached_root = Path(config["attached_spikes_root"])
    tracked_summary = pd.read_csv(tracked_root / "tracked_unit_summary.csv")
    cluster_to_tracked = pd.read_csv(tracked_root / "cluster_to_tracked_unit.csv")
    valid_mapping = cluster_to_tracked.loc[~series_to_bool_mask(cluster_to_tracked["conflict_flag"])].copy()

    session_stats_frames: list[pd.DataFrame] = []
    for session_index, session_name in enumerate(config["session_names"], start=1):
        attached_csv_gz = attached_root / session_name / f"{session_name}_tracked_spikes.csv.gz"
        cluster_map = valid_mapping.loc[valid_mapping["session_name"] == session_name, ["tracked_unit_id", "cluster_id"]].copy()
        session_stats_frames.append(
            compute_session_stats(
                session_name=session_name,
                session_index=session_index,
                attached_csv_gz=attached_csv_gz,
                cluster_map=cluster_map,
            )
        )

    session_stats = pd.concat(session_stats_frames, ignore_index=True)
    coverage_summary = build_coverage_summary(
        tracked_summary=tracked_summary,
        session_stats=session_stats,
        min_sessions_present=args.min_sessions_present,
        min_good_tracked_spikes=args.min_good_tracked_spikes,
        min_mean_probability=args.min_mean_probability,
    )
    selected = select_depth_diverse_units(coverage_summary, args.select_count)

    session_stats_path = tracked_root / "tracked_unit_session_summary.csv"
    coverage_path = tracked_root / "tracked_unit_coverage_summary.csv"
    selected_path = tracked_root / "selected_tracked_units.csv"
    csv_write(session_stats, session_stats_path)
    csv_write(coverage_summary, coverage_path)
    csv_write(selected, selected_path)

    dump_json(
        tracked_root / "tracked_unit_coverage_summary.json",
        {
            "created_at": now_iso(),
            "min_sessions_present": int(args.min_sessions_present),
            "min_good_tracked_spikes": int(args.min_good_tracked_spikes),
            "min_mean_probability": float(args.min_mean_probability),
            "select_count": int(args.select_count),
            "session_stats_csv": str(session_stats_path),
            "coverage_summary_csv": str(coverage_path),
            "selected_tracked_units_csv": str(selected_path),
            "n_tracked_units_total": int(coverage_summary.shape[0]),
            "n_selection_eligible": int(np.count_nonzero(coverage_summary["selection_eligible"])),
            "selected_tracked_unit_ids": [int(v) for v in selected["tracked_unit_id"].tolist()],
        },
    )


if __name__ == "__main__":
    main()
