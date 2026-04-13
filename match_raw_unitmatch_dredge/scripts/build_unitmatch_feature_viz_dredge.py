#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import pickle
from datetime import datetime
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/tmp/mplconfig"

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge")
UNITMATCH_ROOT = ROOT / "outputs" / "unitmatch_dredge_12session"
FEATURE_VIZ_ROOT = UNITMATCH_ROOT / "feature_viz"
TRACKED_ROOT = ROOT / "outputs" / "tracked_tables"


FEATURE_SPECS = [
    {"name": "centroid_x_um", "family": "spatial", "label": "Centroid X (um)", "definition": "Mean x-coordinate of the UnitMatch centroid across the two cross-validation halves.", "interpretation": "Spatial lateral position."},
    {"name": "centroid_y_um", "family": "spatial", "label": "Centroid Y (um)", "definition": "Mean y-coordinate of the UnitMatch centroid across the two cross-validation halves.", "interpretation": "Depth-like coordinate."},
    {"name": "centroid_z_um", "family": "spatial", "label": "Centroid Z (um)", "definition": "Mean z-coordinate of the UnitMatch centroid across the two cross-validation halves.", "interpretation": "Third centroid axis."},
    {"name": "centroid_cv_disagreement_um", "family": "spatial", "label": "Centroid CV disagreement (um)", "definition": "Euclidean distance between centroid estimates from the two CV halves.", "interpretation": "Internal centroid consistency; lower is better."},
    {"name": "max_site_mean_index", "family": "spatial", "label": "Mean max-site index", "definition": "Average of the peak-channel indices across the two CV halves.", "interpretation": "Channel-index proxy for footprint peak."},
    {"name": "max_site_cv_gap", "family": "spatial", "label": "Max-site CV gap", "definition": "Absolute difference in peak-channel index between the two CV halves.", "interpretation": "Peak-channel consistency; lower is better."},
    {"name": "waveform_trough_raw", "family": "waveform", "label": "Waveform trough (raw)", "definition": "Minimum value of the average waveform after averaging across CV halves.", "interpretation": "Negative waveform amplitude in raw units."},
    {"name": "waveform_peak_raw", "family": "waveform", "label": "Waveform peak (raw)", "definition": "Maximum value of the average waveform after averaging across CV halves.", "interpretation": "Positive waveform amplitude in raw units."},
    {"name": "waveform_ptp_raw", "family": "waveform", "label": "Waveform peak-to-trough (raw)", "definition": "Waveform peak minus waveform trough.", "interpretation": "Overall waveform magnitude in raw units."},
    {"name": "waveform_energy_raw", "family": "waveform", "label": "Waveform energy (raw)", "definition": "Sum of squared values of the average waveform.", "interpretation": "Energy-like summary of waveform magnitude."},
    {"name": "waveform_cv_corr", "family": "waveform", "label": "Waveform CV correlation", "definition": "Correlation between the two CV average waveforms.", "interpretation": "Internal waveform consistency; closer to 1 is better."},
    {"name": "trough_to_peak_samples", "family": "waveform", "label": "Trough-to-peak (samples)", "definition": "Number of samples from waveform trough to the next positive peak.", "interpretation": "Waveform timing feature."},
    {"name": "active_timepoint_count", "family": "trajectory", "label": "Active trajectory timepoints", "definition": "Count of waveform timepoints with valid projected trajectory coordinates.", "interpretation": "How much of the waveform has a usable trajectory."},
    {"name": "trajectory_total_path_um", "family": "trajectory", "label": "Trajectory total path (um)", "definition": "Total path length of the mean waveform trajectory over time.", "interpretation": "How much the projected center moves over the waveform duration."},
    {"name": "trajectory_displacement_um", "family": "trajectory", "label": "Trajectory displacement (um)", "definition": "Straight-line distance from first to last valid point in the mean trajectory.", "interpretation": "Net movement of the trajectory."},
    {"name": "trajectory_y_span_um", "family": "trajectory", "label": "Trajectory Y span (um)", "definition": "Maximum y minus minimum y along the mean trajectory.", "interpretation": "Depth span of the within-waveform trajectory."},
    {"name": "trajectory_cv_distance_um", "family": "trajectory", "label": "Trajectory CV distance (um)", "definition": "Mean distance between the two CV trajectories at overlapping valid timepoints.", "interpretation": "Internal trajectory consistency; lower is better."},
]


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a_m = a[mask]
    b_m = b[mask]
    a_s = a_m.std()
    b_s = b_m.std()
    if a_s == 0 or b_s == 0:
        return float("nan")
    return float(np.corrcoef(a_m, b_m)[0, 1])


def nanmean_no_warn(arr: np.ndarray, axis: int) -> np.ndarray:
    valid_count = np.sum(np.isfinite(arr), axis=axis)
    summed = np.nansum(arr, axis=axis)
    out = np.full_like(summed, np.nan, dtype=float)
    np.divide(summed, valid_count, out=out, where=valid_count > 0)
    return out


def mean_trajectory(traj_unit: np.ndarray) -> np.ndarray:
    traj = np.moveaxis(traj_unit, 0, -1)
    return nanmean_no_warn(traj, axis=1)


def finite_rows(arr: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(arr), axis=1)


def derive_unit_features(unit_idx: int, wave_info: dict) -> dict[str, float]:
    avg_centroid = wave_info["avg_centroid"][:, unit_idx, :]
    avg_waveform = wave_info["avg_waveform"][:, unit_idx, :]
    avg_waveform_per_tp = wave_info["avg_waveform_per_tp"][:, unit_idx, :, :]
    max_site = wave_info["max_site"][unit_idx, :]

    centroid_mean = np.nanmean(avg_centroid, axis=1)
    centroid_cv_disagreement = float(
        np.linalg.norm(avg_centroid[:, 0] - avg_centroid[:, 1])
    ) if np.all(np.isfinite(avg_centroid)) else float("nan")

    max_site_mean_index = float(np.mean(max_site))
    max_site_cv_gap = float(abs(int(max_site[0]) - int(max_site[1])))

    waveform_mean = nanmean_no_warn(avg_waveform, axis=1)
    waveform_trough_raw = float(np.nanmin(waveform_mean))
    waveform_peak_raw = float(np.nanmax(waveform_mean))
    waveform_ptp_raw = waveform_peak_raw - waveform_trough_raw
    waveform_energy_raw = float(np.nansum(waveform_mean**2))
    waveform_cv_corr = safe_corr(avg_waveform[:, 0], avg_waveform[:, 1])

    trough_idx = int(np.nanargmin(waveform_mean))
    if trough_idx < len(waveform_mean) - 1:
        peak_after = int(trough_idx + np.nanargmax(waveform_mean[trough_idx:]))
    else:
        peak_after = trough_idx
    trough_to_peak_samples = float(peak_after - trough_idx)

    traj_mean = mean_trajectory(avg_waveform_per_tp)
    valid_mean = finite_rows(traj_mean)
    active_timepoint_count = float(valid_mean.sum())
    if valid_mean.sum() >= 2:
        valid_pts = traj_mean[valid_mean]
        trajectory_total_path_um = float(np.linalg.norm(np.diff(valid_pts, axis=0), axis=1).sum())
        trajectory_displacement_um = float(np.linalg.norm(valid_pts[-1] - valid_pts[0]))
        trajectory_y_span_um = float(np.nanmax(valid_pts[:, 1]) - np.nanmin(valid_pts[:, 1]))
    else:
        trajectory_total_path_um = float("nan")
        trajectory_displacement_um = float("nan")
        trajectory_y_span_um = float("nan")

    traj_cv0 = np.moveaxis(avg_waveform_per_tp[:, :, 0], 0, -1)
    traj_cv1 = np.moveaxis(avg_waveform_per_tp[:, :, 1], 0, -1)
    valid_overlap = finite_rows(traj_cv0) & finite_rows(traj_cv1)
    if valid_overlap.any():
        trajectory_cv_distance_um = float(
            np.linalg.norm(traj_cv0[valid_overlap] - traj_cv1[valid_overlap], axis=1).mean()
        )
    else:
        trajectory_cv_distance_um = float("nan")

    return {
        "centroid_x_um": float(centroid_mean[0]),
        "centroid_y_um": float(centroid_mean[1]),
        "centroid_z_um": float(centroid_mean[2]),
        "centroid_cv_disagreement_um": centroid_cv_disagreement,
        "max_site_mean_index": max_site_mean_index,
        "max_site_cv_gap": max_site_cv_gap,
        "waveform_trough_raw": waveform_trough_raw,
        "waveform_peak_raw": waveform_peak_raw,
        "waveform_ptp_raw": waveform_ptp_raw,
        "waveform_energy_raw": waveform_energy_raw,
        "waveform_cv_corr": waveform_cv_corr,
        "trough_to_peak_samples": trough_to_peak_samples,
        "active_timepoint_count": active_timepoint_count,
        "trajectory_total_path_um": trajectory_total_path_um,
        "trajectory_displacement_um": trajectory_displacement_um,
        "trajectory_y_span_um": trajectory_y_span_um,
        "trajectory_cv_distance_um": trajectory_cv_distance_um,
    }


def make_feature_inventory(um_scores: dict, wave_info: dict, match_table_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, value in um_scores.items():
        rows.append(
            {
                "feature_name": key,
                "source_file": "UM Scores.npz",
                "source_key": key,
                "shape": "x".join(str(x) for x in value.shape),
                "feature_kind": "pairwise",
                "usable_for_session_timecourse": False,
                "notes": "Pairwise UnitMatch score matrix; not used directly for per-unit session trajectories.",
            }
        )
    for key, value in wave_info.items():
        rows.append(
            {
                "feature_name": key,
                "source_file": "WaveformInfo.npz",
                "source_key": key,
                "shape": "x".join(str(x) for x in value.shape),
                "feature_kind": "per_unit_array",
                "usable_for_session_timecourse": True,
                "notes": "Native per-unit waveform/centroid array; scalar trajectory features are derived from this array.",
            }
        )
    pairwise_cols = [
        "UM Probabilities",
        "TotalScore",
        "amp_score",
        "spatial_decay_score",
        "centroid_overlord_score",
        "centroid_dist",
        "waveform_score",
        "trajectory_score",
    ]
    for key in pairwise_cols:
        if key in match_table_columns:
            rows.append(
                {
                    "feature_name": key,
                    "source_file": "MatchTable.csv",
                    "source_key": key,
                    "shape": f"{len(match_table_columns)} columns / pairwise rows",
                    "feature_kind": "pairwise_table_column",
                    "usable_for_session_timecourse": False,
                    "notes": "Pairwise match-score column in MatchTable; useful for pairwise evaluation, not for per-unit feature trajectories.",
                }
            )
    for spec in FEATURE_SPECS:
        rows.append(
            {
                "feature_name": spec["name"],
                "source_file": "WaveformInfo.npz",
                "source_key": "derived",
                "shape": "scalar per unit-session",
                "feature_kind": spec["family"],
                "usable_for_session_timecourse": True,
                "notes": spec["definition"],
            }
        )
    return pd.DataFrame(rows)


def draw_feature_panels(
    df: pd.DataFrame,
    feature_specs: list[dict[str, str]],
    output_path: Path,
    drift: bool = False,
    labeled: bool = False,
):
    n_features = len(feature_specs)
    ncols = 4
    nrows = math.ceil(n_features / ncols)
    fig_height = 4.6 * nrows + (1.0 if labeled else 0.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, fig_height), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    unit_ids = sorted(df["tracked_unit_id"].dropna().unique())
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(unit_ids)))
    color_map = {uid: colors[i] for i, uid in enumerate(unit_ids)}
    legend_handles = []
    legend_labels = []

    for ax, spec in zip(axes, feature_specs):
        name = spec["name"]
        sub = df[df["feature_name"] == name].copy()
        if drift:
            x_col = "days_since_first_appearance"
            y_col = "feature_drift_from_first"
            xlabel = "Days since first appearance"
        else:
            x_col = "session_date_dt"
            y_col = "feature_value"
            xlabel = "Session date"

        for uid, grp in sub.groupby("tracked_unit_id"):
            grp = grp.sort_values(x_col)
            line, = ax.plot(
                grp[x_col],
                grp[y_col],
                color=color_map[uid],
                linewidth=1.0,
                alpha=0.55,
                marker="o",
                markersize=2.5,
            )
            if name == feature_specs[0]["name"]:
                legend_handles.append(line)
                legend_labels.append(f"T{uid}")

        if drift:
            stat = sub.groupby("days_since_first_appearance")[y_col].agg(
                median="median",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
            ).reset_index().sort_values("days_since_first_appearance")
            ax.fill_between(stat["days_since_first_appearance"], stat["q1"], stat["q3"], color="black", alpha=0.12)
            ax.plot(stat["days_since_first_appearance"], stat["median"], color="black", linewidth=2.6)
        else:
            stat = sub.groupby("session_date_dt")[y_col].agg(
                median="median",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
            ).reset_index().sort_values("session_date_dt")
            ax.fill_between(stat["session_date_dt"], stat["q1"], stat["q3"], color="black", alpha=0.12)
            ax.plot(stat["session_date_dt"], stat["median"], color="black", linewidth=2.6)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.tick_params(axis="x", rotation=45)

        panel_title = spec["label"] if not labeled else f"{spec['label']}\n{spec['interpretation']}"
        ax.set_title(panel_title, fontsize=10 if labeled else 11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Feature value", fontsize=9)
        ax.grid(alpha=0.18, linewidth=0.5)

    for ax in axes[n_features:]:
        ax.axis("off")

    title = "UnitMatch per-unit feature trajectories"
    if drift:
        title += " (delta from first appearance)"
    else:
        title += " (raw values)"
    fig.suptitle(title, fontsize=18, y=1.01)
    if labeled:
        fig.text(
            0.5,
            0.992,
            "Colored thin lines = tracked units | thick black line = median across selected units | gray band = IQR",
            ha="center",
            va="top",
            fontsize=10,
            color="dimgray",
        )
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=7,
            frameon=False,
            fontsize=9,
            title="Tracked units",
            title_fontsize=10,
        )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_color_legend_table(selected_long_lived: pd.DataFrame) -> pd.DataFrame:
    unit_ids = sorted(selected_long_lived["tracked_unit_id"].astype(int).tolist())
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(unit_ids)))
    rows = []
    for idx, uid in enumerate(unit_ids):
        rgb = colors[idx][:3]
        subset = selected_long_lived[selected_long_lived["tracked_unit_id"] == uid].iloc[0]
        rows.append(
            {
                "tracked_unit_id": uid,
                "tracked_label": f"T{uid}",
                "color_hex": "#{:02x}{:02x}{:02x}".format(
                    int(round(rgb[0] * 255)),
                    int(round(rgb[1] * 255)),
                    int(round(rgb[2] * 255)),
                ),
                "color_r": int(round(rgb[0] * 255)),
                "color_g": int(round(rgb[1] * 255)),
                "color_b": int(round(rgb[2] * 255)),
                "n_sessions_present": int(subset["n_sessions_present"]),
                "mean_cross_session_probability": float(subset["mean_cross_session_probability"]),
                "depth_center_um": float(subset["depth_center_um"]),
            }
        )
    return pd.DataFrame(rows)


def draw_color_legend(legend_df: pd.DataFrame, output_path: Path):
    n = len(legend_df)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, ax = plt.subplots(figsize=(12, 0.58 * nrows + 1.1))
    ax.axis("off")
    fig.suptitle("Tracked-unit color legend for UnitMatch feature panels", fontsize=16, y=0.98)
    x_positions = [0.05, 0.37, 0.69]
    y0 = 0.86
    dy = 0.12
    for i, row in legend_df.reset_index(drop=True).iterrows():
        col = i // nrows
        row_idx = i % nrows
        x = x_positions[col]
        y = y0 - row_idx * dy
        ax.add_patch(plt.Rectangle((x, y - 0.028), 0.03, 0.05, color=row["color_hex"], transform=ax.transAxes, clip_on=False))
        ax.text(
            x + 0.04,
            y,
            f"{row['tracked_label']} | sessions={row['n_sessions_present']} | p={row['mean_cross_session_probability']:.3f} | depth={row['depth_center_um']:.1f} um",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=9.5,
        )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    FEATURE_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp" / "mplconfig").mkdir(parents=True, exist_ok=True)

    with np.load(UNITMATCH_ROOT / "UM Scores.npz", allow_pickle=True) as um_scores_npz:
        um_scores = {k: um_scores_npz[k] for k in um_scores_npz.files}
    with np.load(UNITMATCH_ROOT / "WaveformInfo.npz", allow_pickle=True) as wave_info_npz:
        wave_info = {k: wave_info_npz[k] for k in wave_info_npz.files}
    with open(UNITMATCH_ROOT / "ClusInfo.pickle", "rb") as f:
        clus_info = pickle.load(f)

    match_table = pd.read_csv(UNITMATCH_ROOT / "MatchTable.csv", nrows=5)
    cluster_map = pd.read_csv(TRACKED_ROOT / "cluster_to_tracked_unit.csv")
    coverage = pd.read_csv(TRACKED_ROOT / "tracked_unit_coverage_summary.csv")

    feature_inventory = make_feature_inventory(um_scores, wave_info, match_table.columns.tolist())
    feature_inventory.to_csv(FEATURE_VIZ_ROOT / "feature_inventory.csv", index=False)

    cluster_map = cluster_map[cluster_map["conflict_flag"] == False].copy()
    cluster_map["cluster_id"] = cluster_map["cluster_id"].astype(int)
    tracked_lookup = {
        (row.session_name, int(row.cluster_id)): int(row.tracked_unit_id)
        for row in cluster_map.itertuples(index=False)
    }

    selected_long_lived = coverage[
        (coverage["selection_eligible"] == True)
        & (coverage["n_sessions_present"] >= 10)
        & (coverage["min_good_tracked_spikes"] >= 1000)
        & (coverage["mean_cross_session_probability"] >= 0.6)
    ].copy()
    selected_long_lived = selected_long_lived.sort_values(
        ["depth_center_um", "tracked_unit_id"]
    ).reset_index(drop=True)
    selected_long_lived.to_csv(FEATURE_VIZ_ROOT / "selected_long_lived_units.csv", index=False)
    selected_ids = set(selected_long_lived["tracked_unit_id"].astype(int))
    color_legend_df = build_color_legend_table(selected_long_lived)
    color_legend_df.to_csv(FEATURE_VIZ_ROOT / "tracked_unit_color_legend.csv", index=False)
    draw_color_legend(color_legend_df, FEATURE_VIZ_ROOT / "tracked_unit_color_legend.png")

    session_names = list(clus_info["session_names"])
    session_ids = np.asarray(clus_info["session_id"]).astype(int)
    original_ids = np.asarray(clus_info["original_ids"]).reshape(-1)

    rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []

    for unit_idx in range(len(session_ids)):
        session_name = session_names[session_ids[unit_idx]]
        cluster_id = int(original_ids[unit_idx])
        tracked_unit_id = tracked_lookup.get((session_name, cluster_id))
        if tracked_unit_id is None or tracked_unit_id not in selected_ids:
            continue

        session_date = session_name.split("_", 1)[1]
        session_date_dt = datetime.strptime(session_date, "%Y-%m-%d")
        features = derive_unit_features(unit_idx, wave_info)

        wide_row = {
            "unit_index": unit_idx,
            "tracked_unit_id": tracked_unit_id,
            "session_name": session_name,
            "session_date": session_date,
            "cluster_id": cluster_id,
        }
        wide_row.update(features)
        wide_rows.append(wide_row)

        for feature_name, feature_value in features.items():
            rows.append(
                {
                    "unit_index": unit_idx,
                    "tracked_unit_id": tracked_unit_id,
                    "session_name": session_name,
                    "session_date": session_date,
                    "session_date_dt": session_date_dt,
                    "cluster_id": cluster_id,
                    "feature_name": feature_name,
                    "feature_family": next(spec["family"] for spec in FEATURE_SPECS if spec["name"] == feature_name),
                    "feature_value": feature_value,
                }
            )

    feature_table = pd.DataFrame(rows)
    feature_wide = pd.DataFrame(wide_rows)
    feature_wide["session_date_dt"] = pd.to_datetime(feature_wide["session_date"])

    first_vals = (
        feature_table.sort_values("session_date_dt")
        .groupby(["tracked_unit_id", "feature_name"], as_index=False)
        .first()[["tracked_unit_id", "feature_name", "session_date_dt", "feature_value"]]
        .rename(columns={"session_date_dt": "first_date_dt", "feature_value": "first_feature_value"})
    )
    feature_table = feature_table.merge(first_vals, on=["tracked_unit_id", "feature_name"], how="left")
    feature_table["days_since_first_appearance"] = (
        pd.to_datetime(feature_table["session_date_dt"]) - pd.to_datetime(feature_table["first_date_dt"])
    ).dt.days
    feature_table["feature_drift_from_first"] = feature_table["feature_value"] - feature_table["first_feature_value"]

    feature_table_out = feature_table.drop(columns=["session_date_dt", "first_date_dt"])
    feature_table_out.to_csv(FEATURE_VIZ_ROOT / "tracked_unit_feature_table.csv.gz", index=False, compression="gzip")
    feature_wide.to_csv(FEATURE_VIZ_ROOT / "tracked_unit_feature_table_wide.csv", index=False)

    raw_plot_path = FEATURE_VIZ_ROOT / "unitmatch_feature_timecourses.png"
    drift_plot_path = FEATURE_VIZ_ROOT / "unitmatch_feature_drift.png"
    draw_feature_panels(feature_table, FEATURE_SPECS, raw_plot_path, drift=False)
    draw_feature_panels(feature_table, FEATURE_SPECS, drift_plot_path, drift=True)
    raw_plot_labeled_path = FEATURE_VIZ_ROOT / "unitmatch_feature_timecourses_labeled.png"
    drift_plot_labeled_path = FEATURE_VIZ_ROOT / "unitmatch_feature_drift_labeled.png"
    draw_feature_panels(feature_table, FEATURE_SPECS, raw_plot_labeled_path, drift=False, labeled=True)
    draw_feature_panels(feature_table, FEATURE_SPECS, drift_plot_labeled_path, drift=True, labeled=True)

    feature_guide = pd.DataFrame(
        [
            {
                "feature_name": spec["name"],
                "label": spec["label"],
                "family": spec["family"],
                "definition": spec["definition"],
                "interpretation": spec["interpretation"],
            }
            for spec in FEATURE_SPECS
        ]
    )
    feature_guide.to_csv(FEATURE_VIZ_ROOT / "feature_panel_guide.csv", index=False)

    summary = {
        "created_at": datetime.now().isoformat(),
        "unitmatch_root": str(UNITMATCH_ROOT),
        "feature_viz_root": str(FEATURE_VIZ_ROOT),
        "n_native_um_score_keys": len(um_scores),
        "n_native_waveform_keys": len(wave_info),
        "n_inventory_rows": int(len(feature_inventory)),
        "n_feature_panels": len(FEATURE_SPECS),
        "selected_long_lived_unit_count": int(len(selected_long_lived)),
        "selected_long_lived_tracked_unit_ids": selected_long_lived["tracked_unit_id"].astype(int).tolist(),
        "feature_names": [spec["name"] for spec in FEATURE_SPECS],
        "thresholds": {
            "min_sessions_present": 10,
            "min_good_tracked_spikes": 1000,
            "min_mean_cross_session_probability": 0.6,
        },
        "outputs": {
            "feature_inventory_csv": str(FEATURE_VIZ_ROOT / "feature_inventory.csv"),
            "selected_long_lived_units_csv": str(FEATURE_VIZ_ROOT / "selected_long_lived_units.csv"),
            "tracked_unit_color_legend_csv": str(FEATURE_VIZ_ROOT / "tracked_unit_color_legend.csv"),
            "tracked_unit_color_legend_png": str(FEATURE_VIZ_ROOT / "tracked_unit_color_legend.png"),
            "feature_panel_guide_csv": str(FEATURE_VIZ_ROOT / "feature_panel_guide.csv"),
            "tracked_unit_feature_table_csv_gz": str(FEATURE_VIZ_ROOT / "tracked_unit_feature_table.csv.gz"),
            "tracked_unit_feature_table_wide_csv": str(FEATURE_VIZ_ROOT / "tracked_unit_feature_table_wide.csv"),
            "unitmatch_feature_timecourses_png": str(raw_plot_path),
            "unitmatch_feature_timecourses_labeled_png": str(raw_plot_labeled_path),
            "unitmatch_feature_drift_png": str(drift_plot_path),
            "unitmatch_feature_drift_labeled_png": str(drift_plot_labeled_path),
        },
    }
    with (FEATURE_VIZ_ROOT / "feature_timecourse_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
