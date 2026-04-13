#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/tmp/mplconfig"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge")
FEATURE_VIZ_ROOT = ROOT / "outputs" / "unitmatch_dredge_12session" / "feature_viz"

FEATURE_META = [
    ("centroid_x_um", "Centroid X (um)", "spatial", "Mean x-coordinate of the UnitMatch centroid."),
    ("centroid_y_um", "Centroid Y (um)", "spatial", "Mean y-coordinate of the UnitMatch centroid."),
    ("centroid_z_um", "Centroid Z (um)", "spatial", "Mean z-coordinate of the UnitMatch centroid."),
    ("centroid_cv_disagreement_um", "Centroid CV disagreement (um)", "spatial", "Distance between centroid estimates from the two CV halves."),
    ("max_site_mean_index", "Mean max-site index", "spatial", "Average peak-channel index across CV halves."),
    ("max_site_cv_gap", "Max-site CV gap", "spatial", "Absolute difference in peak-channel index between CV halves."),
    ("waveform_trough_raw", "Waveform trough (raw)", "waveform", "Minimum value of the average waveform."),
    ("waveform_peak_raw", "Waveform peak (raw)", "waveform", "Maximum value of the average waveform."),
    ("waveform_ptp_raw", "Waveform peak-to-trough (raw)", "waveform", "Peak minus trough of the average waveform."),
    ("waveform_energy_raw", "Waveform energy (raw)", "waveform", "Sum of squared average-waveform values."),
    ("waveform_cv_corr", "Waveform CV correlation", "waveform", "Correlation between the two CV average waveforms."),
    ("trough_to_peak_samples", "Trough-to-peak (samples)", "waveform", "Number of samples from trough to next positive peak."),
    ("active_timepoint_count", "Active trajectory timepoints", "trajectory", "Count of waveform timepoints with valid trajectory coordinates."),
    ("trajectory_total_path_um", "Trajectory total path (um)", "trajectory", "Total path length of the average waveform trajectory."),
    ("trajectory_displacement_um", "Trajectory displacement (um)", "trajectory", "Straight-line distance from first to last valid trajectory point."),
    ("trajectory_y_span_um", "Trajectory Y span (um)", "trajectory", "Maximum minus minimum y along the average waveform trajectory."),
    ("trajectory_cv_distance_um", "Trajectory CV distance (um)", "trajectory", "Mean distance between the two CV trajectories."),
]

MANUAL_DROP_REASONS = {
    "centroid_x_um": "degenerate_constant_feature",
    "max_site_mean_index": "discrete_index_proxy_not_useful_for_timecourse",
    "max_site_cv_gap": "discrete_index_proxy_not_useful_for_timecourse",
}

FAMILY_COLORS = {
    "spatial": "#4c78a8",
    "waveform": "#f58518",
    "trajectory": "#54a24b",
}


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    q05, q95 = np.quantile(finite, [0.05, 0.95])
    if np.isclose(q05, q95):
        center = float(np.nanmedian(finite))
        span = max(abs(center) * 0.1, 1.0)
        return center - span, center + span
    pad = 0.1 * (q95 - q05)
    return q05 - pad, q95 + pad


def choose_example_units(selected: pd.DataFrame, n_examples: int = 4) -> pd.DataFrame:
    ordered = selected.sort_values(["depth_center_um", "tracked_unit_id"]).reset_index(drop=True)
    if len(ordered) <= n_examples:
        return ordered.copy()
    idx = np.linspace(0, len(ordered) - 1, n_examples)
    idx = np.unique(np.round(idx).astype(int))
    chosen = ordered.iloc[idx].copy().reset_index(drop=True)
    return chosen


def draw_summary_family(df: pd.DataFrame, features: list[dict], output_path: Path):
    n = len(features)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.8 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, feat in zip(axes, features):
        sub = df[df["feature_name"] == feat["feature_name"]].copy()
        stat = (
            sub.groupby("days_since_first_appearance")["normalized_drift"]
            .agg(
                median="median",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
            )
            .reset_index()
            .sort_values("days_since_first_appearance")
        )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.fill_between(stat["days_since_first_appearance"], stat["q1"], stat["q3"], color="#777777", alpha=0.22)
        ax.plot(stat["days_since_first_appearance"], stat["median"], color="black", linewidth=2.4)
        ymin, ymax = robust_limits(sub["normalized_drift"].to_numpy())
        ax.set_ylim(ymin, ymax)
        ax.set_title(feat["label"], fontsize=11)
        ax.set_xlabel("Days since first appearance", fontsize=9)
        ax.set_ylabel("Normalized drift\n(Δ / feature IQR)", fontsize=9)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.text(
            0.02,
            0.96,
            feat["short_note"],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            color="dimgray",
        )

    for ax in axes[n:]:
        ax.axis("off")

    family = features[0]["family"] if features else "features"
    fig.suptitle(
        f"UnitMatch feature stability summary: {family} features",
        fontsize=16,
        y=1.02,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_example_family(df: pd.DataFrame, features: list[dict], example_units: pd.DataFrame, output_path: Path):
    n = len(features)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.8 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    unit_ids = example_units["tracked_unit_id"].astype(int).tolist()
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(unit_ids)))
    color_map = {uid: colors[i] for i, uid in enumerate(unit_ids)}

    legend_handles = []
    legend_labels = []
    for ax, feat in zip(axes, features):
        sub = df[(df["feature_name"] == feat["feature_name"]) & (df["tracked_unit_id"].isin(unit_ids))].copy()
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        for uid, grp in sub.groupby("tracked_unit_id"):
            grp = grp.sort_values("days_since_first_appearance")
            line, = ax.plot(
                grp["days_since_first_appearance"],
                grp["normalized_drift"],
                color=color_map[uid],
                linewidth=1.8,
                marker="o",
                markersize=3,
                alpha=0.9,
            )
            if feat["feature_name"] == features[0]["feature_name"]:
                legend_handles.append(line)
                legend_labels.append(f"T{uid}")
        ymin, ymax = robust_limits(sub["normalized_drift"].to_numpy())
        ax.set_ylim(ymin, ymax)
        ax.set_title(feat["label"], fontsize=11)
        ax.set_xlabel("Days since first appearance", fontsize=9)
        ax.set_ylabel("Normalized drift\n(Δ / feature IQR)", fontsize=9)
        ax.grid(alpha=0.18, linewidth=0.5)

    for ax in axes[n:]:
        ax.axis("off")

    family = features[0]["family"] if features else "features"
    fig.suptitle(
        f"Example tracked units: {family} features",
        fontsize=16,
        y=1.02,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(6, len(legend_labels)),
        frameon=False,
        title="Example long-lived tracked units",
        title_fontsize=10,
        fontsize=9,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_volatility_ranking(df: pd.DataFrame, output_path: Path):
    ranked = df.sort_values("median_abs_normalized_drift", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    colors = [FAMILY_COLORS.get(fam, "#888888") for fam in ranked["family"]]
    ax.bar(ranked["label"], ranked["median_abs_normalized_drift"], color=colors, alpha=0.9)
    ax.set_ylabel("Median |normalized drift|", fontsize=10)
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_title("UnitMatch feature volatility ranking", fontsize=16)
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=45, labelsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[fam]) for fam in ["spatial", "waveform", "trajectory"]
    ]
    ax.legend(handles, ["Spatial", "Waveform", "Trajectory"], frameon=False, fontsize=9)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    feature_table = pd.read_csv(FEATURE_VIZ_ROOT / "tracked_unit_feature_table.csv.gz")
    selected = pd.read_csv(FEATURE_VIZ_ROOT / "selected_long_lived_units.csv")

    meta = pd.DataFrame(FEATURE_META, columns=["feature_name", "label", "family", "definition"])

    stats = []
    for feat, grp in feature_table.groupby("feature_name"):
        vals = grp["feature_value"]
        dr = grp["feature_drift_from_first"]
        abs_dr = dr.abs()
        value_iqr = float(vals.quantile(0.75) - vals.quantile(0.25))
        abs_drift_q95 = float(abs_dr.quantile(0.95))
        zero_drift_fraction = float((abs_dr < 1e-9).mean())
        keep = True
        reason = ""
        if feat in MANUAL_DROP_REASONS:
            keep = False
            reason = MANUAL_DROP_REASONS[feat]
        elif value_iqr < 1e-8 or abs_drift_q95 < 1e-8:
            keep = False
            reason = "degenerate_near_constant_feature"
        stats.append(
            {
                "feature_name": feat,
                "value_std": float(vals.std()),
                "value_iqr": value_iqr,
                "drift_std": float(dr.std()),
                "drift_iqr": float(dr.quantile(0.75) - dr.quantile(0.25)),
                "median_abs_drift": float(abs_dr.median()),
                "abs_drift_q95": abs_drift_q95,
                "zero_drift_fraction": zero_drift_fraction,
                "keep_for_plots": keep,
                "drop_reason": reason,
            }
        )
    filter_summary = meta.merge(pd.DataFrame(stats), on="feature_name", how="left")
    filter_summary["short_note"] = filter_summary["definition"]

    scale_map = {}
    for row in filter_summary.itertuples(index=False):
        scale = float(row.value_iqr) if float(row.value_iqr) > 1e-8 else np.nan
        scale_map[row.feature_name] = scale
    feature_table["feature_scale_iqr"] = feature_table["feature_name"].map(scale_map)
    feature_table["normalized_drift"] = feature_table["feature_drift_from_first"] / feature_table["feature_scale_iqr"]

    kept_features = filter_summary[filter_summary["keep_for_plots"]].copy()
    volatility = (
        feature_table[feature_table["feature_name"].isin(kept_features["feature_name"])]
        .groupby("feature_name")["normalized_drift"]
        .apply(lambda s: float(np.nanmedian(np.abs(s))))
        .reset_index(name="median_abs_normalized_drift")
    )
    kept_features = kept_features.merge(volatility, on="feature_name", how="left")
    kept_features.to_csv(FEATURE_VIZ_ROOT / "feature_filter_summary.csv", index=False)

    example_units = choose_example_units(selected, n_examples=4).copy()
    colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(example_units)))
    example_units["color_hex"] = [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in colors
    ]
    example_units.to_csv(FEATURE_VIZ_ROOT / "example_units_selected.csv", index=False)

    feature_table = feature_table[feature_table["feature_name"].isin(kept_features["feature_name"])].copy()

    plot_files = []
    for family in ["spatial", "waveform", "trajectory"]:
        fam_features = kept_features[kept_features["family"] == family].copy()
        if fam_features.empty:
            continue
        summary_path = FEATURE_VIZ_ROOT / f"feature_stability_summary_{family}.png"
        example_path = FEATURE_VIZ_ROOT / f"feature_example_units_{family}.png"
        draw_summary_family(feature_table, fam_features.to_dict("records"), summary_path)
        draw_example_family(feature_table, fam_features.to_dict("records"), example_units, example_path)
        plot_files.extend([str(summary_path), str(example_path)])

    ranking_path = FEATURE_VIZ_ROOT / "feature_volatility_ranking.png"
    draw_volatility_ranking(kept_features[["feature_name", "label", "family", "median_abs_normalized_drift"]], ranking_path)
    plot_files.append(str(ranking_path))

    summary = {
        "created_at": pd.Timestamp.now().isoformat(),
        "n_total_features": int(len(filter_summary)),
        "n_kept_features": int(kept_features["feature_name"].nunique()),
        "kept_features": kept_features["feature_name"].tolist(),
        "dropped_features": filter_summary.loc[~filter_summary["keep_for_plots"], "feature_name"].tolist(),
        "selected_example_units": example_units["tracked_unit_id"].astype(int).tolist(),
        "normalization": "normalized_drift = (feature_value - first_value) / feature_global_IQR",
        "design": {
            "summary_plots": "median + IQR only, no per-unit spaghetti",
            "example_plots": "4 depth-diverse long-lived units only",
            "volatility_metric": "median absolute normalized drift",
            "y_limits": "robust 5th-95th percentile limits per panel",
        },
        "outputs": {
            "feature_filter_summary_csv": str(FEATURE_VIZ_ROOT / "feature_filter_summary.csv"),
            "example_units_selected_csv": str(FEATURE_VIZ_ROOT / "example_units_selected.csv"),
            "plot_files": plot_files,
        },
    }
    with (FEATURE_VIZ_ROOT / "feature_timecourse_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
