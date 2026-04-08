#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/scratch/am15577/UnitMatch/match_raw_unitmatch/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from _pipeline_utils import dump_json, now_iso, parse_session_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UnitMatch paper-style QC figures for the 12-session AL032 run.")
    parser.add_argument(
        "--config",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/configs/unitmatch_run_config.json",
    )
    parser.add_argument(
        "--cluster-to-tracked-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/tracked_tables/cluster_to_tracked_unit.csv",
    )
    parser.add_argument(
        "--tracked-summary-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/tracked_tables/tracked_unit_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/unitmatch_raw_12session/outputs_unitmatch",
    )
    parser.add_argument("--match-threshold", type=float, default=0.5)
    return parser.parse_args()


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


def parse_json_list(text: str) -> list[str]:
    if not isinstance(text, str) or not text:
        return []
    payload = json.loads(text)
    return [str(item) for item in payload]


def load_inputs(
    config_path: Path,
    cluster_to_tracked_csv: Path,
    tracked_summary_csv: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict, np.ndarray]:
    config = json.loads(config_path.read_text())
    unitmatch_root = Path(config["output_root"])
    match_prob = np.load(unitmatch_root / "MatchProb.npy", mmap_mode="r")
    with open(unitmatch_root / "ClusInfo.pickle", "rb") as handle:
        clus_info = pickle.load(handle)

    cluster_to_tracked = pd.read_csv(cluster_to_tracked_csv)
    tracked_summary = pd.read_csv(tracked_summary_csv)
    return config, cluster_to_tracked, tracked_summary, clus_info, match_prob


def build_session_metadata(clus_info: dict) -> tuple[list[str], list[str], np.ndarray]:
    session_names = [str(name) for name in clus_info["session_names"].tolist()]
    session_dates = [parse_session_date(name) for name in session_names]
    session_switch = np.asarray(clus_info["session_switch"], dtype=np.int64)
    return session_names, session_dates, session_switch


def build_pairwise_tracking_tables(
    session_names: list[str],
    tracked_summary: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    valid_summary = tracked_summary.loc[series_to_bool_mask(tracked_summary["conflict_free_validity_flag"])].copy()
    n_sessions = len(session_names)
    index_by_name = {name: idx for idx, name in enumerate(session_names)}

    shared_counts = np.zeros((n_sessions, n_sessions), dtype=np.int64)
    for row in valid_summary.itertuples(index=False):
        present_names = parse_json_list(row.sessions_present)
        present_indices = sorted({index_by_name[name] for name in present_names if name in index_by_name})
        if not present_indices:
            continue
        for i in present_indices:
            shared_counts[i, i] += 1
        for pos_i, i in enumerate(present_indices):
            for j in present_indices[pos_i + 1 :]:
                shared_counts[i, j] += 1
                shared_counts[j, i] += 1

    units_per_session = np.diag(shared_counts).astype(np.int64)
    possible = np.minimum.outer(units_per_session, units_per_session).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        shared_fraction = np.divide(shared_counts, possible, out=np.zeros_like(possible), where=possible > 0)
    np.fill_diagonal(shared_fraction, 1.0)

    dates = pd.to_datetime([parse_session_date(name) for name in session_names])
    rows: list[dict[str, object]] = []
    for i, session_i in enumerate(session_names):
        for j, session_j in enumerate(session_names):
            rows.append(
                {
                    "session_name_i": session_i,
                    "session_name_j": session_j,
                    "session_date_i": parse_session_date(session_i),
                    "session_date_j": parse_session_date(session_j),
                    "session_index_i": i,
                    "session_index_j": j,
                    "days_apart": abs(int((dates[j] - dates[i]).days)),
                    "units_in_session_i": int(units_per_session[i]),
                    "units_in_session_j": int(units_per_session[j]),
                    "shared_tracked_units": int(shared_counts[i, j]),
                    "max_possible_shared_units": int(min(units_per_session[i], units_per_session[j])),
                    "shared_fraction_of_possible": float(shared_fraction[i, j]),
                }
            )
    pair_metrics = pd.DataFrame(rows)
    return shared_counts, shared_fraction, units_per_session, pair_metrics


def compute_probability_distributions(match_prob: np.ndarray, session_switch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    self_hist = np.zeros(edges.size - 1, dtype=np.int64)
    within_hist = np.zeros(edges.size - 1, dtype=np.int64)
    across_hist = np.zeros(edges.size - 1, dtype=np.int64)

    self_hist += np.histogram(np.diag(match_prob), bins=edges)[0]
    n_sessions = session_switch.size - 1
    for i in range(n_sessions):
        start_i = int(session_switch[i])
        stop_i = int(session_switch[i + 1])
        block = np.asarray(match_prob[start_i:stop_i, start_i:stop_i], dtype=np.float32)
        if block.size:
            triu = block[np.triu_indices(block.shape[0], k=1)]
            if triu.size:
                within_hist += np.histogram(triu[np.isfinite(triu)], bins=edges)[0]
        for j in range(i + 1, n_sessions):
            start_j = int(session_switch[j])
            stop_j = int(session_switch[j + 1])
            cross_block = np.asarray(match_prob[start_i:stop_i, start_j:stop_j], dtype=np.float32)
            if cross_block.size:
                across_hist += np.histogram(cross_block[np.isfinite(cross_block)], bins=edges)[0]

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, self_hist, within_hist, across_hist


def add_session_boundaries(ax: plt.Axes, session_switch: np.ndarray) -> None:
    for boundary in session_switch[1:-1]:
        pos = float(boundary) - 0.5
        ax.axvline(pos, color="#cc2f2f", linewidth=0.8, alpha=0.9)
        ax.axhline(pos, color="#cc2f2f", linewidth=0.8, alpha=0.9)


def get_session_tick_positions(session_switch: np.ndarray) -> np.ndarray:
    starts = session_switch[:-1]
    stops = session_switch[1:]
    return 0.5 * (starts + stops - 1)


def plot_match_probability_matrix(
    match_prob: np.ndarray,
    session_switch: np.ndarray,
    session_dates: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 9.5), dpi=220)
    im = ax.imshow(match_prob, cmap="gray_r", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
    add_session_boundaries(ax, session_switch)
    ticks = get_session_tick_positions(session_switch)
    ax.set_xticks(ticks)
    ax.set_xticklabels(session_dates, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(session_dates, fontsize=8)
    ax.set_xlabel("session-ordered units")
    ax.set_ylabel("session-ordered units")
    ax.set_title("UnitMatch probability matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("match probability")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_thresholded_match_matrix(
    match_prob: np.ndarray,
    session_switch: np.ndarray,
    session_dates: list[str],
    match_threshold: float,
    output_path: Path,
) -> None:
    binary = (np.asarray(match_prob) >= float(match_threshold)).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(10.5, 9.5), dpi=220)
    ax.imshow(binary, cmap=ListedColormap(["white", "black"]), vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    add_session_boundaries(ax, session_switch)
    ticks = get_session_tick_positions(session_switch)
    ax.set_xticks(ticks)
    ax.set_xticklabels(session_dates, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(session_dates, fontsize=8)
    ax.set_xlabel("session-ordered units")
    ax.set_ylabel("session-ordered units")
    ax.set_title(f"UnitMatch thresholded matrix (p >= {match_threshold:.2f})")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray, fmt: str, text_threshold: float | None = None) -> None:
    if text_threshold is None:
        text_threshold = float(np.nanmax(matrix)) * 0.55 if np.isfinite(matrix).any() else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if value >= text_threshold else "black"
            display_value = int(round(float(value))) if fmt == "d" else float(value)
            ax.text(j, i, format(display_value, fmt), ha="center", va="center", fontsize=7, color=color)


def plot_session_pair_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    cmap: str,
    colorbar_label: str,
    output_path: Path,
    fmt: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.5), dpi=220)
    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest", aspect="equal")
    annotate_heatmap(ax, matrix, fmt=fmt)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("session")
    ax.set_ylabel("session")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_probability_distributions(
    centers: np.ndarray,
    self_hist: np.ndarray,
    within_hist: np.ndarray,
    across_hist: np.ndarray,
    match_threshold: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=220)
    total_self = max(int(self_hist.sum()), 1)
    total_within = max(int(within_hist.sum()), 1)
    total_across = max(int(across_hist.sum()), 1)

    axes[0].plot(centers, self_hist / total_self, color="#218c3c", linewidth=2.0, label="self")
    axes[0].plot(centers, within_hist / total_within, color="#2459a6", linewidth=2.0, label="within session")
    axes[0].plot(centers, across_hist / total_across, color="#c03a2b", linewidth=2.0, label="across sessions")
    axes[0].axvline(match_threshold, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("match probability")
    axes[0].set_ylabel("fraction of pairs")
    axes[0].set_title("Probability distributions")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(centers, np.cumsum(self_hist) / total_self, color="#218c3c", linewidth=2.0, label="self")
    axes[1].plot(centers, np.cumsum(within_hist) / total_within, color="#2459a6", linewidth=2.0, label="within session")
    axes[1].plot(centers, np.cumsum(across_hist) / total_across, color="#c03a2b", linewidth=2.0, label="across sessions")
    axes[1].axvline(match_threshold, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("match probability")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].set_title("Probability cumulative distributions")

    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_lifespan_histogram(
    tracked_summary: pd.DataFrame,
    output_path: Path,
) -> pd.Series:
    valid_summary = tracked_summary.loc[series_to_bool_mask(tracked_summary["conflict_free_validity_flag"])].copy()
    counts = valid_summary["n_sessions_present"].astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220)
    x = np.arange(1, 13, dtype=int)
    y = np.array([int(counts.get(i, 0)) for i in x], dtype=np.int64)
    ax.bar(x, y, color="#4c72b0", width=0.82)
    ax.set_xticks(x)
    ax.set_xlabel("sessions present")
    ax.set_ylabel("tracked units")
    ax.set_title("Tracked-unit lifespan across sessions")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return counts


def plot_tracking_vs_gap(pair_metrics: pd.DataFrame, output_path: Path) -> None:
    pair_df = pair_metrics.loc[pair_metrics["session_index_j"] > pair_metrics["session_index_i"]].copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=220)
    sizes = 20.0 + 0.18 * pair_df["shared_tracked_units"].to_numpy(dtype=np.float64)
    ax.scatter(
        pair_df["days_apart"],
        pair_df["shared_fraction_of_possible"],
        s=sizes,
        color="#202020",
        alpha=0.65,
        linewidth=0.4,
        edgecolor="white",
    )
    gap_summary = (
        pair_df.groupby("days_apart", as_index=False)
        .agg(
            mean_fraction=("shared_fraction_of_possible", "mean"),
            median_fraction=("shared_fraction_of_possible", "median"),
        )
        .sort_values("days_apart")
    )
    ax.plot(gap_summary["days_apart"], gap_summary["median_fraction"], color="#c03a2b", linewidth=2.0, label="median")
    ax.plot(gap_summary["days_apart"], gap_summary["mean_fraction"], color="#2459a6", linewidth=1.8, linestyle="--", label="mean")
    ax.set_xlabel("days apart")
    ax.set_ylabel("shared fraction of smaller session")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Tracking stability across session gaps")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_summary_panel(
    shared_counts: np.ndarray,
    shared_fraction: np.ndarray,
    session_dates: list[str],
    lifespan_counts: pd.Series,
    pair_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), dpi=220)

    im0 = axes[0, 0].imshow(shared_counts, cmap="viridis", interpolation="nearest", aspect="equal")
    axes[0, 0].set_title("Shared tracked units")
    annotate_heatmap(axes[0, 0], shared_counts, fmt="d")

    im1 = axes[0, 1].imshow(shared_fraction, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    axes[0, 1].set_title("Shared fraction of smaller session")
    annotate_heatmap(axes[0, 1], shared_fraction, fmt=".2f", text_threshold=0.55)

    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xticks(np.arange(len(session_dates)))
        ax.set_xticklabels(session_dates, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(session_dates)))
        ax.set_yticklabels(session_dates, fontsize=7)
        ax.set_xlabel("session")
        ax.set_ylabel("session")

    x = np.arange(1, 13, dtype=int)
    y = np.array([int(lifespan_counts.get(i, 0)) for i in x], dtype=np.int64)
    axes[1, 0].bar(x, y, color="#4c72b0", width=0.82)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xlabel("sessions present")
    axes[1, 0].set_ylabel("tracked units")
    axes[1, 0].set_title("Tracked-unit lifespan")
    axes[1, 0].grid(axis="y", alpha=0.2, linewidth=0.5)

    pair_df = pair_metrics.loc[pair_metrics["session_index_j"] > pair_metrics["session_index_i"]].copy()
    sizes = 20.0 + 0.18 * pair_df["shared_tracked_units"].to_numpy(dtype=np.float64)
    axes[1, 1].scatter(
        pair_df["days_apart"],
        pair_df["shared_fraction_of_possible"],
        s=sizes,
        color="#202020",
        alpha=0.65,
        linewidth=0.4,
        edgecolor="white",
    )
    gap_summary = (
        pair_df.groupby("days_apart", as_index=False)
        .agg(median_fraction=("shared_fraction_of_possible", "median"))
        .sort_values("days_apart")
    )
    axes[1, 1].plot(gap_summary["days_apart"], gap_summary["median_fraction"], color="#c03a2b", linewidth=2.0)
    axes[1, 1].set_xlabel("days apart")
    axes[1, 1].set_ylabel("shared fraction")
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].set_title("Tracking vs session gap")
    axes[1, 1].grid(alpha=0.2, linewidth=0.5)

    cbar0 = fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    cbar0.set_label("tracked units")
    cbar1 = fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar1.set_label("fraction")

    fig.suptitle("AL032 UnitMatch tracking summary", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, cluster_to_tracked, tracked_summary, clus_info, match_prob = load_inputs(
        Path(args.config),
        Path(args.cluster_to_tracked_csv),
        Path(args.tracked_summary_csv),
    )

    session_names, session_dates, session_switch = build_session_metadata(clus_info)
    shared_counts, shared_fraction, units_per_session, pair_metrics = build_pairwise_tracking_tables(
        session_names,
        tracked_summary,
    )
    centers, self_hist, within_hist, across_hist = compute_probability_distributions(match_prob, session_switch)

    plot_match_probability_matrix(
        match_prob,
        session_switch,
        session_dates,
        output_dir / "unitmatch_match_probability_matrix.png",
    )
    plot_thresholded_match_matrix(
        match_prob,
        session_switch,
        session_dates,
        args.match_threshold,
        output_dir / "unitmatch_thresholded_match_matrix.png",
    )
    plot_session_pair_heatmap(
        shared_counts.astype(np.float64),
        session_dates,
        "Session-by-session tracked-unit counts",
        "viridis",
        "tracked units",
        output_dir / "unitmatch_session_pair_tracked_counts.png",
        fmt="d",
    )
    plot_session_pair_heatmap(
        shared_fraction,
        session_dates,
        "Session-by-session tracked fraction",
        "magma",
        "fraction of smaller session",
        output_dir / "unitmatch_session_pair_tracked_fraction.png",
        fmt=".2f",
    )
    plot_probability_distributions(
        centers,
        self_hist,
        within_hist,
        across_hist,
        args.match_threshold,
        output_dir / "unitmatch_probability_distributions.png",
    )
    lifespan_counts = plot_lifespan_histogram(
        tracked_summary,
        output_dir / "unitmatch_tracked_unit_lifespan_histogram.png",
    )
    plot_tracking_vs_gap(
        pair_metrics,
        output_dir / "unitmatch_tracking_vs_session_gap.png",
    )
    plot_summary_panel(
        shared_counts,
        shared_fraction,
        session_dates,
        lifespan_counts,
        pair_metrics,
        output_dir / "unitmatch_tracking_summary_panel.png",
    )

    pair_metrics.to_csv(output_dir / "session_pair_tracking_metrics.csv", index=False)
    lifespan_df = pd.DataFrame(
        {
            "n_sessions_present": np.arange(1, 13, dtype=int),
            "tracked_unit_count": [int(lifespan_counts.get(i, 0)) for i in range(1, 13)],
        }
    )
    lifespan_df.to_csv(output_dir / "tracked_unit_lifespan_counts.csv", index=False)

    max_pair_row = (
        pair_metrics.loc[pair_metrics["session_index_j"] > pair_metrics["session_index_i"]]
        .sort_values(["shared_tracked_units", "shared_fraction_of_possible"], ascending=[False, False])
        .iloc[0]
    )
    summary = {
        "created_at": now_iso(),
        "output_dir": str(output_dir),
        "session_names": session_names,
        "session_dates": session_dates,
        "n_sessions": len(session_names),
        "n_units_in_match_probability_matrix": int(match_prob.shape[0]),
        "match_threshold": float(args.match_threshold),
        "n_conflict_free_tracked_units": int(
            tracked_summary.loc[series_to_bool_mask(tracked_summary["conflict_free_validity_flag"])].shape[0]
        ),
        "units_per_session": {session_names[idx]: int(units_per_session[idx]) for idx in range(len(session_names))},
        "max_pair_shared_units": {
            "session_name_i": str(max_pair_row["session_name_i"]),
            "session_name_j": str(max_pair_row["session_name_j"]),
            "shared_tracked_units": int(max_pair_row["shared_tracked_units"]),
            "shared_fraction_of_possible": float(max_pair_row["shared_fraction_of_possible"]),
        },
        "probability_distribution_counts": {
            "self_pairs": int(self_hist.sum()),
            "within_session_pairs": int(within_hist.sum()),
            "across_session_pairs": int(across_hist.sum()),
        },
        "files": {
            "match_probability_matrix": str(output_dir / "unitmatch_match_probability_matrix.png"),
            "thresholded_match_matrix": str(output_dir / "unitmatch_thresholded_match_matrix.png"),
            "session_pair_tracked_counts": str(output_dir / "unitmatch_session_pair_tracked_counts.png"),
            "session_pair_tracked_fraction": str(output_dir / "unitmatch_session_pair_tracked_fraction.png"),
            "probability_distributions": str(output_dir / "unitmatch_probability_distributions.png"),
            "tracked_unit_lifespan_histogram": str(output_dir / "unitmatch_tracked_unit_lifespan_histogram.png"),
            "tracking_vs_session_gap": str(output_dir / "unitmatch_tracking_vs_session_gap.png"),
            "tracking_summary_panel": str(output_dir / "unitmatch_tracking_summary_panel.png"),
            "session_pair_tracking_metrics": str(output_dir / "session_pair_tracking_metrics.csv"),
            "tracked_unit_lifespan_counts": str(output_dir / "tracked_unit_lifespan_counts.csv"),
        },
    }
    dump_json(output_dir / "unitmatch_plot_summary.json", summary)


if __name__ == "__main__":
    main()
