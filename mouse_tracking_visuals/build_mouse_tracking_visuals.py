#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass
class SessionSpec:
    mouse: str
    group: str
    label: str
    ks_dir: Path
    raw_waveform_dir: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build depth/time and waveform tracking visuals for one mouse."
    )
    parser.add_argument(
        "--mouse",
        default="AL032",
        help="Mouse ID to visualize. Default: AL032",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to mouse_tracking_visuals/<mouse> at the repo root.",
    )
    parser.add_argument(
        "--top-uids",
        type=int,
        default=12,
        help="Number of tracked UIDs to highlight. Default: 12",
    )
    parser.add_argument(
        "--max-all-spikes",
        type=int,
        default=250000,
        help="Maximum spikes per session in the amplitude-colored plot. Default: 250000",
    )
    parser.add_argument(
        "--max-background-spikes",
        type=int,
        default=120000,
        help="Maximum background spikes per session in the matched-unit plot. Default: 120000",
    )
    parser.add_argument(
        "--max-highlight-spikes",
        type=int,
        default=18000,
        help="Maximum spikes per UID per session in the matched-unit plot. Default: 18000",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def configure_runtime(root: Path) -> None:
    mpl_dir = root / ".mplconfig"
    cache_dir = root / ".cache"
    mpl_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def extract_available_unit_ids(raw_waveform_dir: Path) -> set[int]:
    unit_ids: set[int] = set()
    for path in raw_waveform_dir.glob("Unit*_RawSpikes.npy"):
        match = re.match(r"Unit(\d+)_RawSpikes\.npy$", path.name)
        if match:
            unit_ids.add(int(match.group(1)))
    return unit_ids


def build_filtered_label_tsv(source_tsv: Path, raw_waveform_dir: Path, dest_tsv: Path) -> Path:
    source_df = pd.read_csv(source_tsv, sep="\t")
    available_ids = extract_available_unit_ids(raw_waveform_dir)
    if "cluster_id" not in source_df.columns or "group" not in source_df.columns:
        raise ValueError(f"Unexpected label TSV schema in {source_tsv}")

    filtered = source_df[source_df["cluster_id"].isin(sorted(available_ids))].copy()
    if filtered.empty:
        filtered = pd.DataFrame(
            {"cluster_id": sorted(available_ids), "group": ["good"] * len(available_ids)}
        )

    dest_tsv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(dest_tsv, sep="\t", index=False)
    return dest_tsv


def enumerate_sessions(data_root: Path, generated_label_dir: Path) -> tuple[list[SessionSpec], list[object]]:
    sessions: list[SessionSpec] = []

    for mouse_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        mouse = mouse_dir.name

        for day_dir in sorted(p for p in mouse_dir.iterdir() if p.is_dir() and p.name != "Chronic"):
            raw_dirs = sorted(day_dir.glob("Probe*/**/RawWaveforms"))
            if not raw_dirs:
                continue
            raw_dir = raw_dirs[0].resolve()
            ks_dir = raw_dir.parent.resolve()
            label_path = (ks_dir / "cluster_group.tsv").resolve()
            sessions.append(
                SessionSpec(
                    mouse=mouse,
                    group="acute",
                    label=day_dir.name,
                    ks_dir=ks_dir,
                    raw_waveform_dir=raw_dir,
                    label_path=label_path,
                )
            )

        chronic_root = mouse_dir / "Chronic"
        if chronic_root.exists():
            for raw_dir in sorted(chronic_root.glob("Probe*/**/[0-9]/RawWaveforms")):
                raw_dir = raw_dir.resolve()
                ks_dir = raw_dir.parent.parent.resolve()
                session_token = raw_dir.parent.name
                generated_tsv = generated_label_dir / f"{mouse}_chronic_{session_token}_cluster_group.tsv"
                label_path = build_filtered_label_tsv(ks_dir / "cluster_group.tsv", raw_dir, generated_tsv)
                sessions.append(
                    SessionSpec(
                        mouse=mouse,
                        group="chronic",
                        label=f"Chronic_{session_token}",
                        ks_dir=ks_dir,
                        raw_waveform_dir=raw_dir,
                        label_path=label_path.resolve(),
                    )
                )

    return sessions, []


def load_unit_table(classic_dir: Path) -> pd.DataFrame:
    match_table = pd.read_csv(classic_dir / "MatchTable.csv")
    diag = match_table[
        (match_table["ID1"] == match_table["ID2"]) & (match_table["RecSes 1"] == match_table["RecSes 2"])
    ].copy()
    diag = diag[
        [
            "ID1",
            "RecSes 1",
            "UID Liberal 1",
            "UID int 1",
            "UID Conservative 1",
        ]
    ].drop_duplicates()
    diag = diag.rename(
        columns={
            "ID1": "unit_id",
            "RecSes 1": "recording_index",
            "UID Liberal 1": "uid_liberal",
            "UID int 1": "uid_default",
            "UID Conservative 1": "uid_conservative",
        }
    )
    diag["recording_index"] = diag["recording_index"].astype(int)
    return diag.reset_index(drop=True)


def short_session_labels(session_labels: list[str]) -> list[str]:
    out = []
    acute_idx = 1
    chronic_idx = 1
    for label in session_labels:
        if label.startswith("Chronic_"):
            out.append(f"C{chronic_idx}")
            chronic_idx += 1
        else:
            out.append(f"A{acute_idx}")
            acute_idx += 1
    return out


def build_global_unit_table(classic_dir: Path) -> pd.DataFrame:
    import pickle

    clus_info = pickle.load(open(classic_dir / "ClusInfo.pickle", "rb"))
    session_id = np.asarray(clus_info["session_id"]).astype(int)
    original_ids = np.asarray(clus_info["original_ids"]).astype(int)
    global_df = pd.DataFrame(
        {
            "global_index": np.arange(len(original_ids), dtype=int),
            "unit_id": original_ids,
            "recording_index": session_id + 1,
        }
    )
    unit_df = load_unit_table(classic_dir)
    return unit_df.merge(global_df, on=["unit_id", "recording_index"], how="left")


def deterministic_subsample(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, num=max_points, dtype=int)


def parse_chronic_token(label: str) -> int | None:
    match = re.match(r"Chronic_(\d+)$", label)
    if not match:
        return None
    return int(match.group(1))


def load_session_spikes(spec: SessionSpec, recording_index: int, allowed_unit_ids: set[int]) -> dict[str, object]:
    prepared_path = spec.ks_dir / "PreparedData.mat"
    with h5py.File(prepared_path, "r") as f:
        st = f["sp/st"][:].reshape(-1)
        depth = f["sp/spikeDepths"][:].reshape(-1)
        amp = f["sp/spikeAmps"][:].reshape(-1)
        clu = f["sp/clu"][:].reshape(-1).astype(int)

        chronic_token = parse_chronic_token(spec.label)
        if chronic_token is not None:
            recses = f["sp/RecSes"][:].reshape(-1).astype(int)
            session_mask = recses == chronic_token
            st = st[session_mask]
            depth = depth[session_mask]
            amp = amp[session_mask]
            clu = clu[session_mask]

    unit_mask = np.isin(clu, np.asarray(sorted(allowed_unit_ids), dtype=int))
    st = st[unit_mask]
    depth = depth[unit_mask]
    amp = amp[unit_mask]
    clu = clu[unit_mask]

    if st.size == 0:
        raise ValueError(f"No spikes found for {spec.mouse} {spec.label}")

    st = st - float(st.min())
    cluster_depth = {
        int(cid): float(np.median(depth[clu == cid]))
        for cid in np.unique(clu)
    }

    return {
        "recording_index": recording_index,
        "label": spec.label,
        "short_label": "",
        "spec": spec,
        "time_s": st,
        "depth_um": depth,
        "amp_uv": amp,
        "cluster_id": clu,
        "duration_s": float(st.max()),
        "cluster_depth_um": cluster_depth,
    }


def compute_depth_offsets(session_spikes: list[dict[str, object]], unit_df: pd.DataFrame) -> dict[int, float]:
    offsets = {1: 0.0}
    ref_depth_map = session_spikes[0].get("registered_cluster_depth_um", session_spikes[0]["cluster_depth_um"])
    ref = unit_df[unit_df["recording_index"] == 1]
    ref_depth = {
        int(row.uid_conservative): ref_depth_map[int(row.unit_id)]
        for row in ref.itertuples()
        if int(row.unit_id) in ref_depth_map
    }
    for data in session_spikes[1:]:
        cluster_depth_map = data.get("registered_cluster_depth_um", data["cluster_depth_um"])
        current = unit_df[unit_df["recording_index"] == data["recording_index"]]
        diffs = []
        for row in current.itertuples():
            uid = int(row.uid_conservative)
            unit_id = int(row.unit_id)
            if uid not in ref_depth or unit_id not in cluster_depth_map:
                continue
            diffs.append(ref_depth[uid] - cluster_depth_map[unit_id])
        offsets[data["recording_index"]] = float(np.median(diffs)) if diffs else 0.0
    return offsets


def apply_centroid_registered_depths(
    session_spikes: list[dict[str, object]],
    unit_df: pd.DataFrame,
    waveform_info: np.lib.npyio.NpzFile,
) -> None:
    avg_centroid = waveform_info["avg_centroid"]
    registered_depth = np.nanmean(avg_centroid[2], axis=1)

    unit_rows = unit_df.dropna(subset=["global_index"]).copy()
    unit_rows["global_index"] = unit_rows["global_index"].astype(int)

    for data in session_spikes:
        session_units = unit_rows[unit_rows["recording_index"] == data["recording_index"]]
        reg_map = {
            int(row.unit_id): float(registered_depth[int(row.global_index)])
            for row in session_units.itertuples(index=False)
        }
        shift_map = {
            cluster_id: reg_map[cluster_id] - raw_depth
            for cluster_id, raw_depth in data["cluster_depth_um"].items()
            if cluster_id in reg_map
        }
        shifts = np.asarray([shift_map.get(int(cid), 0.0) for cid in data["cluster_id"]], dtype=float)
        data["registered_depth_um"] = data["depth_um"] + shifts
        data["registered_cluster_depth_um"] = {
            cluster_id: raw_depth + shift_map.get(cluster_id, 0.0)
            for cluster_id, raw_depth in data["cluster_depth_um"].items()
        }


def rank_tracked_uids(match_table: pd.DataFrame) -> pd.DataFrame:
    same = match_table[
        (match_table["UID Conservative 1"] == match_table["UID Conservative 2"])
        & (match_table["RecSes 1"] < match_table["RecSes 2"])
    ].copy()
    rows = []
    for uid, group in same.groupby("UID Conservative 1"):
        sessions = sorted(
            set(group["RecSes 1"].astype(int)).union(group["RecSes 2"].astype(int))
        )
        rows.append(
            {
                "uid_conservative": int(uid),
                "recording_count": len(sessions),
                "mean_same_uid_probability": float(group["UM Probabilities"].mean()),
                "min_same_uid_probability": float(group["UM Probabilities"].min()),
                "pair_count": int(len(group)),
                "recordings": sessions,
            }
        )
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    return ranked.sort_values(
        ["recording_count", "mean_same_uid_probability", "min_same_uid_probability"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def add_time_offsets(session_spikes: list[dict[str, object]], gap_s: float = 120.0) -> None:
    current = 0.0
    for data in session_spikes:
        data["time_offset_s"] = current
        data["time_plot_min"] = (data["time_s"] + current) / 60.0
        current += data["duration_s"] + gap_s


def plot_all_spikes(
    out_path: Path,
    mouse: str,
    session_spikes: list[dict[str, object]],
    depth_offsets: dict[int, float],
    max_points_per_session: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15.5, 7.5), constrained_layout=True)

    all_log_amp = []
    for data in session_spikes:
        idx = deterministic_subsample(len(data["time_s"]), max_points_per_session)
        log_amp = np.log10(np.clip(data["amp_uv"][idx], 1.0, None))
        all_log_amp.append(log_amp)
    vmin = float(np.percentile(np.concatenate(all_log_amp), 2))
    vmax = float(np.percentile(np.concatenate(all_log_amp), 98))

    scatter = None
    centers = []
    boundaries = []
    for data in session_spikes:
        idx = deterministic_subsample(len(data["time_s"]), max_points_per_session)
        y = data["registered_depth_um"][idx] + depth_offsets[data["recording_index"]]
        scatter = ax.scatter(
            data["time_plot_min"][idx],
            y,
            c=np.log10(np.clip(data["amp_uv"][idx], 1.0, None)),
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            s=0.35,
            alpha=0.28,
            linewidths=0.0,
            rasterized=True,
        )
        xmin = data["time_offset_s"] / 60.0
        xmax = (data["time_offset_s"] + data["duration_s"]) / 60.0
        centers.append((xmin + xmax) / 2)
        boundaries.append(xmax)

    for boundary in boundaries[:-1]:
        ax.axvline(boundary, color="#111827", ls="--", lw=1.0, alpha=0.6)

    ax.set_xticks(centers)
    ax.set_xticklabels([data["short_label"] for data in session_spikes], fontsize=10)
    ax.set_xlabel("Recording block")
    ax.set_ylabel("Anchor-registered depth (um)")
    ax.set_title(f"{mouse}: spike depth vs time for all UnitMatch units")
    ax.invert_yaxis()
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10 spike amplitude")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_matched_units(
    out_path: Path,
    mouse: str,
    session_spikes: list[dict[str, object]],
    unit_df: pd.DataFrame,
    selected_uids: pd.DataFrame,
    depth_offsets: dict[int, float],
    max_background_points: int,
    max_highlight_points: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(15.5, 7.5), constrained_layout=True)

    centers = []
    boundaries = []
    for data in session_spikes:
        bg_idx = deterministic_subsample(len(data["time_s"]), max_background_points)
        y = data["registered_depth_um"][bg_idx] + depth_offsets[data["recording_index"]]
        ax.scatter(
            data["time_plot_min"][bg_idx],
            y,
            s=0.3,
            color="#9ca3af",
            alpha=0.08,
            linewidths=0.0,
            rasterized=True,
        )
        xmin = data["time_offset_s"] / 60.0
        xmax = (data["time_offset_s"] + data["duration_s"]) / 60.0
        centers.append((xmin + xmax) / 2)
        boundaries.append(xmax)

    cmap = plt.get_cmap("tab20")
    legend_handles = []
    for color_idx, row in enumerate(selected_uids.itertuples(index=False)):
        color = cmap(color_idx % 20)
        uid_units = unit_df[unit_df["uid_conservative"] == row.uid_conservative]
        for session_row in uid_units.itertuples(index=False):
            data = session_spikes[int(session_row.recording_index) - 1]
            mask = data["cluster_id"] == int(session_row.unit_id)
            if not np.any(mask):
                continue
            time_vals = data["time_plot_min"][mask]
            depth_vals = data["registered_depth_um"][mask] + depth_offsets[data["recording_index"]]
            idx = deterministic_subsample(len(time_vals), max_highlight_points)
            ax.scatter(
                time_vals[idx],
                depth_vals[idx],
                s=0.55,
                color=color,
                alpha=0.62,
                linewidths=0.0,
                rasterized=True,
            )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markersize=6,
                label=f"UID {row.uid_conservative} ({row.recording_count} recs)",
            )
        )

    for boundary in boundaries[:-1]:
        ax.axvline(boundary, color="#111827", ls="--", lw=1.0, alpha=0.6)

    ax.set_xticks(centers)
    ax.set_xticklabels([data["short_label"] for data in session_spikes], fontsize=10)
    ax.set_xlabel("Recording block")
    ax.set_ylabel("Anchor-registered depth (um)")
    ax.set_title(f"{mouse}: conservative matched units highlighted across recordings")
    ax.invert_yaxis()
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_waveform_gallery(
    out_path: Path,
    mouse: str,
    selected_uids: pd.DataFrame,
    unit_df: pd.DataFrame,
    session_labels: list[str],
    waveform_info: np.lib.npyio.NpzFile,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    avg_waveform = waveform_info["avg_waveform"]
    short_labels = short_session_labels(session_labels)
    session_colors = {
        idx + 1: color
        for idx, color in enumerate(["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"])
    }

    rows = list(selected_uids.itertuples(index=False))
    ncols = 3
    nrows = int(math.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15.5, 3.4 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    x = np.arange(avg_waveform.shape[0])

    for ax, row in zip(axes.flat, rows):
        uid_rows = unit_df[unit_df["uid_conservative"] == row.uid_conservative].sort_values("recording_index")
        for unit_row in uid_rows.itertuples(index=False):
            recording_index = int(unit_row.recording_index)
            global_index = int(unit_row.global_index)
            wf0 = avg_waveform[:, global_index, 0]
            wf1 = avg_waveform[:, global_index, 1]
            wf_mean = np.where(
                np.isnan(wf0),
                wf1,
                np.where(np.isnan(wf1), wf0, (wf0 + wf1) / 2.0),
            )
            color = session_colors[recording_index]
            ax.plot(x, wf_mean, color=color, lw=2.0, label=short_labels[recording_index - 1])
            ax.fill_between(
                x,
                np.minimum(wf0, wf1),
                np.maximum(wf0, wf1),
                color=color,
                alpha=0.16,
                linewidth=0.0,
            )
        ax.axhline(0.0, color="#9ca3af", lw=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"UID {row.uid_conservative}\n"
            f"{row.recording_count} recs, mean p={row.mean_same_uid_probability:.3f}",
            fontsize=10,
        )

    for ax in axes.flat[len(rows):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=min(4, len(handles)),
        frameon=False,
    )
    fig.suptitle(f"{mouse}: superimposed average waveforms for tracked units", y=1.05)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(
    selected_uids: pd.DataFrame,
    unit_df: pd.DataFrame,
    session_labels: list[str],
) -> pd.DataFrame:
    session_name_map = {idx + 1: label for idx, label in enumerate(session_labels)}
    rows = []
    for row in selected_uids.itertuples(index=False):
        uid_rows = unit_df[unit_df["uid_conservative"] == row.uid_conservative].sort_values("recording_index")
        session_units = {
            session_name_map[int(unit_row.recording_index)]: int(unit_row.unit_id)
            for unit_row in uid_rows.itertuples(index=False)
        }
        rows.append(
            {
                "uid_conservative": int(row.uid_conservative),
                "recording_count": int(row.recording_count),
                "mean_same_uid_probability": float(row.mean_same_uid_probability),
                "min_same_uid_probability": float(row.min_same_uid_probability),
                "session_unit_ids": json.dumps(session_units, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    root = repo_root()
    configure_runtime(root)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (root / "mouse_tracking_visuals" / args.mouse).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = root / "UnitMatchData"
    generated_label_dir = root / "local_runs" / "full_data_classic" / "generated_labels"
    sessions, _ = enumerate_sessions(data_root, generated_label_dir)
    session_specs = [spec for spec in sessions if spec.mouse == args.mouse]
    if not session_specs:
        raise ValueError(f"No sessions found for mouse {args.mouse}")

    classic_dir = root / "local_runs" / "multi_session_by_mouse" / args.mouse / "classic"
    if not classic_dir.exists():
        raise ValueError(f"Missing classic multi-session output for {args.mouse}: {classic_dir}")

    unit_df = build_global_unit_table(classic_dir)
    match_table = pd.read_csv(classic_dir / "MatchTable.csv")
    waveform_info = np.load(classic_dir / "WaveformInfo.npz")
    ranked_uids = rank_tracked_uids(match_table)
    if ranked_uids.empty:
        raise ValueError(f"No multi-session tracked UIDs found for {args.mouse}")

    max_recordings = int(ranked_uids["recording_count"].max())
    best_uids = ranked_uids[ranked_uids["recording_count"] == max_recordings].head(args.top_uids).copy()
    if len(best_uids) < args.top_uids:
        fallback = ranked_uids[
            ~ranked_uids["uid_conservative"].isin(best_uids["uid_conservative"])
        ].head(args.top_uids - len(best_uids))
        best_uids = pd.concat([best_uids, fallback], ignore_index=True)

    session_spikes = []
    short_labels = short_session_labels([spec.label for spec in session_specs])
    for idx, (spec, short_label) in enumerate(zip(session_specs, short_labels), start=1):
        allowed_units = set(unit_df.loc[unit_df["recording_index"] == idx, "unit_id"].astype(int))
        data = load_session_spikes(spec, idx, allowed_units)
        data["short_label"] = short_label
        session_spikes.append(data)

    apply_centroid_registered_depths(session_spikes, unit_df, waveform_info)
    depth_offsets = compute_depth_offsets(session_spikes, unit_df)
    add_time_offsets(session_spikes)

    all_spikes_path = output_dir / "all_units_depth_vs_time.png"
    matched_path = output_dir / "matched_units_depth_vs_time.png"
    waveform_path = output_dir / "tracked_waveform_gallery.png"
    summary_csv = output_dir / "selected_tracked_units.csv"
    summary_json = output_dir / "summary.json"

    plot_all_spikes(
        all_spikes_path,
        args.mouse,
        session_spikes,
        depth_offsets,
        args.max_all_spikes,
    )
    plot_matched_units(
        matched_path,
        args.mouse,
        session_spikes,
        unit_df,
        best_uids,
        depth_offsets,
        args.max_background_spikes,
        args.max_highlight_spikes,
    )
    plot_waveform_gallery(
        waveform_path,
        args.mouse,
        best_uids,
        unit_df,
        [spec.label for spec in session_specs],
        waveform_info,
    )

    summary_df = build_summary_table(best_uids, unit_df, [spec.label for spec in session_specs])
    summary_df.to_csv(summary_csv, index=False)

    result = {
        "mouse": args.mouse,
        "session_labels": [spec.label for spec in session_specs],
        "short_labels": short_labels,
        "depth_offsets_um": depth_offsets,
        "depth_registration": "registered unit centroid depth from WaveformInfo.avg_centroid[2] plus original per-spike residual from PreparedData spikeDepths",
        "all_units_plot": str(all_spikes_path),
        "matched_units_plot": str(matched_path),
        "waveform_gallery_plot": str(waveform_path),
        "selected_uids_csv": str(summary_csv),
    }
    summary_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
