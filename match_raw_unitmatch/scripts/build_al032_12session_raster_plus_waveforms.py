#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/scratch/am15577/UnitMatch/match_raw_unitmatch/tmp/matplotlib"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
import mtscomp
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SPIKE_ROSTER_ROOT = PROJECT_ROOT / "spike_rosters_raw"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SPIKE_ROSTER_ROOT))

from _pipeline_utils import dump_json, now_iso  # noqa: E402
from build_al032_12session_raster import (  # noqa: E402
    SessionConfig as RasterSessionConfig,
    build_counts_and_overlay_table,
    counts_to_region,
    determine_depth_range,
    load_session_configs,
    overlay_counts,
)
from build_localized_single_session_raster import PlotLayout, scale_values  # noqa: E402
from build_single_session_waveform_overlay import (  # noqa: E402
    extract_cluster_waveforms_batched,
    nice_scale_value,
    parse_meta,
)


@dataclass
class WaveformSessionBundle:
    session_name: str
    session_date: str
    raw_dir: Path
    ks_dir: Path
    attached_csv_gz: Path
    cluster_summary_csv: Path
    cbin_path: Path
    ch_path: Path
    meta_path: Path
    sample_rate_hz: float
    duration_s: float
    block_start_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an AL032 12-session tracked raster plus waveform overlays figure."
    )
    parser.add_argument(
        "--config",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/configs/unitmatch_run_config.json",
    )
    parser.add_argument(
        "--selected-tracked-units-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/tracked_tables/selected_tracked_units.csv",
    )
    parser.add_argument(
        "--session-manifest-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/manifests/unitmatch_input_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/figures",
    )
    parser.add_argument("--candidate-select-count", type=int, default=8)
    parser.add_argument("--final-select-count", type=int, default=6)
    parser.add_argument("--min-waveform-sessions", type=int, default=10)
    parser.add_argument("--waveform-min-good-spikes", type=int, default=5000)
    parser.add_argument("--waveform-min-candidate-spikes", type=int, default=1000)
    parser.add_argument("--waveform-sample-count", type=int, default=250)
    parser.add_argument("--waveform-oversample-count", type=int, default=400)
    parser.add_argument("--waveform-min-aligned-spikes", type=int, default=180)
    parser.add_argument("--waveform-display-traces-per-session", type=int, default=40)
    parser.add_argument("--pre-samples", type=int, default=20)
    parser.add_argument("--post-samples", type=int, default=40)
    parser.add_argument("--alignment-padding", type=int, default=16)
    parser.add_argument("--waveform-read-window-seconds", type=float, default=10.0)
    parser.add_argument("--session-gap-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--force", action="store_true")
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


def parse_json_mapping(value: object) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    payload = json.loads(text)
    return {str(k): int(v) for k, v in payload.items()}


def choose_depth_even_subset_df(df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    if df.empty or df.shape[0] <= target_count:
        return df.sort_values("depth_center_um").reset_index(drop=True)

    ordered = df.sort_values("depth_center_um").reset_index(drop=True)
    raw_indices = np.round(np.linspace(0, len(ordered) - 1, target_count)).astype(int)
    chosen_indices: list[int] = []
    used: set[int] = set()
    for idx in raw_indices:
        idx = int(idx)
        if idx not in used:
            chosen_indices.append(idx)
            used.add(idx)
    if len(chosen_indices) < target_count:
        for idx in range(len(ordered)):
            if idx in used:
                continue
            chosen_indices.append(idx)
            used.add(idx)
            if len(chosen_indices) == target_count:
                break
    return ordered.iloc[chosen_indices].sort_values("depth_center_um").reset_index(drop=True)


def build_waveform_session_bundles(
    config: dict,
    session_manifest_csv: Path,
    session_gap_s: float,
) -> tuple[list[WaveformSessionBundle], list[RasterSessionConfig]]:
    manifest_df = pd.read_csv(session_manifest_csv).sort_values("session_order")
    manifest_rows = {str(row.session_name): row for row in manifest_df.itertuples(index=False)}
    localization_root = Path(config["localization_root_for_attachment"])

    waveform_sessions: list[WaveformSessionBundle] = []
    raster_sessions: list[RasterSessionConfig] = []
    block_start = 0.0
    for session_name in config["session_names"]:
        row = manifest_rows[session_name]
        ks_dir = Path(row.ks_path)
        raw_dir = ks_dir.parent
        meta_path = next(raw_dir.glob("*.ap.meta"))
        cbin_path = next(raw_dir.glob("*.ap.cbin"))
        ch_path = next(raw_dir.glob("*.ap.ch"))
        meta = parse_meta(meta_path)
        sample_rate_hz = float(meta.get("imSampRate", row.sample_rate_hz))
        duration_s = float(row.duration_s)
        attached_csv_gz = Path(config["attached_spikes_root"]) / session_name / f"{session_name}_tracked_spikes.csv.gz"
        cluster_summary_csv = localization_root / session_name / f"{session_name}_good_cluster_summary.csv"
        waveform_sessions.append(
            WaveformSessionBundle(
                session_name=session_name,
                session_date=session_name.replace("AL032_", ""),
                raw_dir=raw_dir,
                ks_dir=ks_dir,
                attached_csv_gz=attached_csv_gz,
                cluster_summary_csv=cluster_summary_csv,
                cbin_path=cbin_path,
                ch_path=ch_path,
                meta_path=meta_path,
                sample_rate_hz=sample_rate_hz,
                duration_s=duration_s,
                block_start_s=block_start,
            )
        )
        raster_sessions.append(
            RasterSessionConfig(
                session_name=session_name,
                session_date=session_name.replace("AL032_", ""),
                attached_csv_gz=attached_csv_gz,
                duration_s=duration_s,
                block_start_s=block_start,
            )
        )
        block_start += duration_s + float(session_gap_s)

    return waveform_sessions, raster_sessions


def corrupted_exclusion_intervals(
    localization_root: Path,
    session_name: str,
    read_window_seconds: float,
) -> list[tuple[float, float]]:
    summary_path = localization_root / session_name / f"{session_name}_corrupted_region_summary.json"
    if not summary_path.exists():
        return []

    payload = json.loads(summary_path.read_text())
    intervals: list[tuple[float, float]] = []
    for region in payload.get("corrupted_regions", []):
        start = float(region.get("skip_time_start_s", region.get("time_start_s", 0.0)))
        end = float(region.get("skip_time_end_s", region.get("time_end_s", 0.0)))
        expanded_start = math.floor(start / read_window_seconds) * read_window_seconds
        expanded_end = math.ceil(end / read_window_seconds) * read_window_seconds
        intervals.append((expanded_start, expanded_end))

    if not intervals:
        return []
    intervals.sort()
    merged: list[list[float]] = [[intervals[0][0], intervals[0][1]]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def load_cluster_summary(bundle: WaveformSessionBundle) -> pd.DataFrame:
    return pd.read_csv(bundle.cluster_summary_csv).set_index("cluster_id", drop=False)


def load_candidate_spike_times(
    tracked_spike_table_path: Path,
    tracked_unit_ids: set[int],
    read_pre_samples: int,
    read_post_samples: int,
    n_samples: int,
    exclusion_intervals_s: list[tuple[float, float]],
) -> dict[int, np.ndarray]:
    usecols = [
        "spike_time_samples",
        "spike_time_s",
        "tracked_unit_id",
        "conflict_flag",
        "is_good_cluster",
        "localization_success",
        "skip_reason",
    ]
    dtypes = {
        "spike_time_samples": "int64",
        "spike_time_s": "float64",
        "tracked_unit_id": "float64",
        "is_good_cluster": "bool",
        "localization_success": "bool",
        "skip_reason": "string",
    }
    candidate_chunks: dict[int, list[np.ndarray]] = {tracked_unit_id: [] for tracked_unit_id in tracked_unit_ids}

    for chunk in pd.read_csv(
        tracked_spike_table_path,
        compression="gzip",
        usecols=usecols,
        dtype=dtypes,
        chunksize=500_000,
        low_memory=False,
    ):
        tracked_mask = chunk["tracked_unit_id"].notna().to_numpy()
        if "conflict_flag" in chunk.columns:
            tracked_mask &= ~series_to_bool_mask(chunk["conflict_flag"])
        if not np.any(tracked_mask):
            continue

        chunk = chunk.loc[tracked_mask].copy()
        chunk["tracked_unit_id"] = chunk["tracked_unit_id"].astype(np.int64)
        chunk = chunk.loc[chunk["tracked_unit_id"].isin(tracked_unit_ids)].copy()
        if chunk.empty:
            continue

        keep = (
            chunk["is_good_cluster"].to_numpy(dtype=bool)
            & chunk["localization_success"].to_numpy(dtype=bool)
            & (chunk["skip_reason"].fillna("").astype(str).to_numpy() == "")
            & (chunk["spike_time_samples"].to_numpy(dtype=np.int64) >= read_pre_samples)
            & (chunk["spike_time_samples"].to_numpy(dtype=np.int64) + read_post_samples <= n_samples)
        )
        if exclusion_intervals_s:
            time_values = chunk["spike_time_s"].to_numpy(dtype=np.float64)
            exclusion_mask = np.zeros(time_values.shape[0], dtype=bool)
            for start_s, end_s in exclusion_intervals_s:
                exclusion_mask |= (time_values >= start_s) & (time_values < end_s)
            keep &= ~exclusion_mask
        if not np.any(keep):
            continue

        filtered = chunk.loc[keep, ["tracked_unit_id", "spike_time_samples"]].copy()
        for tracked_unit_id, group in filtered.groupby("tracked_unit_id", sort=False):
            candidate_chunks[int(tracked_unit_id)].append(group["spike_time_samples"].to_numpy(dtype=np.int64))

    return {
        tracked_unit_id: np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
        for tracked_unit_id, chunks in candidate_chunks.items()
    }


def choose_common_reference_channel(cluster_rows: list[pd.Series]) -> tuple[int, str]:
    peak_channels = np.asarray([int(row["peak_channel_index"]) for row in cluster_rows], dtype=np.int64)
    unique, counts = np.unique(peak_channels, return_counts=True)
    winner_count = int(counts.max())
    winners = unique[counts == winner_count]
    if winners.size == 1:
        return int(winners[0]), "mode_peak_channel"
    median_peak = float(np.median(peak_channels))
    best = winners[np.argmin(np.abs(winners.astype(np.float64) - median_peak))]
    return int(best), "mode_peak_channel_tie_broken_by_median"


def choose_session_colors(session_names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    colors: dict[str, tuple[float, float, float, float]] = {}
    for idx, session_name in enumerate(session_names):
        colors[session_name] = tuple(float(v) for v in cmap(idx % 20))
    return colors


def build_raster_region(
    raster_sessions: list[RasterSessionConfig],
    selected_units: pd.DataFrame,
    output_overlay_csv_gz: Path,
    session_gap_s: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, int]]:
    plot_layout = PlotLayout(canvas_w=3200, canvas_h=1650, plot_left=130, plot_top=110, plot_w=2350, plot_h=1280, legend_left=2530, legend_top=150)
    total_duration_s = max(config.block_start_s + config.duration_s for config in raster_sessions)
    time_pad = max(total_duration_s * 0.005, 2.0)
    time_min = -time_pad
    time_max = total_duration_s + time_pad

    depth_min_raw, depth_max_raw, background_counts_by_session = determine_depth_range(raster_sessions)
    depth_pad = max((depth_max_raw - depth_min_raw) * 0.03, 20.0)
    depth_min = depth_min_raw - depth_pad
    depth_max = depth_max_raw + depth_pad

    background_counts, overlay_counts_by_unit, overlay_counts_by_session = build_counts_and_overlay_table(
        session_configs=raster_sessions,
        selected_units=selected_units,
        plot_w=plot_layout.plot_w,
        plot_h=plot_layout.plot_h,
        time_min=time_min,
        time_max=time_max,
        depth_min=depth_min,
        depth_max=depth_max,
        overlay_output_csv_gz=output_overlay_csv_gz,
    )

    region = counts_to_region(background_counts)
    for row in selected_units.itertuples(index=False):
        overlay_counts(region, overlay_counts_by_unit[int(row.tracked_unit_id)], (int(row.color_r), int(row.color_g), int(row.color_b)))

    meta = {
        "time_min": time_min,
        "time_max": time_max,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "session_gap_s": float(session_gap_s),
        "depth_min_raw": float(depth_min_raw),
        "depth_max_raw": float(depth_max_raw),
    }
    meta["background_good_spike_counts_by_session"] = background_counts_by_session
    meta["overlay_selected_spike_counts_by_session"] = overlay_counts_by_session
    return region, meta, overlay_counts_by_session


def choose_waveform_unit_subset(selected_units: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    candidates = selected_units.loc[
        (selected_units["n_sessions_present"] >= int(args.min_waveform_sessions))
        & (selected_units["min_good_tracked_spikes"] >= int(args.waveform_min_good_spikes))
    ].copy()
    if candidates.empty:
        raise RuntimeError("No selected tracked units satisfy the waveform candidate thresholds")

    candidates = candidates.sort_values(
        ["n_sessions_present", "min_good_tracked_spikes", "mean_cross_session_probability", "depth_center_um"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    candidates = choose_depth_even_subset_df(candidates, min(int(args.candidate_select_count), int(candidates.shape[0])))
    return candidates.reset_index(drop=True)


def pad_waveforms_and_spike_times(
    waveforms: np.ndarray,
    spike_times: np.ndarray,
    sample_count: int,
    waveform_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    padded_waveforms = np.full((sample_count, waveform_width), np.nan, dtype=np.float32)
    padded_spike_times = np.full(sample_count, -1, dtype=np.int64)
    keep = min(sample_count, int(waveforms.shape[0]))
    if keep > 0:
        padded_waveforms[:keep] = waveforms[:keep].astype(np.float32, copy=False)
        padded_spike_times[:keep] = spike_times[:keep].astype(np.int64, copy=False)
    return padded_waveforms, padded_spike_times


def extract_session_waveforms(
    reader: mtscomp.Reader,
    spike_times: np.ndarray,
    reference_channel: int,
    args: argparse.Namespace,
    read_window_frames: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    if spike_times.size < args.waveform_min_candidate_spikes:
        return (
            np.full((args.waveform_sample_count, args.pre_samples + args.post_samples + 1), np.nan, dtype=np.float32),
            np.full(args.waveform_sample_count, -1, dtype=np.int64),
            0,
        )

    draw_count = min(int(spike_times.size), max(int(args.waveform_oversample_count), int(args.waveform_sample_count)))
    chosen = rng.choice(spike_times.size, size=draw_count, replace=False)
    sampled = np.sort(spike_times[chosen])
    waveforms, kept_spikes, _, _ = extract_cluster_waveforms_batched(
        reader=reader,
        spike_times_samples=sampled,
        reference_channel=reference_channel,
        pre_samples=int(args.pre_samples),
        post_samples=int(args.post_samples),
        alignment_padding=int(args.alignment_padding),
        read_window_frames=read_window_frames,
    )
    aligned_count = int(waveforms.shape[0])
    if aligned_count > int(args.waveform_sample_count):
        keep_idx = rng.choice(aligned_count, size=int(args.waveform_sample_count), replace=False)
        keep_idx.sort()
        waveforms = waveforms[keep_idx]
        kept_spikes = kept_spikes[keep_idx]
        aligned_count = int(waveforms.shape[0])

    if aligned_count < int(args.waveform_min_aligned_spikes):
        aligned_count = 0
        waveforms = np.empty((0, args.pre_samples + args.post_samples + 1), dtype=np.float32)
        kept_spikes = np.empty(0, dtype=np.int64)

    padded_waveforms, padded_spike_times = pad_waveforms_and_spike_times(
        waveforms=waveforms,
        spike_times=kept_spikes,
        sample_count=int(args.waveform_sample_count),
        waveform_width=int(args.pre_samples + args.post_samples + 1),
    )
    return padded_waveforms, padded_spike_times, aligned_count


def choose_final_waveform_entries(entries: list[dict[str, object]], target_count: int) -> list[dict[str, object]]:
    if not entries:
        raise RuntimeError("No tracked units produced waveform overlays across enough sessions")
    ordered = sorted(
        entries,
        key=lambda item: (
            -int(item["extracted_session_count"]),
            -int(item["min_aligned_spike_count"]),
            -float(item["mean_cross_session_probability"]),
            float(item["depth_center_um"]),
        ),
    )
    pool = pd.DataFrame(
        {
            "tracked_unit_id": [int(entry["tracked_unit_id"]) for entry in ordered],
            "depth_center_um": [float(entry["depth_center_um"]) for entry in ordered],
        }
    )
    chosen_ids = set(choose_depth_even_subset_df(pool, min(target_count, len(ordered)))["tracked_unit_id"].astype(int).tolist())
    selected = [entry for entry in ordered if int(entry["tracked_unit_id"]) in chosen_ids]
    return sorted(selected, key=lambda item: float(item["depth_center_um"]))


def plot_raster_plus_waveforms(
    region: np.ndarray,
    raster_meta: dict[str, float],
    selected_units: pd.DataFrame,
    waveform_entries: list[dict[str, object]],
    session_names: list[str],
    sample_rate_hz: float,
    pre_samples: int,
    post_samples: int,
    output_path: Path,
) -> None:
    n_units = len(waveform_entries)
    fig = plt.figure(figsize=(20.0, max(11.0, 2.55 * n_units)), dpi=220)
    gs = fig.add_gridspec(n_units, 2, width_ratios=[3.3, 1.65], wspace=0.18, hspace=0.18)

    ax_raster = fig.add_subplot(gs[:, 0])
    ax_raster.imshow(
        region,
        extent=[
            raster_meta["time_min"],
            raster_meta["time_max"],
            raster_meta["depth_max"],
            raster_meta["depth_min"],
        ],
        aspect="auto",
        interpolation="nearest",
    )
    ax_raster.set_title(
        "AL032 tracked units across 12 sessions\nmonopolar triangulation raster with raw waveform overlays",
        fontsize=15,
    )
    ax_raster.set_xlabel("session blocks (60 s gaps)")
    ax_raster.set_ylabel("depth (um)")

    session_centers = []
    for entry in waveform_entries[0]["session_metadata"]:
        block_start = float(entry["block_start_s"])
        duration = float(entry["duration_s"])
        session_centers.append(block_start + duration / 2.0)
    ax_raster.set_xticks(session_centers)
    ax_raster.set_xticklabels([name.replace("AL032_", "")[5:] for name in session_names], rotation=0, fontsize=8)

    for idx in range(len(session_names) - 1):
        boundary = waveform_entries[0]["session_metadata"][idx]["block_start_s"] + waveform_entries[0]["session_metadata"][idx]["duration_s"] + raster_meta["session_gap_s"] / 2.0
        ax_raster.axvline(boundary, color=(0.35, 0.35, 0.35), linestyle="--", linewidth=0.8, alpha=0.75)

    raster_handles: list[Line2D] = []
    raster_labels: list[str] = []
    for row in selected_units.itertuples(index=False):
        color = np.asarray([row.color_r, row.color_g, row.color_b], dtype=np.float64) / 255.0
        raster_handles.append(Line2D([0], [0], color=color, marker="o", linestyle="", markersize=5))
        raster_labels.append(f"T{int(row.tracked_unit_id)}")
    ax_raster.legend(
        raster_handles,
        raster_labels,
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=8.5,
        title="tracked units",
    )

    session_colors = choose_session_colors(session_names)
    session_handles = [Line2D([0], [0], color=session_colors[name], linewidth=2.0) for name in session_names]
    session_labels = [name.replace("AL032_", "")[5:] for name in session_names]

    waveform_arrays = []
    for entry in waveform_entries:
        waveform_arrays.append(np.asarray(entry["waveforms"], dtype=np.float32))
    stacked = np.concatenate([arr[np.isfinite(arr)] for arr in waveform_arrays if np.isfinite(arr).any()])
    y_abs = np.nanpercentile(np.abs(stacked), 99.5)
    y_limit = max(float(nice_scale_value(y_abs * 1.12)), 1.0)
    time_ms = (np.arange(pre_samples + post_samples + 1, dtype=np.float64) - pre_samples) / sample_rate_hz * 1000.0

    panel_axes = [fig.add_subplot(gs[idx, 1]) for idx in range(n_units)]
    for idx, (ax, entry) in enumerate(zip(panel_axes, waveform_entries)):
        waveforms = np.asarray(entry["waveforms"], dtype=np.float32)
        present_mask = np.asarray(entry["present_mask"], dtype=bool)

        for session_index, session_name in enumerate(session_names):
            if not present_mask[session_index]:
                continue
            valid_rows = np.all(np.isfinite(waveforms[session_index]), axis=1)
            if not np.any(valid_rows):
                continue
            session_waveforms = waveforms[session_index][valid_rows]
            color = session_colors[session_name]
            display_count = min(int(entry["display_trace_count"]), int(session_waveforms.shape[0]))
            if display_count > 0:
                ax.plot(
                    time_ms,
                    session_waveforms[:display_count].T,
                    color=color,
                    alpha=0.010,
                    linewidth=0.18,
                )
            mean_wave = session_waveforms.mean(axis=0)
            std_wave = session_waveforms.std(axis=0, ddof=1) if session_waveforms.shape[0] > 1 else np.zeros_like(mean_wave)
            ax.fill_between(
                time_ms,
                mean_wave - std_wave,
                mean_wave + std_wave,
                color=color,
                alpha=0.08,
                linewidth=0.0,
            )
            ax.plot(time_ms, mean_wave, color=color, linewidth=1.25)

        ax.axvline(0.0, color=(0.72, 0.72, 0.72), linewidth=0.8, linestyle="--")
        ax.axhline(0.0, color=(0.9, 0.9, 0.9), linewidth=0.7)
        ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))
        ax.set_ylim(-y_limit, y_limit)
        ax.set_title(
            (
                f"{entry['tracked_label']} | sess {int(entry['extracted_session_count'])} | "
                f"ref ch {int(entry['reference_channel'])} | depth {float(entry['depth_center_um']):.1f} um"
            ),
            fontsize=9,
            color=np.asarray(entry["color_rgb"], dtype=np.float64) / 255.0,
        )
        if idx < n_units - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time (ms)", fontsize=9)
        ax.tick_params(labelsize=8)
        if idx == math.floor(n_units / 2):
            ax.set_ylabel("baseline-subtracted raw units", fontsize=9)
        else:
            ax.set_yticklabels([])

        connector = ConnectionPatch(
            xyA=(raster_meta["time_max"], float(entry["depth_center_um"])),
            coordsA="data",
            xyB=(0.0, 0.5),
            coordsB="axes fraction",
            color=np.asarray(entry["color_rgb"], dtype=np.float64) / 255.0,
            linewidth=0.9,
            linestyle=(0, (3, 3)),
            alpha=0.65,
            axesA=ax_raster,
            axesB=ax,
        )
        fig.add_artist(connector)

    fig.legend(
        session_handles,
        session_labels,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.987),
        frameon=False,
        ncol=4,
        fontsize=8,
        title="waveform session colors",
    )
    fig.text(
        0.78,
        0.955,
        "Per panel: session means with 1 s.d. bands and lightly overplotted aligned raw snippets",
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.985, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text())
    selected_units_all = pd.read_csv(args.selected_tracked_units_csv)
    waveform_candidates = choose_waveform_unit_subset(selected_units_all, args)
    waveform_sessions, raster_sessions = build_waveform_session_bundles(
        config=config,
        session_manifest_csv=Path(args.session_manifest_csv),
        session_gap_s=float(args.session_gap_s),
    )
    session_names = [bundle.session_name for bundle in waveform_sessions]
    session_index_lookup = {name: idx for idx, name in enumerate(session_names)}
    localization_root = Path(config["localization_root_for_attachment"])

    cluster_summaries = {bundle.session_name: load_cluster_summary(bundle) for bundle in waveform_sessions}
    readers = {
        bundle.session_name: mtscomp.decompress(bundle.cbin_path, cmeta=bundle.ch_path)
        for bundle in waveform_sessions
    }
    exclusion_intervals = {
        bundle.session_name: corrupted_exclusion_intervals(localization_root, bundle.session_name, args.waveform_read_window_seconds)
        for bundle in waveform_sessions
    }
    read_pre_samples = int(args.pre_samples + args.alignment_padding)
    read_post_samples = int(args.post_samples + args.alignment_padding + 1)
    candidate_tracked_ids = set(waveform_candidates["tracked_unit_id"].astype(int).tolist())
    candidate_spikes = {
        bundle.session_name: load_candidate_spike_times(
            tracked_spike_table_path=bundle.attached_csv_gz,
            tracked_unit_ids=candidate_tracked_ids,
            read_pre_samples=read_pre_samples,
            read_post_samples=read_post_samples,
            n_samples=int(readers[bundle.session_name].n_samples),
            exclusion_intervals_s=exclusion_intervals[bundle.session_name],
        )
        for bundle in waveform_sessions
    }

    waveform_entries: list[dict[str, object]] = []
    waveform_width = int(args.pre_samples + args.post_samples + 1)
    for row in waveform_candidates.itertuples(index=False):
        cluster_ids_by_session = parse_json_mapping(row.cluster_ids_by_session)
        cluster_rows: list[pd.Series] = []
        available_sessions: list[str] = []
        for session_name, raw_cluster_id in cluster_ids_by_session.items():
            summary = cluster_summaries.get(session_name)
            if summary is None or raw_cluster_id not in summary.index:
                continue
            cluster_rows.append(summary.loc[raw_cluster_id])
            available_sessions.append(session_name)

        if len(cluster_rows) < int(args.min_waveform_sessions):
            continue

        reference_channel, reference_policy = choose_common_reference_channel(cluster_rows)
        waveforms = np.full(
            (len(waveform_sessions), int(args.waveform_sample_count), waveform_width),
            np.nan,
            dtype=np.float32,
        )
        spike_times = np.full((len(waveform_sessions), int(args.waveform_sample_count)), -1, dtype=np.int64)
        raw_cluster_ids = np.full(len(waveform_sessions), -1, dtype=np.int32)
        peak_channels = np.full(len(waveform_sessions), -1, dtype=np.int32)
        aligned_counts = np.zeros(len(waveform_sessions), dtype=np.int32)
        candidate_counts = np.zeros(len(waveform_sessions), dtype=np.int32)
        present_mask = np.zeros(len(waveform_sessions), dtype=bool)

        for session_name, raw_cluster_id in cluster_ids_by_session.items():
            if session_name not in session_index_lookup:
                continue
            session_idx = session_index_lookup[session_name]
            bundle = waveform_sessions[session_idx]
            summary = cluster_summaries[session_name]
            if raw_cluster_id not in summary.index:
                continue
            spikes = candidate_spikes[session_name].get(int(row.tracked_unit_id), np.empty(0, dtype=np.int64))
            candidate_counts[session_idx] = int(spikes.size)
            if spikes.size == 0:
                continue
            read_window_frames = max(1, int(round(args.waveform_read_window_seconds * bundle.sample_rate_hz)))
            padded_waveforms, padded_spikes, aligned_count = extract_session_waveforms(
                reader=readers[session_name],
                spike_times=spikes,
                reference_channel=reference_channel,
                args=args,
                read_window_frames=read_window_frames,
                rng=rng,
            )
            if aligned_count < int(args.waveform_min_aligned_spikes):
                continue
            waveforms[session_idx] = padded_waveforms
            spike_times[session_idx] = padded_spikes
            raw_cluster_ids[session_idx] = int(raw_cluster_id)
            peak_channels[session_idx] = int(summary.loc[raw_cluster_id, "peak_channel_index"])
            aligned_counts[session_idx] = int(aligned_count)
            present_mask[session_idx] = True

        extracted_session_count = int(np.count_nonzero(present_mask))
        if extracted_session_count < int(args.min_waveform_sessions):
            continue

        waveform_entries.append(
            {
                "tracked_unit_id": int(row.tracked_unit_id),
                "tracked_label": str(row.tracked_label),
                "depth_center_um": float(row.depth_center_um),
                "n_sessions_present": int(row.n_sessions_present),
                "mean_cross_session_probability": float(row.mean_cross_session_probability),
                "min_good_tracked_spikes": int(row.min_good_tracked_spikes),
                "reference_channel": int(reference_channel),
                "reference_channel_policy": reference_policy,
                "color_rgb": (int(row.color_r), int(row.color_g), int(row.color_b)),
                "waveforms": waveforms,
                "spike_times": spike_times,
                "raw_cluster_ids": raw_cluster_ids,
                "peak_channels": peak_channels,
                "aligned_counts": aligned_counts,
                "candidate_counts": candidate_counts,
                "present_mask": present_mask,
                "extracted_session_count": extracted_session_count,
                "min_aligned_spike_count": int(aligned_counts[present_mask].min()),
                "display_trace_count": int(args.waveform_display_traces_per_session),
                "session_metadata": [
                    {
                        "session_name": bundle.session_name,
                        "session_date": bundle.session_date,
                        "block_start_s": float(bundle.block_start_s),
                        "duration_s": float(bundle.duration_s),
                    }
                    for bundle in waveform_sessions
                ],
            }
        )

    final_entries = choose_final_waveform_entries(waveform_entries, int(args.final_select_count))
    final_ids = {int(entry["tracked_unit_id"]) for entry in final_entries}
    selected_units_final = waveform_candidates.loc[waveform_candidates["tracked_unit_id"].isin(final_ids)].copy()
    selected_units_final = selected_units_final.sort_values("depth_center_um").reset_index(drop=True)
    selected_units_final["selection_rank"] = np.arange(1, len(selected_units_final) + 1, dtype=np.int64)

    overlay_output_csv_gz = output_dir / "al032_12session_waveform_overlay_spikes.csv.gz"
    waveform_selection_csv = output_dir / "al032_12session_selected_waveform_tracked_units.csv"
    waveform_npz = output_dir / "al032_12session_tracked_unit_waveform_samples.npz"
    composite_png = output_dir / "al032_12session_raster_plus_waveforms.png"
    summary_json = output_dir / "al032_12session_raster_plus_waveforms_summary.json"

    region, raster_meta, overlay_counts_by_session = build_raster_region(
        raster_sessions=raster_sessions,
        selected_units=selected_units_final,
        output_overlay_csv_gz=overlay_output_csv_gz,
        session_gap_s=float(args.session_gap_s),
    )

    selection_rows = []
    waveforms_stack = []
    spike_times_stack = []
    raw_clusters_stack = []
    peak_channels_stack = []
    aligned_counts_stack = []
    candidate_counts_stack = []
    present_masks_stack = []
    reference_channels = []
    tracked_ids = []
    tracked_labels = []
    for selection_rank, entry in enumerate(final_entries, start=1):
        tracked_ids.append(int(entry["tracked_unit_id"]))
        tracked_labels.append(str(entry["tracked_label"]))
        reference_channels.append(int(entry["reference_channel"]))
        waveforms_stack.append(np.asarray(entry["waveforms"], dtype=np.float32))
        spike_times_stack.append(np.asarray(entry["spike_times"], dtype=np.int64))
        raw_clusters_stack.append(np.asarray(entry["raw_cluster_ids"], dtype=np.int32))
        peak_channels_stack.append(np.asarray(entry["peak_channels"], dtype=np.int32))
        aligned_counts_stack.append(np.asarray(entry["aligned_counts"], dtype=np.int32))
        candidate_counts_stack.append(np.asarray(entry["candidate_counts"], dtype=np.int32))
        present_masks_stack.append(np.asarray(entry["present_mask"], dtype=bool))
        selection_rows.append(
            {
                "selection_rank": selection_rank,
                "tracked_unit_id": int(entry["tracked_unit_id"]),
                "tracked_label": str(entry["tracked_label"]),
                "depth_center_um": float(entry["depth_center_um"]),
                "n_sessions_present": int(entry["n_sessions_present"]),
                "waveform_sessions_extracted": int(entry["extracted_session_count"]),
                "min_aligned_spike_count": int(entry["min_aligned_spike_count"]),
                "reference_channel": int(entry["reference_channel"]),
                "reference_channel_policy": str(entry["reference_channel_policy"]),
                "color_hex": "#{:02x}{:02x}{:02x}".format(*entry["color_rgb"]),
                "session_presence_mask_json": json.dumps(
                    {name: bool(mask) for name, mask in zip(session_names, entry["present_mask"].tolist())},
                    sort_keys=True,
                ),
                "raw_cluster_ids_by_session_json": json.dumps(
                    {name: int(value) for name, value in zip(session_names, entry["raw_cluster_ids"].tolist()) if int(value) >= 0},
                    sort_keys=True,
                ),
                "aligned_counts_by_session_json": json.dumps(
                    {name: int(value) for name, value in zip(session_names, entry["aligned_counts"].tolist()) if int(value) > 0},
                    sort_keys=True,
                ),
                "candidate_counts_by_session_json": json.dumps(
                    {name: int(value) for name, value in zip(session_names, entry["candidate_counts"].tolist()) if int(value) > 0},
                    sort_keys=True,
                ),
            }
        )

    pd.DataFrame(selection_rows).to_csv(waveform_selection_csv, index=False)
    np.savez_compressed(
        waveform_npz,
        tracked_unit_ids=np.asarray(tracked_ids, dtype=np.int64),
        tracked_labels=np.asarray(tracked_labels),
        session_names=np.asarray(session_names),
        waveforms=np.stack(waveforms_stack, axis=0).astype(np.float32, copy=False),
        spike_times=np.stack(spike_times_stack, axis=0).astype(np.int64, copy=False),
        raw_cluster_ids=np.stack(raw_clusters_stack, axis=0).astype(np.int32, copy=False),
        peak_channels=np.stack(peak_channels_stack, axis=0).astype(np.int32, copy=False),
        aligned_counts=np.stack(aligned_counts_stack, axis=0).astype(np.int32, copy=False),
        candidate_counts=np.stack(candidate_counts_stack, axis=0).astype(np.int32, copy=False),
        present_mask=np.stack(present_masks_stack, axis=0).astype(bool, copy=False),
        reference_channels=np.asarray(reference_channels, dtype=np.int32),
        sample_rate_hz=np.asarray([waveform_sessions[0].sample_rate_hz], dtype=np.float64),
        pre_samples=np.asarray([int(args.pre_samples)], dtype=np.int32),
        post_samples=np.asarray([int(args.post_samples)], dtype=np.int32),
        waveform_sample_count=np.asarray([int(args.waveform_sample_count)], dtype=np.int32),
    )

    plot_raster_plus_waveforms(
        region=region,
        raster_meta=raster_meta,
        selected_units=selected_units_final,
        waveform_entries=final_entries,
        session_names=session_names,
        sample_rate_hz=float(waveform_sessions[0].sample_rate_hz),
        pre_samples=int(args.pre_samples),
        post_samples=int(args.post_samples),
        output_path=composite_png,
    )

    dump_json(
        summary_json,
        {
            "created_at": now_iso(),
            "localization_method": str(config["localization_method_for_attachment"]),
            "session_names": session_names,
            "waveform_thresholds": {
                "candidate_select_count": int(args.candidate_select_count),
                "final_select_count": int(args.final_select_count),
                "min_waveform_sessions": int(args.min_waveform_sessions),
                "waveform_min_good_spikes": int(args.waveform_min_good_spikes),
                "waveform_min_candidate_spikes": int(args.waveform_min_candidate_spikes),
                "waveform_sample_count": int(args.waveform_sample_count),
                "waveform_oversample_count": int(args.waveform_oversample_count),
                "waveform_min_aligned_spikes": int(args.waveform_min_aligned_spikes),
                "waveform_read_window_seconds": float(args.waveform_read_window_seconds),
                "session_gap_s": float(args.session_gap_s),
            },
            "candidate_tracked_unit_ids": [int(v) for v in waveform_candidates["tracked_unit_id"].tolist()],
            "selected_tracked_unit_ids": tracked_ids,
            "n_selected_tracked_units": len(tracked_ids),
            "raster_meta": {
                "depth_min_raw_um": float(raster_meta["depth_min_raw"]),
                "depth_max_raw_um": float(raster_meta["depth_max_raw"]),
                "time_min_s": float(raster_meta["time_min"]),
                "time_max_s": float(raster_meta["time_max"]),
                "background_good_spike_counts_by_session": raster_meta["background_good_spike_counts_by_session"],
                "overlay_selected_spike_counts_by_session": overlay_counts_by_session,
            },
            "outputs": {
                "waveform_selection_csv": str(waveform_selection_csv),
                "waveform_samples_npz": str(waveform_npz),
                "overlay_spikes_csv_gz": str(overlay_output_csv_gz),
                "raster_plus_waveforms_png": str(composite_png),
            },
        },
    )


if __name__ == "__main__":
    main()
