#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import json
import math
from dataclasses import dataclass
from pathlib import Path

import mtscomp
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from spikeinterface.core import BaseRecording, BaseRecordingSegment
import spikeinterface.preprocessing as spre
from spikeinterface.sortingcomponents.peak_localization import localize_peaks


SKIP_REASON_NONE = 0
SKIP_REASON_CORRUPTED_RAW_CHUNK = 1
SKIP_REASON_READ_FAILURE = 2
SKIP_REASON_INVALID_PEAK = 3
SKIP_REASON_OTHER = 4
SKIP_REASON_UNRESOLVED_PIPELINE_GAP = 5
SKIP_REASON_CATEGORIES = [
    "",
    "corrupted_raw_chunk",
    "read_failure",
    "invalid_peak",
    "other",
    "unresolved_pipeline_gap",
]
SUPPORTED_LOCALIZATION_METHODS = [
    "center_of_mass",
    "monopolar_triangulation",
]


@dataclass
class PlotLayout:
    canvas_w: int = 2900
    canvas_h: int = 1550
    plot_left: int = 130
    plot_top: int = 110
    plot_w: int = 2100
    plot_h: int = 1220
    legend_left: int = 2270
    legend_top: int = 150


class MTSCompRecordingSegment(BaseRecordingSegment):
    def __init__(self, reader: mtscomp.Reader, sampling_frequency: float, n_neural_channels: int):
        super().__init__(sampling_frequency=sampling_frequency)
        self.reader = reader
        self.n_neural_channels = n_neural_channels

    def get_num_samples(self) -> int:
        return int(self.reader.n_samples)

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        start_frame = 0 if start_frame is None else int(start_frame)
        end_frame = self.reader.n_samples if end_frame is None else int(end_frame)
        traces = np.asarray(self.reader[start_frame:end_frame, : self.n_neural_channels], dtype=np.int16)
        if channel_indices is not None:
            traces = traces[:, channel_indices]
        return traces


class MTSCompRecording(BaseRecording):
    def __init__(
        self,
        cbin_path: Path,
        ch_path: Path,
        sampling_frequency: float,
        channel_locations: np.ndarray,
    ):
        self._reader = mtscomp.decompress(cbin_path, cmeta=ch_path)
        n_neural_channels = int(channel_locations.shape[0])
        channel_ids = np.arange(n_neural_channels)
        super().__init__(sampling_frequency=sampling_frequency, channel_ids=channel_ids, dtype="int16")
        self.add_recording_segment(
            MTSCompRecordingSegment(self._reader, sampling_frequency, n_neural_channels)
        )
        self.set_channel_locations(channel_locations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partial raw-localization pipeline for a single Kilosort session."
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/raw_data/extracted/AL032_2019-11-21"),
        help="Path to the extracted session directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw"),
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--highlight-count",
        type=int,
        default=10,
        help="Number of good clusters to highlight across depth.",
    )
    parser.add_argument(
        "--radius-um",
        type=float,
        default=80.0,
        help="Localization radius in microns for the SpikeInterface localization call.",
    )
    parser.add_argument(
        "--localization-method",
        type=str,
        choices=SUPPORTED_LOCALIZATION_METHODS,
        default="center_of_mass",
        help="SpikeInterface localization method to use for per-spike depth estimation.",
    )
    parser.add_argument(
        "--localization-feature",
        type=str,
        default="ptp",
        help="Feature passed to the localization method, for example ptp or energy.",
    )
    parser.add_argument(
        "--monopolar-max-distance-um",
        type=float,
        default=150.0,
        help="Maximum distance parameter for monopolar_triangulation.",
    )
    parser.add_argument(
        "--coordinate-margin-um",
        type=float,
        default=200.0,
        help="Allowed localization margin beyond the probe y-range before a spike is flagged out of range.",
    )
    parser.add_argument(
        "--ms-before",
        type=float,
        default=0.3,
        help="Milliseconds before the peak for waveform extraction.",
    )
    parser.add_argument(
        "--ms-after",
        type=float,
        default=0.6,
        help="Milliseconds after the peak for waveform extraction.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=str,
        default="10s",
        help="Processing chunk duration, for example 2s or 10s.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of workers for each SpikeInterface localization call.",
    )
    parser.add_argument(
        "--partial-mode",
        action="store_true",
        help="Continue past unreadable raw regions instead of crashing.",
    )
    parser.add_argument(
        "--corrupted-chunk-start",
        type=int,
        default=None,
        help="Inclusive compressed chunk index for the start of a known corrupted region.",
    )
    parser.add_argument(
        "--corrupted-chunk-end",
        type=int,
        default=None,
        help="Inclusive compressed chunk index for the end of a known corrupted region.",
    )
    parser.add_argument(
        "--filter-margin-ms",
        type=float,
        default=5.0,
        help="Filter margin in milliseconds used by SpikeInterface preprocessing.",
    )
    return parser.parse_args()


def parse_meta(meta_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in meta_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def parse_duration_seconds(text: str) -> float:
    text = text.strip().lower()
    if text.endswith("ms"):
        return float(text[:-2]) / 1000.0
    if text.endswith("s"):
        return float(text[:-1])
    return float(text)


def cluster_color(cluster_id: int) -> tuple[int, int, int]:
    hue = (cluster_id * 0.61803398875) % 1.0
    sat = 0.72
    val = 0.96
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


def scale_values(
    values: np.ndarray,
    src_min: float,
    src_max: float,
    dst_min: int,
    dst_max: int,
) -> np.ndarray:
    if src_max == src_min:
        return np.full(values.shape, (dst_min + dst_max) // 2, dtype=np.int32)
    scaled = (values - src_min) / (src_max - src_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.rint(dst_min + scaled * (dst_max - dst_min)).astype(np.int32)


def build_background_region(x_px: np.ndarray, y_px: np.ndarray, plot_h: int, plot_w: int) -> np.ndarray:
    counts = np.zeros((plot_h, plot_w), dtype=np.uint32)
    np.add.at(counts, (y_px, x_px), 1)
    if counts.max() == 0:
        return np.full((plot_h, plot_w, 3), 255, dtype=np.uint8)
    log_counts = np.log1p(counts.astype(np.float32))
    norm = log_counts / float(log_counts.max())
    gray = (255.0 - norm * 160.0).clip(80.0, 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def overlay_cluster_region(
    region: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    color_rgb: tuple[int, int, int],
) -> None:
    counts = np.zeros(region.shape[:2], dtype=np.uint32)
    np.add.at(counts, (y_px, x_px), 1)
    mask = counts > 0
    if not np.any(mask):
        return
    norm = np.log1p(counts[mask].astype(np.float32))
    norm /= float(norm.max())
    alpha = (0.30 + 0.70 * norm).reshape(-1, 1)
    base = region[mask].astype(np.float32)
    color = np.asarray(color_rgb, dtype=np.float32).reshape(1, 3)
    region[mask] = np.clip(base * (1.0 - alpha) + color * alpha, 0.0, 255.0).astype(np.uint8)


def draw_axes(
    draw: ImageDraw.ImageDraw,
    layout: PlotLayout,
    duration_s: float,
    depth_min: float,
    depth_max: float,
    font: ImageFont.ImageFont,
    title: str,
    subtitle: str,
) -> None:
    x0 = layout.plot_left
    y0 = layout.plot_top
    x1 = layout.plot_left + layout.plot_w
    y1 = layout.plot_top + layout.plot_h
    draw.rectangle((x0, y0, x1, y1), outline=(55, 55, 55), width=2)
    draw.text((x0, 38), title, fill=(20, 20, 20), font=font)
    draw.text((x0, 66), subtitle, fill=(55, 55, 55), font=font)

    for frac in np.linspace(0.0, 1.0, 6):
        x = int(round(x0 + frac * layout.plot_w))
        draw.line((x, y1, x, y1 + 8), fill=(55, 55, 55), width=2)
        draw.text((x - 10, y1 + 12), f"{duration_s * frac:.0f}", fill=(20, 20, 20), font=font)

    for frac in np.linspace(0.0, 1.0, 6):
        y = int(round(y0 + frac * layout.plot_h))
        draw.line((x0 - 8, y, x0, y), fill=(55, 55, 55), width=2)
        depth = depth_min + frac * (depth_max - depth_min)
        draw.text((18, y - 6), f"{depth:.0f}", fill=(20, 20, 20), font=font)

    draw.text((layout.plot_left + layout.plot_w // 2 - 35, y1 + 42), "time (s)", fill=(20, 20, 20), font=font)
    draw.text((18, layout.plot_top - 18), "depth (um)", fill=(20, 20, 20), font=font)
    draw.text((16, layout.plot_top + 8), "shallow", fill=(90, 90, 90), font=font)
    draw.text((16, y1 - 18), "deep", fill=(90, 90, 90), font=font)


def draw_histogram(
    values: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str,
    x_label: str,
) -> None:
    width = 1800
    height = 1100
    left = 130
    right = 80
    top = 120
    bottom = 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), color=(250, 250, 248))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((left, 28), title, fill=(20, 20, 20), font=font)
    draw.text((left, 56), subtitle, fill=(55, 55, 55), font=font)
    x0 = left
    y0 = top
    x1 = left + plot_w
    y1 = top + plot_h
    draw.rectangle((x0, y0, x1, y1), outline=(55, 55, 55), width=2)

    if values.size == 0:
        draw.text((left, top + 20), "No localized spikes available.", fill=(120, 40, 40), font=font)
        img.save(out_path)
        return

    counts, edges = np.histogram(values, bins=80)
    max_count = int(max(1, counts.max()))
    for i, count in enumerate(counts):
        bar_left = x0 + int(round(i * plot_w / len(counts)))
        bar_right = x0 + int(round((i + 1) * plot_w / len(counts))) - 1
        bar_h = int(round(count / max_count * (plot_h - 10)))
        draw.rectangle(
            (bar_left, y1 - bar_h, max(bar_left + 1, bar_right), y1),
            fill=(120, 135, 160),
            outline=None,
        )

    x_min = float(edges[0])
    x_max = float(edges[-1])
    for frac in np.linspace(0.0, 1.0, 6):
        x = int(round(x0 + frac * plot_w))
        draw.line((x, y1, x, y1 + 8), fill=(55, 55, 55), width=2)
        draw.text((x - 12, y1 + 12), f"{x_min + frac * (x_max - x_min):.0f}", fill=(20, 20, 20), font=font)

    for frac in np.linspace(0.0, 1.0, 5):
        y = int(round(y1 - frac * plot_h))
        draw.line((x0 - 8, y, x0, y), fill=(55, 55, 55), width=2)
        draw.text((18, y - 6), f"{int(round(frac * max_count))}", fill=(20, 20, 20), font=font)

    draw.text((width // 2 - 60, height - 52), x_label, fill=(20, 20, 20), font=font)
    draw.text((18, top - 18), "count", fill=(20, 20, 20), font=font)
    img.save(out_path)


def draw_cluster_spread_plot(summary_df: pd.DataFrame, out_path: Path, title: str) -> None:
    font = ImageFont.load_default()
    if summary_df.empty:
        img = Image.new("RGB", (1600, 500), color=(250, 250, 248))
        draw = ImageDraw.Draw(img)
        draw.text((40, 40), title, fill=(20, 20, 20), font=font)
        draw.text((40, 80), "No localized clusters available.", fill=(120, 40, 40), font=font)
        img.save(out_path)
        return

    row_h = 72
    width = 1900
    height = 180 + row_h * len(summary_df)
    left = 260
    right = 120
    top = 120
    bottom = 80
    plot_w = width - left - right
    img = Image.new("RGB", (width, height), color=(250, 250, 248))
    draw = ImageDraw.Draw(img)
    draw.text((left, 32), title, fill=(20, 20, 20), font=font)
    draw.text(
        (left, 60),
        "gray bar = localized min-max, black dot = median, blue = template peak, green = template COM",
        fill=(55, 55, 55),
        font=font,
    )

    x_min = float(
        min(
            summary_df["localized_y_min_um"].min(),
            summary_df["template_peak_y_um"].min(),
            summary_df["template_com_y_um"].min(),
        )
    )
    x_max = float(
        max(
            summary_df["localized_y_max_um"].max(),
            summary_df["template_peak_y_um"].max(),
            summary_df["template_com_y_um"].max(),
        )
    )
    x_pad = max((x_max - x_min) * 0.05, 20.0)
    x_min -= x_pad
    x_max += x_pad

    for frac in np.linspace(0.0, 1.0, 6):
        x = int(round(left + frac * plot_w))
        draw.line((x, top - 6, x, height - bottom), fill=(225, 225, 225), width=1)
        draw.text((x - 14, height - bottom + 10), f"{x_min + frac * (x_max - x_min):.0f}", fill=(20, 20, 20), font=font)

    for row_idx, row in enumerate(summary_df.itertuples(index=False), start=0):
        y = top + row_idx * row_h + 22
        draw.text(
            (20, y - 8),
            f"id {int(row.cluster_id):>3}  n={int(row.localized_spike_count):>6}  frac={row.localized_fraction:.3f}",
            fill=(20, 20, 20),
            font=font,
        )
        min_x = int(scale_values(np.array([row.localized_y_min_um]), x_min, x_max, left, left + plot_w)[0])
        max_x = int(scale_values(np.array([row.localized_y_max_um]), x_min, x_max, left, left + plot_w)[0])
        med_x = int(scale_values(np.array([row.localized_y_median_um]), x_min, x_max, left, left + plot_w)[0])
        peak_x = int(scale_values(np.array([row.template_peak_y_um]), x_min, x_max, left, left + plot_w)[0])
        com_x = int(scale_values(np.array([row.template_com_y_um]), x_min, x_max, left, left + plot_w)[0])
        draw.line((min_x, y, max_x, y), fill=(120, 120, 120), width=4)
        draw.ellipse((med_x - 4, y - 4, med_x + 4, y + 4), fill=(20, 20, 20))
        draw.line((peak_x, y - 10, peak_x, y + 10), fill=(50, 105, 210), width=3)
        draw.line((com_x, y - 10, com_x, y + 10), fill=(34, 139, 34), width=3)

    img.save(out_path)


def draw_example_cluster_scatter(
    localized_good_spikes: pd.DataFrame,
    selected_clusters: pd.DataFrame,
    out_path: Path,
) -> None:
    font = ImageFont.load_default()
    chosen = selected_clusters.head(4)
    if chosen.empty:
        img = Image.new("RGB", (1600, 1000), color=(250, 250, 248))
        draw = ImageDraw.Draw(img)
        draw.text((40, 40), "No selected clusters available for example scatter panels.", fill=(120, 40, 40), font=font)
        img.save(out_path)
        return

    panel_w = 720
    panel_h = 360
    width = panel_w * 2 + 140
    height = panel_h * 2 + 160
    img = Image.new("RGB", (width, height), color=(250, 250, 248))
    draw = ImageDraw.Draw(img)
    draw.text((60, 28), "Example cluster time-depth panels", fill=(20, 20, 20), font=font)

    for panel_idx, row in enumerate(chosen.itertuples(index=False)):
        panel_x = 60 + (panel_idx % 2) * (panel_w + 40)
        panel_y = 90 + (panel_idx // 2) * (panel_h + 40)
        cid = int(row.cluster_id)
        cluster_df = localized_good_spikes.loc[localized_good_spikes["cluster_id"] == cid]
        if cluster_df.empty:
            continue
        if len(cluster_df) > 5000:
            sample_idx = np.linspace(0, len(cluster_df) - 1, 5000, dtype=np.int64)
            cluster_df = cluster_df.iloc[sample_idx]
        x_vals = cluster_df["spike_time_s"].to_numpy(dtype=np.float64)
        y_vals = cluster_df["y_um"].to_numpy(dtype=np.float64)
        x_px = scale_values(x_vals, float(x_vals.min()), float(x_vals.max()), 0, panel_w - 1)
        y_px = scale_values(y_vals, float(y_vals.min()), float(y_vals.max()), 0, panel_h - 1)
        region = build_background_region(x_px, y_px, panel_h, panel_w)
        overlay_cluster_region(region, x_px, y_px, cluster_color(cid))
        img.paste(Image.fromarray(region, mode="RGB"), (panel_x, panel_y))
        draw.rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), outline=(55, 55, 55), width=2)
        draw.text(
            (panel_x, panel_y - 22),
            f"cluster {cid}  n={len(cluster_df)}  median={row.localized_y_median_um:.1f} um  std={row.localized_y_std_um:.1f}",
            fill=(20, 20, 20),
            font=font,
        )

    img.save(out_path)


def select_highlight_clusters(good_cluster_summary: pd.DataFrame, highlight_count: int) -> pd.DataFrame:
    if good_cluster_summary.empty:
        return good_cluster_summary.copy()

    usable = good_cluster_summary.loc[
        (good_cluster_summary["localized_spike_count"] > 0)
        & np.isfinite(good_cluster_summary["localized_y_median_um"])
    ].copy()
    if usable.empty:
        return usable

    n_target = min(highlight_count, len(usable))
    depth_min = float(usable["localized_y_median_um"].min())
    depth_max = float(usable["localized_y_median_um"].max())
    edges = np.linspace(depth_min, depth_max, n_target + 1)

    picks: list[int] = []
    for idx in range(n_target):
        if idx == n_target - 1:
            mask = (usable["localized_y_median_um"] >= edges[idx]) & (
                usable["localized_y_median_um"] <= edges[idx + 1]
            )
        else:
            mask = (usable["localized_y_median_um"] >= edges[idx]) & (
                usable["localized_y_median_um"] < edges[idx + 1]
            )
        bucket = usable.loc[mask].sort_values(
            ["localized_spike_count", "localized_fraction", "Amplitude", "firing_rate_hz"],
            ascending=[False, False, False, False],
        )
        if not bucket.empty:
            picks.append(int(bucket.iloc[0]["cluster_id"]))

    selected = usable.loc[usable["cluster_id"].isin(picks)].copy()
    if len(selected) < n_target:
        remaining = usable.loc[~usable["cluster_id"].isin(picks)].sort_values(
            ["localized_spike_count", "localized_fraction", "Amplitude", "firing_rate_hz"],
            ascending=[False, False, False, False],
        )
        selected = pd.concat([selected, remaining.head(n_target - len(selected))], ignore_index=True)

    selected = selected.sort_values("localized_y_median_um").reset_index(drop=True)
    selected["color_rgb"] = selected["cluster_id"].map(cluster_color)
    return selected


def load_cluster_tables(ks_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_group = pd.read_csv(ks_dir / "cluster_group.tsv", sep="\t")
    cluster_amp = pd.read_csv(ks_dir / "cluster_Amplitude.tsv", sep="\t")
    cluster_metrics = pd.read_csv(ks_dir / "ibl_format" / "cluster_metrics.csv")
    if "cluster_id.1" in cluster_metrics.columns:
        cluster_metrics = cluster_metrics.drop(columns=["cluster_id.1"])
    cluster_metrics = cluster_metrics.rename(
        columns={"ks2_contamination_pct": "contamination_pct", "firing_rate": "firing_rate_hz"}
    )
    keep_cols = ["cluster_id", "firing_rate_hz", "contamination_pct", "spike_count"]
    cluster_metrics = cluster_metrics[keep_cols].rename(columns={"spike_count": "cluster_metrics_spike_count"})
    return cluster_group, cluster_amp, cluster_metrics


def compute_template_depths(
    templates: np.ndarray,
    channel_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    template_weights = np.ptp(templates, axis=1).astype(np.float64)
    peak_channel_idx = np.argmax(template_weights, axis=1).astype(np.int32)
    template_peak_y_um = channel_positions[peak_channel_idx, 1].astype(np.float64)
    weight_sums = template_weights.sum(axis=1)
    safe_weight_sums = np.where(weight_sums > 0.0, weight_sums, 1.0)
    template_com_y_um = (template_weights @ channel_positions[:, 1].astype(np.float64)) / safe_weight_sums
    return template_weights, peak_channel_idx, template_peak_y_um, template_com_y_um


def compute_primary_template_per_cluster(
    spike_clusters: np.ndarray,
    spike_templates: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[int, int]:
    primary: dict[int, int] = {}
    for cluster_id in cluster_ids.astype(int):
        template_ids = spike_templates[spike_clusters == cluster_id]
        if template_ids.size == 0:
            continue
        uniq, counts = np.unique(template_ids, return_counts=True)
        primary[cluster_id] = int(uniq[np.argmax(counts)])
    return primary


def build_corrupted_regions(
    reader: mtscomp.Reader,
    sample_rate_hz: float,
    spike_times_samples: np.ndarray,
    is_good_cluster: np.ndarray,
    processing_chunk_frames: int,
    context_margin_frames: int,
    corrupted_chunk_start: int | None,
    corrupted_chunk_end: int | None,
) -> list[dict]:
    if corrupted_chunk_start is None or corrupted_chunk_end is None:
        return []

    if corrupted_chunk_end < corrupted_chunk_start:
        raise ValueError("corrupted_chunk_end must be greater than or equal to corrupted_chunk_start")

    sample_start = int(reader.chunk_bounds[corrupted_chunk_start])
    sample_end = int(reader.chunk_bounds[corrupted_chunk_end + 1])
    compressed_start = int(reader.chunk_offsets[corrupted_chunk_start])
    compressed_end = int(reader.chunk_offsets[corrupted_chunk_end + 1])
    skip_start = max(0, sample_start - context_margin_frames)
    skip_end = min(int(reader.n_samples), sample_end + context_margin_frames)
    exact_mask = (spike_times_samples >= sample_start) & (spike_times_samples < sample_end)
    skip_mask = (spike_times_samples >= skip_start) & (spike_times_samples < skip_end)
    chunk_overlap = list(
        range(
            int(sample_start // processing_chunk_frames),
            int((sample_end - 1) // processing_chunk_frames) + 1,
        )
    )
    return [
        {
            "compressed_chunk_start": int(corrupted_chunk_start),
            "compressed_chunk_end": int(corrupted_chunk_end),
            "compressed_chunk_ids": [int(v) for v in range(corrupted_chunk_start, corrupted_chunk_end + 1)],
            "compressed_byte_start": compressed_start,
            "compressed_byte_end": compressed_end,
            "sample_start": sample_start,
            "sample_end": sample_end,
            "time_start_s": sample_start / sample_rate_hz,
            "time_end_s": sample_end / sample_rate_hz,
            "skip_sample_start": skip_start,
            "skip_sample_end": skip_end,
            "skip_time_start_s": skip_start / sample_rate_hz,
            "skip_time_end_s": skip_end / sample_rate_hz,
            "context_margin_frames": int(context_margin_frames),
            "processing_chunk_ids_overlapping_exact_region": chunk_overlap,
            "spike_count_exact_region": int(exact_mask.sum()),
            "good_cluster_spike_count_exact_region": int(np.count_nonzero(exact_mask & is_good_cluster)),
            "spike_count_skipped_region": int(skip_mask.sum()),
            "good_cluster_spike_count_skipped_region": int(np.count_nonzero(skip_mask & is_good_cluster)),
        }
    ]


def build_readable_windows(
    num_samples: int,
    processing_chunk_frames: int,
    skip_intervals: list[tuple[int, int]],
) -> tuple[list[dict], int]:
    windows: list[dict] = []
    num_processing_chunks = int(math.ceil(num_samples / processing_chunk_frames))
    for processing_chunk_id in range(num_processing_chunks):
        base_start = processing_chunk_id * processing_chunk_frames
        base_end = min(num_samples, base_start + processing_chunk_frames)
        segments = [(base_start, base_end)]
        for skip_start, skip_end in skip_intervals:
            updated: list[tuple[int, int]] = []
            for seg_start, seg_end in segments:
                if seg_end <= skip_start or seg_start >= skip_end:
                    updated.append((seg_start, seg_end))
                    continue
                if seg_start < skip_start:
                    updated.append((seg_start, skip_start))
                if seg_end > skip_end:
                    updated.append((skip_end, seg_end))
            segments = updated
        for segment_index, (segment_start, segment_end) in enumerate(segments):
            if segment_end <= segment_start:
                continue
            windows.append(
                {
                    "window_id": len(windows),
                    "processing_chunk_id": int(processing_chunk_id),
                    "segment_index": int(segment_index),
                    "segment_start_sample": int(segment_start),
                    "segment_end_sample": int(segment_end),
                }
            )
    return windows, num_processing_chunks


def constrain_slice_bounds(
    segment_start: int,
    segment_end: int,
    num_samples: int,
    context_margin_frames: int,
    skip_intervals: list[tuple[int, int]],
) -> tuple[int, int]:
    slice_start = max(0, segment_start - context_margin_frames)
    slice_end = min(num_samples, segment_end + context_margin_frames)
    for skip_start, skip_end in skip_intervals:
        if segment_end <= skip_start and slice_end > skip_start:
            slice_end = skip_start
        if segment_start >= skip_end and slice_start < skip_end:
            slice_start = skip_end
    if slice_end <= slice_start:
        raise ValueError("Invalid slice bounds after applying corrupted-interval constraints")
    return int(slice_start), int(slice_end)


def normalize_skip_reason(exc: Exception) -> int:
    message = f"{type(exc).__name__}: {exc}".lower()
    if "compressed chunk" in message or "decompress" in message or "corrupt" in message or "zlib" in message:
        return SKIP_REASON_CORRUPTED_RAW_CHUNK
    if "read" in message or "trace" in message or "ioerror" in message or "oserror" in message:
        return SKIP_REASON_READ_FAILURE
    if "peak" in message and ("invalid" in message or "out of bounds" in message):
        return SKIP_REASON_INVALID_PEAK
    return SKIP_REASON_OTHER


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def build_localization_kwargs(args: argparse.Namespace, single_call_duration_s: float) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "method": args.localization_method,
        "ms_before": args.ms_before,
        "ms_after": args.ms_after,
        "n_jobs": args.n_jobs,
        "chunk_duration": f"{single_call_duration_s:.3f}s",
    }
    if args.localization_method == "center_of_mass":
        kwargs["radius_um"] = args.radius_um
        kwargs["feature"] = args.localization_feature
    elif args.localization_method == "monopolar_triangulation":
        kwargs["radius_um"] = args.radius_um
        kwargs["feature"] = args.localization_feature
        kwargs["max_distance_um"] = args.monopolar_max_distance_um
    return kwargs


def main() -> None:
    args = parse_args()
    session_dir = args.session.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session_name = session_dir.name
    run_mode = "partial" if args.partial_mode else "exact"
    ks_dir = session_dir / "ks"
    meta_path = next(session_dir.glob("*.ap.meta"))
    cbin_path = next(session_dir.glob("*.ap.cbin"))
    ch_path = next(session_dir.glob("*.ap.ch"))
    meta = parse_meta(meta_path)
    sample_rate_hz = float(meta.get("imSampRate", "30000"))

    spike_times_samples = np.asarray(np.load(ks_dir / "spike_times.npy", mmap_mode="r"), dtype=np.int64)
    spike_clusters = np.asarray(np.load(ks_dir / "spike_clusters.npy", mmap_mode="r"), dtype=np.int32)
    spike_templates = np.asarray(np.load(ks_dir / "spike_templates.npy", mmap_mode="r"), dtype=np.int32)
    amplitudes = np.asarray(np.load(ks_dir / "amplitudes.npy", mmap_mode="r"), dtype=np.float32)
    templates = np.load(ks_dir / "templates.npy", mmap_mode="r")
    channel_positions = np.load(ks_dir / "channel_positions.npy")

    cluster_group, cluster_amp, cluster_metrics = load_cluster_tables(ks_dir)
    good_cluster_ids = np.sort(cluster_group.loc[cluster_group["group"] == "good", "cluster_id"].to_numpy(dtype=np.int32))
    is_good_cluster = np.isin(spike_clusters, good_cluster_ids)
    cluster_primary_template = compute_primary_template_per_cluster(spike_clusters, spike_templates, cluster_group["cluster_id"].to_numpy(dtype=np.int32))

    _, template_peak_channel_idx, template_peak_y_um, template_com_y_um = compute_template_depths(
        templates, channel_positions
    )
    spike_peak_channel_idx = template_peak_channel_idx[spike_templates]
    spike_template_peak_y_um = template_peak_y_um[spike_templates].astype(np.float32)
    spike_template_com_y_um = template_com_y_um[spike_templates].astype(np.float32)
    probe_y_min_um = float(np.min(channel_positions[:, 1]))
    probe_y_max_um = float(np.max(channel_positions[:, 1]))

    if not np.all(spike_times_samples[:-1] <= spike_times_samples[1:]):
        raise ValueError("spike_times.npy is not sorted; the partial pipeline assumes Kilosort spike order is time-sorted")

    processing_chunk_duration_s = parse_duration_seconds(args.chunk_duration)
    processing_chunk_frames = max(1, int(round(processing_chunk_duration_s * sample_rate_hz)))
    filter_margin_frames = int(round(args.filter_margin_ms * sample_rate_hz / 1000.0))
    waveform_margin_frames = int(math.ceil(max(args.ms_before, args.ms_after) * sample_rate_hz / 1000.0))
    context_margin_frames = filter_margin_frames + waveform_margin_frames

    base_recording = MTSCompRecording(cbin_path, ch_path, sample_rate_hz, channel_positions)
    num_samples = int(base_recording.get_num_samples())

    corrupted_regions = build_corrupted_regions(
        base_recording._reader,
        sample_rate_hz,
        spike_times_samples,
        is_good_cluster,
        processing_chunk_frames,
        context_margin_frames,
        args.corrupted_chunk_start,
        args.corrupted_chunk_end,
    )
    corrupted_region_summary = {
        "session_name": session_name,
        "session_path": str(session_dir),
        "raw_cbin_path": str(cbin_path),
        "raw_ch_path": str(ch_path),
        "sample_rate_hz": sample_rate_hz,
        "localization_method": args.localization_method,
        "localization_feature": args.localization_feature,
        "processing_chunk_duration_s": processing_chunk_duration_s,
        "processing_chunk_frames": processing_chunk_frames,
        "filter_margin_frames": filter_margin_frames,
        "waveform_margin_frames": waveform_margin_frames,
        "context_margin_frames": context_margin_frames,
        "corrupted_regions": corrupted_regions,
    }

    skip_intervals = [
        (int(region["skip_sample_start"]), int(region["skip_sample_end"])) for region in corrupted_regions
    ]
    readable_windows, num_processing_chunks = build_readable_windows(num_samples, processing_chunk_frames, skip_intervals)

    processing_chunk_id = (spike_times_samples // processing_chunk_frames).astype(np.int32)
    read_window_id = np.full(spike_times_samples.size, -1, dtype=np.int32)
    source_spike_index = np.arange(spike_times_samples.size, dtype=np.int64)
    localization_attempted = np.zeros(spike_times_samples.size, dtype=bool)
    localization_success = np.zeros(spike_times_samples.size, dtype=bool)
    x_um = np.full(spike_times_samples.size, np.nan, dtype=np.float32)
    y_um = np.full(spike_times_samples.size, np.nan, dtype=np.float32)
    skip_reason_code = np.zeros(spike_times_samples.size, dtype=np.int8)

    valid_time_mask = (spike_times_samples >= 0) & (spike_times_samples < num_samples)
    valid_channel_mask = (spike_peak_channel_idx >= 0) & (spike_peak_channel_idx < channel_positions.shape[0])
    valid_amplitude_mask = np.isfinite(amplitudes)
    invalid_peak_mask = ~(valid_time_mask & valid_channel_mask & valid_amplitude_mask)
    if np.any(invalid_peak_mask):
        skip_reason_code[invalid_peak_mask] = SKIP_REASON_INVALID_PEAK

    for skip_start, skip_end in skip_intervals:
        skip_mask = (spike_times_samples >= skip_start) & (spike_times_samples < skip_end)
        skip_reason_code[skip_mask] = SKIP_REASON_CORRUPTED_RAW_CHUNK

    print(
        (
            f"[setup] session={session_name} total_spikes={spike_times_samples.size} "
            f"good_cluster_spikes={int(np.count_nonzero(is_good_cluster))} "
            f"processing_chunks={num_processing_chunks} readable_windows={len(readable_windows)}"
        ),
        flush=True,
    )
    if corrupted_regions:
        region = corrupted_regions[0]
        print(
            (
                f"[corruption] compressed_chunks={region['compressed_chunk_start']}-{region['compressed_chunk_end']} "
                f"exact_time_s={region['time_start_s']:.3f}-{region['time_end_s']:.3f} "
                f"skip_time_s={region['skip_time_start_s']:.3f}-{region['skip_time_end_s']:.3f}"
            ),
            flush=True,
        )

    filtered_recording = spre.bandpass_filter(
        base_recording,
        freq_min=300.0,
        freq_max=6000.0,
        margin_ms=args.filter_margin_ms,
    )
    filtered_recording = spre.common_reference(filtered_recording, reference="global", operator="median")

    peak_dtype = np.dtype(
        [
            ("sample_index", "int64"),
            ("channel_index", "int64"),
            ("amplitude", "float64"),
            ("segment_index", "int64"),
        ]
    )

    window_records: list[dict] = []
    localized_spike_counter = 0
    error_window_counter = 0
    nonfinite_localization_counter = 0
    out_of_range_localization_counter = 0

    for idx, window in enumerate(readable_windows, start=1):
        segment_start = int(window["segment_start_sample"])
        segment_end = int(window["segment_end_sample"])
        lo = int(np.searchsorted(spike_times_samples, segment_start, side="left"))
        hi = int(np.searchsorted(spike_times_samples, segment_end, side="left"))
        spike_count = hi - lo
        good_spike_count = int(np.count_nonzero(is_good_cluster[lo:hi]))
        processable_mask = skip_reason_code[lo:hi] == SKIP_REASON_NONE
        processable_indices = np.arange(lo, hi, dtype=np.int64)[processable_mask]

        slice_start, slice_end = constrain_slice_bounds(
            segment_start,
            segment_end,
            num_samples,
            context_margin_frames,
            skip_intervals,
        )
        window_record = {
            "window_id": int(window["window_id"]),
            "processing_chunk_id": int(window["processing_chunk_id"]),
            "segment_index": int(window["segment_index"]),
            "segment_start_sample": segment_start,
            "segment_end_sample": segment_end,
            "segment_start_s": segment_start / sample_rate_hz,
            "segment_end_s": segment_end / sample_rate_hz,
            "slice_start_sample": slice_start,
            "slice_end_sample": slice_end,
            "slice_start_s": slice_start / sample_rate_hz,
            "slice_end_s": slice_end / sample_rate_hz,
            "spike_count": int(spike_count),
            "processable_spike_count": int(processable_indices.size),
            "preassigned_skip_count": int(spike_count - processable_indices.size),
            "good_cluster_spike_count": int(good_spike_count),
            "status": "no_spikes",
            "localized_success_count": 0,
            "nonfinite_localization_count": 0,
            "out_of_range_localization_count": 0,
            "skip_reason": "",
            "error_message": "",
        }

        if spike_count == 0:
            window_records.append(window_record)
            continue

        if processable_indices.size == 0:
            window_record["status"] = "preassigned_skip_only"
            window_records.append(window_record)
            continue

        peaks = np.zeros(processable_indices.size, dtype=peak_dtype)
        peaks["sample_index"] = spike_times_samples[processable_indices] - slice_start
        peaks["channel_index"] = spike_peak_channel_idx[processable_indices]
        peaks["amplitude"] = amplitudes[processable_indices].astype(np.float64)
        peaks["segment_index"] = 0
        read_window_id[processable_indices] = int(window["window_id"])

        chunk_recording = filtered_recording.frame_slice(slice_start, slice_end)
        single_call_duration_s = max(1.0, (slice_end - slice_start) / sample_rate_hz + 0.01)

        try:
            locations = localize_peaks(chunk_recording, peaks, **build_localization_kwargs(args, single_call_duration_s))
            localization_attempted[processable_indices] = True
            location_x = np.asarray(locations["x"], dtype=np.float32)
            location_y = np.asarray(locations["y"], dtype=np.float32)
            finite_mask = np.isfinite(location_x) & np.isfinite(location_y)
            in_range_mask = finite_mask & (
                (location_y >= (probe_y_min_um - args.coordinate_margin_um))
                & (location_y <= (probe_y_max_um + args.coordinate_margin_um))
            )
            nonfinite_mask = ~finite_mask
            out_of_range_mask = finite_mask & ~in_range_mask
            success_mask = in_range_mask

            successful_indices = processable_indices[success_mask]
            localization_success[successful_indices] = True
            x_um[successful_indices] = location_x[success_mask]
            y_um[successful_indices] = location_y[success_mask]
            localized_spike_counter += int(successful_indices.size)

            if np.any(nonfinite_mask):
                nonfinite_indices = processable_indices[nonfinite_mask]
                skip_reason_code[nonfinite_indices] = SKIP_REASON_OTHER
                nonfinite_localization_counter += int(nonfinite_indices.size)
                window_record["nonfinite_localization_count"] = int(nonfinite_indices.size)

            if np.any(out_of_range_mask):
                out_of_range_indices = processable_indices[out_of_range_mask]
                skip_reason_code[out_of_range_indices] = SKIP_REASON_OTHER
                out_of_range_localization_counter += int(out_of_range_indices.size)
                window_record["out_of_range_localization_count"] = int(out_of_range_indices.size)

            filtered_count = int(np.count_nonzero(nonfinite_mask) + np.count_nonzero(out_of_range_mask))
            window_record["localized_success_count"] = int(successful_indices.size)
            if filtered_count == 0:
                window_record["status"] = "localized"
            elif successful_indices.size > 0:
                window_record["status"] = "localized_with_filtered_spikes"
                window_record["skip_reason"] = SKIP_REASON_CATEGORIES[SKIP_REASON_OTHER]
            else:
                window_record["status"] = "localized_all_filtered"
                window_record["skip_reason"] = SKIP_REASON_CATEGORIES[SKIP_REASON_OTHER]
        except Exception as exc:
            code = normalize_skip_reason(exc)
            localization_attempted[processable_indices] = True
            localization_success[processable_indices] = False
            skip_reason_code[processable_indices] = code
            window_record["status"] = "failed"
            window_record["skip_reason"] = SKIP_REASON_CATEGORIES[code]
            window_record["error_message"] = f"{type(exc).__name__}: {exc}"
            error_window_counter += 1
            print(
                (
                    f"[window-failed] id={window_record['window_id']} processing_chunk_id={window_record['processing_chunk_id']} "
                    f"spikes={spike_count} reason={window_record['skip_reason']}"
                ),
                flush=True,
            )
            if not args.partial_mode:
                raise

        window_records.append(window_record)

        if idx == 1 or idx % 10 == 0 or idx == len(readable_windows):
            print(
                (
                    f"[progress] windows={idx}/{len(readable_windows)} "
                    f"localized_spikes={localized_spike_counter} "
                    f"failed_windows={error_window_counter}"
                ),
                flush=True,
            )

    covered_mask = localization_success | (skip_reason_code != SKIP_REASON_NONE)
    unresolved_pipeline_gap_count = int(np.size(covered_mask) - np.count_nonzero(covered_mask))
    if unresolved_pipeline_gap_count > 0:
        missing_mask = ~covered_mask
        skip_reason_code[missing_mask] = SKIP_REASON_UNRESOLVED_PIPELINE_GAP
        print(
            f"[status-gap] assigned unresolved_pipeline_gap to {unresolved_pipeline_gap_count} spikes",
            flush=True,
        )

    covered_mask = localization_success | (skip_reason_code != SKIP_REASON_NONE)
    if not np.all(covered_mask):
        missing_count = int(np.size(covered_mask) - np.count_nonzero(covered_mask))
        raise RuntimeError(f"{missing_count} spikes were left without localization status")

    skip_reason = pd.Categorical.from_codes(skip_reason_code, categories=SKIP_REASON_CATEGORIES)
    session_spike_table = pd.DataFrame(
        {
            "source_spike_index": source_spike_index,
            "spike_time_samples": spike_times_samples,
            "spike_time_s": spike_times_samples.astype(np.float64) / sample_rate_hz,
            "cluster_id": spike_clusters,
            "is_good_cluster": is_good_cluster,
            "peak_channel_index": spike_peak_channel_idx.astype(np.int32),
            "template_peak_y_um": spike_template_peak_y_um,
            "template_com_y_um": spike_template_com_y_um,
            "processing_chunk_id": processing_chunk_id,
            "read_window_id": read_window_id,
            "localization_attempted": localization_attempted,
            "localization_success": localization_success,
            "localization_missing": ~localization_success,
            "skip_reason": skip_reason,
            "x_um": x_um,
            "y_um": y_um,
            "amplitude": amplitudes,
        }
    )

    localized_good_spikes = session_spike_table.loc[
        session_spike_table["is_good_cluster"] & session_spike_table["localization_success"]
    ].copy()
    all_good_spikes = session_spike_table.loc[session_spike_table["is_good_cluster"]].copy()
    localized_all_spikes = session_spike_table.loc[session_spike_table["localization_success"]].copy()

    good_cluster_base = pd.DataFrame({"cluster_id": good_cluster_ids})
    good_total = all_good_spikes.groupby("cluster_id").size().rename("total_spike_count").reset_index()
    good_localized = (
        localized_good_spikes.groupby("cluster_id", sort=True)
        .agg(
            localized_spike_count=("cluster_id", "size"),
            localized_x_median_um=("x_um", "median"),
            localized_y_median_um=("y_um", "median"),
            localized_y_mean_um=("y_um", "mean"),
            localized_y_std_um=("y_um", "std"),
            localized_y_min_um=("y_um", "min"),
            localized_y_max_um=("y_um", "max"),
            amplitude_median=("amplitude", "median"),
        )
        .reset_index()
    )
    good_skipped = (
        all_good_spikes.loc[~all_good_spikes["localization_success"]]
        .groupby("cluster_id")
        .size()
        .rename("skipped_spike_count")
        .reset_index()
    )

    good_cluster_summary = good_cluster_base.merge(good_total, on="cluster_id", how="left")
    good_cluster_summary = good_cluster_summary.merge(good_localized, on="cluster_id", how="left")
    good_cluster_summary = good_cluster_summary.merge(good_skipped, on="cluster_id", how="left")
    good_cluster_summary["total_spike_count"] = good_cluster_summary["total_spike_count"].fillna(0).astype(np.int64)
    good_cluster_summary["localized_spike_count"] = good_cluster_summary["localized_spike_count"].fillna(0).astype(np.int64)
    good_cluster_summary["skipped_spike_count"] = good_cluster_summary["skipped_spike_count"].fillna(0).astype(np.int64)
    good_cluster_summary["localized_fraction"] = np.where(
        good_cluster_summary["total_spike_count"] > 0,
        good_cluster_summary["localized_spike_count"] / good_cluster_summary["total_spike_count"],
        0.0,
    )
    for col in ["localized_y_std_um", "localized_y_mean_um", "localized_y_min_um", "localized_y_max_um", "localized_x_median_um", "amplitude_median", "localized_y_median_um"]:
        if col in good_cluster_summary.columns:
            good_cluster_summary[col] = good_cluster_summary[col].astype(np.float64)

    good_cluster_summary["primary_template_id"] = good_cluster_summary["cluster_id"].map(cluster_primary_template)
    good_cluster_summary["template_peak_y_um"] = good_cluster_summary["primary_template_id"].map(
        lambda tid: float(template_peak_y_um[int(tid)]) if pd.notna(tid) else np.nan
    )
    good_cluster_summary["template_com_y_um"] = good_cluster_summary["primary_template_id"].map(
        lambda tid: float(template_com_y_um[int(tid)]) if pd.notna(tid) else np.nan
    )
    good_cluster_summary["peak_channel_index"] = good_cluster_summary["primary_template_id"].map(
        lambda tid: int(template_peak_channel_idx[int(tid)]) if pd.notna(tid) else -1
    )
    good_cluster_summary["delta_median_vs_template_peak_um"] = (
        good_cluster_summary["localized_y_median_um"] - good_cluster_summary["template_peak_y_um"]
    )
    good_cluster_summary["delta_median_vs_template_com_um"] = (
        good_cluster_summary["localized_y_median_um"] - good_cluster_summary["template_com_y_um"]
    )
    good_cluster_summary = good_cluster_summary.merge(cluster_amp, on="cluster_id", how="left")
    good_cluster_summary = good_cluster_summary.merge(cluster_metrics, on="cluster_id", how="left")

    selected_clusters = select_highlight_clusters(good_cluster_summary, args.highlight_count)
    selected_cluster_ids = set(int(cid) for cid in selected_clusters["cluster_id"])
    localized_good_spikes["is_highlighted"] = localized_good_spikes["cluster_id"].isin(selected_cluster_ids)

    raster_path = output_dir / f"{session_name}_exact_depth_raster.png"
    histogram_path = output_dir / f"{session_name}_localized_depth_histogram.png"
    spread_path = output_dir / f"{session_name}_cluster_depth_spread.png"
    example_path = output_dir / f"{session_name}_example_cluster_scatter.png"
    spike_table_path = output_dir / f"{session_name}_localized_spike_table.csv.gz"
    cluster_summary_path = output_dir / f"{session_name}_good_cluster_summary.csv"
    selected_clusters_path = output_dir / f"{session_name}_selected_good_clusters.csv"
    processing_windows_path = output_dir / f"{session_name}_processing_windows.csv"
    corrupted_region_summary_path = output_dir / f"{session_name}_corrupted_region_summary.json"
    validation_path = output_dir / f"{session_name}_validation_report.json"
    notes_path = output_dir / f"{session_name}_notes.md"

    processing_windows_df = pd.DataFrame(window_records)
    spike_table_path.parent.mkdir(parents=True, exist_ok=True)
    finite_plot_mask_all = np.isfinite(localized_all_spikes["spike_time_s"]) & np.isfinite(localized_all_spikes["y_um"])
    finite_plot_mask_good = np.isfinite(localized_good_spikes["spike_time_s"]) & np.isfinite(localized_good_spikes["y_um"])
    localized_good_spikes_plot = localized_good_spikes.loc[finite_plot_mask_good].copy()
    plot_excluded_localized_spikes = int(len(localized_all_spikes) - np.count_nonzero(finite_plot_mask_all))
    plot_excluded_good_localized_spikes = int(len(localized_good_spikes) - len(localized_good_spikes_plot))

    localized_overall_count = int(np.count_nonzero(localization_success))
    localized_good_count = int(len(localized_good_spikes))
    skipped_corrupted_count = int(np.count_nonzero(skip_reason_code == SKIP_REASON_CORRUPTED_RAW_CHUNK))
    skipped_read_failure_count = int(np.count_nonzero(skip_reason_code == SKIP_REASON_READ_FAILURE))
    skipped_invalid_peak_count = int(np.count_nonzero(skip_reason_code == SKIP_REASON_INVALID_PEAK))
    skipped_other_count = int(np.count_nonzero(skip_reason_code == SKIP_REASON_OTHER))
    skipped_unresolved_count = int(np.count_nonzero(skip_reason_code == SKIP_REASON_UNRESOLVED_PIPELINE_GAP))
    total_good_spikes = int(np.count_nonzero(is_good_cluster))
    validation_report = {
        "session_name": session_name,
        "run_mode": run_mode,
        "localization_method": args.localization_method,
        "localization_feature": args.localization_feature,
        "total_spikes": int(spike_times_samples.size),
        "total_good_cluster_spikes": total_good_spikes,
        "localized_spikes": localized_overall_count,
        "localized_good_cluster_spikes": localized_good_count,
        "skipped_corrupted_spikes": skipped_corrupted_count,
        "skipped_read_failure_spikes": skipped_read_failure_count,
        "skipped_invalid_peak_spikes": skipped_invalid_peak_count,
        "skipped_other_spikes": skipped_other_count,
        "unresolved_pipeline_gap_spikes": skipped_unresolved_count,
        "localized_fraction_overall": localized_overall_count / max(1, int(spike_times_samples.size)),
        "localized_fraction_good_clusters": localized_good_count / max(1, total_good_spikes),
        "processing_chunk_count": int(num_processing_chunks),
        "readable_window_count": int(len(readable_windows)),
        "failed_window_count": int(error_window_counter),
        "nonfinite_localization_spikes": int(nonfinite_localization_counter),
        "out_of_range_localization_spikes": int(out_of_range_localization_counter),
        "plotting_status": "pending",
        "plotting_excluded_nonfinite_localized_spikes": plot_excluded_localized_spikes,
        "plotting_excluded_nonfinite_good_cluster_spikes": plot_excluded_good_localized_spikes,
        "join_integrity": {
            "row_count_matches_input": bool(len(session_spike_table) == spike_times_samples.size),
            "source_spike_index_unique": bool(session_spike_table["source_spike_index"].is_unique),
            "localized_plus_skipped_equals_total": bool(
                localized_overall_count + int(np.count_nonzero(skip_reason_code != SKIP_REASON_NONE))
                == spike_times_samples.size
            ),
        },
        "corrupted_region_summary_path": str(corrupted_region_summary_path) if corrupted_regions else "",
        "selected_cluster_depth_examples": selected_clusters[
            ["cluster_id", "localized_spike_count", "localized_y_median_um", "localized_y_std_um", "localized_y_min_um", "localized_y_max_um"]
        ].to_dict(orient="records"),
        "outputs": {
            "spike_table_csv_gz": str(spike_table_path),
            "good_cluster_summary_csv": str(cluster_summary_path),
            "selected_clusters_csv": str(selected_clusters_path),
            "processing_windows_csv": str(processing_windows_path),
            "corrupted_region_summary_json": str(corrupted_region_summary_path) if corrupted_regions else "",
            "raster_png": str(raster_path),
            "histogram_png": str(histogram_path),
            "cluster_spread_png": str(spread_path),
            "example_cluster_scatter_png": str(example_path),
        },
    }

    notes = [
        f"# {'Partial' if args.partial_mode else 'Exact'} raw-localization session: {session_name}",
        "",
        "## Objective",
        "",
        (
            "- Localize spikes from the raw `.ap.cbin` while skipping only the known corrupted raw interval."
            if args.partial_mode
            else "- Localize spikes from the raw `.ap.cbin` across the full readable session."
        ),
        "- Preserve one row per Kilosort spike so the same output schema can be reused on clean sessions later.",
        "",
        "## Key behavior",
        "",
        f"- Localization method: `{args.localization_method}` using feature `{args.localization_feature}`.",
        "- Uses the existing Kilosort outputs: spike times, spike clusters, spike templates, and amplitudes.",
        "- Reads the raw `.ap.cbin` through an `mtscomp`-backed SpikeInterface recording.",
        "- Applies bandpass filtering and global median reference before localization.",
        (
            "- Processes the session chunk by chunk and skips only the expanded corrupted raw interval."
            if args.partial_mode
            else "- Processes the session chunk by chunk without session-specific exceptions."
        ),
        "- Records localization success or failure for every spike.",
        "",
        "## Output contract",
        "",
        "- `source_spike_index`, `spike_time_samples`, `spike_time_s`, `cluster_id`, `is_good_cluster`",
        "- `localization_attempted`, `localization_success`, `localization_missing`, `skip_reason`",
        "- `x_um`, `y_um`, `amplitude`, `processing_chunk_id`, `read_window_id`",
        "- `peak_channel_index`, `template_peak_y_um`, `template_com_y_um`",
        "",
        "## Notes",
        "",
        "- `processing_chunk_id` is the base processing chunk derived from the session chunk duration.",
        f"- Non-finite localized spikes reassigned to `skip_reason=other`: {nonfinite_localization_counter}.",
        f"- Out-of-range localized spikes reassigned to `skip_reason=other`: {out_of_range_localization_counter}.",
        f"- Plotting excluded {plot_excluded_good_localized_spikes} good-cluster spikes with non-finite depth/time values.",
        f"- `unresolved_pipeline_gap` count: {skipped_unresolved_count}.",
        (
            "- `skip_reason=corrupted_raw_chunk` marks spikes in the known unreadable raw interval."
            if args.partial_mode
            else "- `skip_reason` should stay empty on clean sessions."
        ),
        (
            "- The raster is explicitly labeled as partial because the corrupted raw interval is absent."
            if args.partial_mode
            else "- The raster is labeled exact because no known raw corruption was configured."
        ),
    ]

    session_spike_table.to_csv(spike_table_path, index=False, compression="gzip")
    good_cluster_summary.to_csv(cluster_summary_path, index=False)
    selected_clusters.to_csv(selected_clusters_path, index=False)
    processing_windows_df.to_csv(processing_windows_path, index=False)
    if corrupted_regions:
        write_json(corrupted_region_summary_path, corrupted_region_summary)
    write_json(validation_path, validation_report)
    notes_path.write_text("\n".join(notes))

    layout = PlotLayout()
    if not localized_good_spikes_plot.empty:
        x_vals = localized_good_spikes_plot["spike_time_s"].to_numpy(dtype=np.float64)
        y_vals = localized_good_spikes_plot["y_um"].to_numpy(dtype=np.float64)
        duration_s = float(spike_times_samples.max() / sample_rate_hz)
        depth_min = float(y_vals.min())
        depth_max = float(y_vals.max())
        time_pad = max(duration_s * 0.005, 2.0)
        depth_pad = max((depth_max - depth_min) * 0.03, 20.0)
        x_px = scale_values(x_vals, 0.0 - time_pad, duration_s + time_pad, 0, layout.plot_w - 1)
        y_px = scale_values(y_vals, depth_min - depth_pad, depth_max + depth_pad, 0, layout.plot_h - 1)
        region = build_background_region(x_px, y_px, layout.plot_h, layout.plot_w)
        cluster_ids_all = localized_good_spikes_plot["cluster_id"].to_numpy(dtype=np.int32)
        for _, row in selected_clusters.iterrows():
            cid = int(row["cluster_id"])
            mask = cluster_ids_all == cid
            overlay_cluster_region(region, x_px[mask], y_px[mask], tuple(row["color_rgb"]))
        img = Image.new("RGB", (layout.canvas_w, layout.canvas_h), color=(250, 250, 248))
        img.paste(Image.fromarray(region, mode="RGB"), (layout.plot_left, layout.plot_top))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw_axes(
            draw,
            layout,
            duration_s=duration_s,
            depth_min=depth_min - depth_pad,
            depth_max=depth_max + depth_pad,
            font=font,
            title=f"{session_name.replace('_', ' ')} {'partial ' if args.partial_mode else ''}exact-depth raster",
            subtitle=(
                f"raw-localized spikes via {args.localization_method}; corrupted raw interval skipped; gray = all good spikes, color = selected good clusters"
                if args.partial_mode
                else f"raw-localized spikes via {args.localization_method}; gray = all good spikes, color = selected good clusters"
            ),
        )
        legend_y = layout.legend_top
        draw.text((layout.legend_left, legend_y - 30), "Selected good clusters", fill=(20, 20, 20), font=font)
        for _, row in selected_clusters.iterrows():
            color = tuple(row["color_rgb"])
            cid = int(row["cluster_id"])
            label = (
                f"id {cid:>3}  y_med {row['localized_y_median_um']:.1f} um  "
                f"localized {int(row['localized_spike_count'])}"
            )
            draw.rectangle(
                (layout.legend_left, legend_y, layout.legend_left + 18, legend_y + 18),
                fill=color,
                outline=(50, 50, 50),
            )
            draw.text((layout.legend_left + 28, legend_y + 2), label, fill=(20, 20, 20), font=font)
            legend_y += 28

        note_y = legend_y + 24
        coverage_note = (
            f"localized good spikes: {int(len(localized_good_spikes_plot))} / {int(len(all_good_spikes))} "
            f"({len(localized_good_spikes_plot) / max(1, len(all_good_spikes)):.3f})"
        )
        raster_notes = [
            f"session: {session_name}",
            f"method: {args.localization_method}",
            f"sample rate: {sample_rate_hz:.0f} Hz",
            coverage_note,
            f"excluded non-finite good spikes: {plot_excluded_good_localized_spikes}",
        ]
        if corrupted_regions:
            region_info = corrupted_regions[0]
            raster_notes.append(
                "skipped raw interval: "
                f"{region_info['skip_time_start_s']:.3f}-{region_info['skip_time_end_s']:.3f} s"
            )
        for note in raster_notes:
            draw.text((layout.legend_left, note_y), note, fill=(60, 60, 60), font=font)
            note_y += 20
        img.save(raster_path)
    else:
        img = Image.new("RGB", (layout.canvas_w, layout.canvas_h), color=(250, 250, 248))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.text((40, 40), f"{session_name} exact-depth raster unavailable", fill=(120, 40, 40), font=font)
        draw.text((40, 68), "No finite localized good-cluster spikes remained for plotting.", fill=(90, 90, 90), font=font)
        img.save(raster_path)

    draw_histogram(
        localized_good_spikes_plot["y_um"].to_numpy(dtype=np.float64),
        histogram_path,
        title=f"{session_name.replace('_', ' ')} localized depth histogram",
        subtitle=(
            "good-cluster spikes with successful raw localization; partial run"
            if args.partial_mode
            else "good-cluster spikes with successful raw localization"
        ),
        x_label="localized y (um)",
    )
    draw_cluster_spread_plot(
        selected_clusters
        if not selected_clusters.empty
        else good_cluster_summary.loc[
            (good_cluster_summary["localized_spike_count"] > 0)
            & np.isfinite(good_cluster_summary["localized_y_median_um"])
        ].head(10),
        spread_path,
        title=f"{session_name.replace('_', ' ')} cluster depth spread",
    )
    draw_example_cluster_scatter(localized_good_spikes_plot, selected_clusters, example_path)

    validation_report["plotting_status"] = "complete"
    write_json(validation_path, validation_report)

    print(
        (
            f"[write] spike_table={spike_table_path.name} localized_overall={localized_overall_count} "
            f"localized_good={localized_good_count} skipped_corrupted={skipped_corrupted_count}"
        ),
        flush=True,
    )
    print(f"Wrote spike table to {spike_table_path}")
    print(f"Wrote good-cluster summary to {cluster_summary_path}")
    print(f"Wrote validation report to {validation_path}")


if __name__ == "__main__":
    main()
