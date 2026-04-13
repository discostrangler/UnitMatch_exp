#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from _pipeline_utils import dump_json, now_iso

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPIKE_ROSTER_ROOT = PROJECT_ROOT / "spike_rosters_raw"
sys.path.insert(0, str(SPIKE_ROSTER_ROOT))

from build_localized_single_session_raster import PlotLayout, scale_values  # noqa: E402


@dataclass
class SessionConfig:
    session_name: str
    session_date: str
    aligned_csv_gz: Path
    duration_s: float
    block_start_s: float = 0.0


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


def counts_to_region(counts: np.ndarray) -> np.ndarray:
    if counts.max() == 0:
        return np.full((counts.shape[0], counts.shape[1], 3), 255, dtype=np.uint8)
    log_counts = np.log1p(counts.astype(np.float32))
    norm = log_counts / float(log_counts.max())
    gray = (255.0 - norm * 160.0).clip(80.0, 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def overlay_counts(region: np.ndarray, counts: np.ndarray, color_rgb: tuple[int, int, int]) -> None:
    mask = counts > 0
    if not np.any(mask):
        return
    norm = np.log1p(counts[mask].astype(np.float32))
    norm /= float(norm.max())
    alpha = (0.30 + 0.70 * norm).reshape(-1, 1)
    base = region[mask].astype(np.float32)
    color = np.asarray(color_rgb, dtype=np.float32).reshape(1, 3)
    region[mask] = np.clip(base * (1.0 - alpha) + color * alpha, 0.0, 255.0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/configs/unitmatch_dredge_run_config.json",
    )
    parser.add_argument(
        "--selected-tracked-units-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/outputs/tracked_tables/selected_tracked_units.csv",
    )
    parser.add_argument(
        "--overlay-output-csv-gz",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/outputs/figures/al032_12session_dredge_session_aligned_overlay_spikes.csv.gz",
    )
    parser.add_argument(
        "--output-png",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/outputs/figures/al032_12session_dredge_session_aligned_raster.png",
    )
    parser.add_argument(
        "--output-summary-json",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/outputs/figures/al032_12session_dredge_session_aligned_raster_summary.json",
    )
    parser.add_argument("--session-gap-s", type=float, default=60.0)
    return parser.parse_args()


def load_session_configs(config: dict, session_gap_s: float) -> list[SessionConfig]:
    manifest = pd.read_csv(Path(config["dredge_manifest_csv"])).sort_values("session_order")
    durations = {row.session_name: float(row.duration_s) for row in manifest.itertuples(index=False)}
    dates = {row.session_name: str(row.session_date) for row in manifest.itertuples(index=False)}
    aligned_root = Path(config["corrected_spikes_root"])
    block_start = 0.0
    out = []
    for session_name in config["session_names"]:
        out.append(
            SessionConfig(
                session_name=session_name,
                session_date=dates[session_name],
                aligned_csv_gz=aligned_root / session_name / f"{session_name}_tracked_spikes_session_aligned.csv.gz",
                duration_s=durations[session_name],
                block_start_s=block_start,
            )
        )
        block_start += durations[session_name] + float(session_gap_s)
    return out


def determine_depth_range(session_configs: list[SessionConfig]) -> tuple[float, float, dict[str, int]]:
    depth_min = math.inf
    depth_max = -math.inf
    background_counts_by_session: dict[str, int] = {}
    for config in session_configs:
        count = 0
        usecols = ["is_good_cluster", "localization_success", "y_um_dredge_session_aligned"]
        for chunk in pd.read_csv(config.aligned_csv_gz, compression="gzip", usecols=usecols, chunksize=250_000, low_memory=False):
            mask = (
                chunk["is_good_cluster"].astype(bool).to_numpy()
                & chunk["localization_success"].astype(bool).to_numpy()
                & np.isfinite(chunk["y_um_dredge_session_aligned"].to_numpy(dtype=np.float64))
            )
            if not np.any(mask):
                continue
            values = chunk.loc[mask, "y_um_dredge_session_aligned"].to_numpy(dtype=np.float64)
            count += int(values.size)
            depth_min = min(depth_min, float(values.min()))
            depth_max = max(depth_max, float(values.max()))
        background_counts_by_session[config.session_name] = count
    return depth_min, depth_max, background_counts_by_session


def build_counts_and_overlay_table(
    session_configs: list[SessionConfig],
    selected_units: pd.DataFrame,
    plot_w: int,
    plot_h: int,
    time_min: float,
    time_max: float,
    depth_min: float,
    depth_max: float,
    overlay_output_csv_gz: Path,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, int]]:
    background_counts = np.zeros((plot_h, plot_w), dtype=np.uint32)
    overlay_counts_by_unit = {
        int(row.tracked_unit_id): np.zeros((plot_h, plot_w), dtype=np.uint32)
        for row in selected_units.itertuples(index=False)
    }
    overlay_counts_by_session: dict[str, int] = {}
    selected_ids = set(overlay_counts_by_unit)

    overlay_output_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    header = True
    with gzip.open(overlay_output_csv_gz, "wt", newline="") as handle:
        for block_index, config in enumerate(session_configs):
            overlay_rows_written = 0
            usecols = [
                "source_spike_index",
                "spike_time_samples",
                "spike_time_s",
                "cluster_id",
                "tracked_unit_id",
                "conflict_flag",
                "is_good_cluster",
                "localization_success",
                "y_um_dredge_session_aligned",
            ]
            for chunk in pd.read_csv(config.aligned_csv_gz, compression="gzip", usecols=usecols, chunksize=250_000, low_memory=False):
                finite_y = np.isfinite(chunk["y_um_dredge_session_aligned"].to_numpy(dtype=np.float64))
                good_background_mask = (
                    chunk["is_good_cluster"].astype(bool).to_numpy()
                    & chunk["localization_success"].astype(bool).to_numpy()
                    & finite_y
                )
                if np.any(good_background_mask):
                    plot_time = chunk.loc[good_background_mask, "spike_time_s"].to_numpy(dtype=np.float64) + float(config.block_start_s)
                    x_px = scale_values(plot_time, time_min, time_max, 0, plot_w - 1)
                    y_px = scale_values(
                        chunk.loc[good_background_mask, "y_um_dredge_session_aligned"].to_numpy(dtype=np.float64),
                        depth_min,
                        depth_max,
                        0,
                        plot_h - 1,
                    )
                    np.add.at(background_counts, (y_px, x_px), 1)

                tracked_mask = chunk["tracked_unit_id"].notna().to_numpy()
                if "conflict_flag" in chunk.columns:
                    tracked_mask &= ~series_to_bool_mask(chunk["conflict_flag"])
                tracked_mask &= good_background_mask
                if not np.any(tracked_mask):
                    continue
                tracked = chunk.loc[tracked_mask].copy()
                tracked["tracked_unit_id"] = tracked["tracked_unit_id"].astype(int)
                tracked = tracked.loc[tracked["tracked_unit_id"].isin(selected_ids)].copy()
                if tracked.empty:
                    continue

                tracked["session_name"] = config.session_name
                tracked["session_block_index"] = int(block_index)
                tracked["session_block_start_s"] = float(config.block_start_s)
                tracked["plot_time_s"] = tracked["spike_time_s"] + float(config.block_start_s)
                tracked[
                    [
                        "session_name",
                        "session_block_index",
                        "session_block_start_s",
                        "source_spike_index",
                        "spike_time_samples",
                        "spike_time_s",
                        "plot_time_s",
                        "cluster_id",
                        "tracked_unit_id",
                        "y_um_dredge_session_aligned",
                    ]
                ].to_csv(handle, index=False, header=header)
                header = False
                overlay_rows_written += int(tracked.shape[0])

                for tracked_unit_id, group in tracked.groupby("tracked_unit_id", sort=False):
                    x_px = scale_values(group["plot_time_s"].to_numpy(dtype=np.float64), time_min, time_max, 0, plot_w - 1)
                    y_px = scale_values(
                        group["y_um_dredge_session_aligned"].to_numpy(dtype=np.float64),
                        depth_min,
                        depth_max,
                        0,
                        plot_h - 1,
                    )
                    np.add.at(overlay_counts_by_unit[int(tracked_unit_id)], (y_px, x_px), 1)
            overlay_counts_by_session[config.session_name] = overlay_rows_written
    return background_counts, overlay_counts_by_unit, overlay_counts_by_session


def draw_raster(
    session_configs: list[SessionConfig],
    background_counts: np.ndarray,
    overlay_counts_by_unit: dict[int, np.ndarray],
    selected_units: pd.DataFrame,
    depth_min: float,
    depth_max: float,
    time_min: float,
    time_max: float,
    session_gap_s: float,
    output_png: Path,
) -> None:
    layout = PlotLayout(canvas_w=3200, canvas_h=1650, plot_left=130, plot_top=110, plot_w=2350, plot_h=1280, legend_left=2530, legend_top=150)
    img = Image.new("RGB", (layout.canvas_w, layout.canvas_h), color=(250, 250, 248))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    region = counts_to_region(background_counts)
    for row in selected_units.itertuples(index=False):
        overlay_counts(region, overlay_counts_by_unit[int(row.tracked_unit_id)], (int(row.color_r), int(row.color_g), int(row.color_b)))
    img.paste(Image.fromarray(region, mode="RGB"), (layout.plot_left, layout.plot_top))

    x0 = layout.plot_left
    y0 = layout.plot_top
    x1 = layout.plot_left + layout.plot_w
    y1 = layout.plot_top + layout.plot_h
    draw.rectangle((x0, y0, x1, y1), outline=(55, 55, 55), width=2)
    draw.text((x0, 38), "AL032 12-session DREDge session-aligned raster", fill=(20, 20, 20), font=font)
    subtitle = "gray = all good corrected spikes, color = selected tracked units, y = y_um_dredge_session_aligned"
    draw.text((x0, 66), subtitle, fill=(55, 55, 55), font=font)

    for frac in np.linspace(0.0, 1.0, 6):
        y = int(round(y0 + frac * layout.plot_h))
        draw.line((x0, y, x1, y), fill=(230, 230, 230), width=1)
        depth_value = depth_max - frac * (depth_max - depth_min)
        draw.text((35, y - 6), f"{depth_value:.0f}", fill=(70, 70, 70), font=font)
    draw.text((14, y0 + layout.plot_h // 2), "depth (um)", fill=(45, 45, 45), font=font)

    for config in session_configs:
        frac = (config.block_start_s - time_min) / max(time_max - time_min, 1e-9)
        x = int(round(x0 + frac * layout.plot_w))
        if config.block_start_s > 0:
            draw.line((x, y0, x, y1), fill=(180, 180, 180), width=1)
        center_frac = ((config.block_start_s + config.duration_s / 2.0) - time_min) / max(time_max - time_min, 1e-9)
        center_x = int(round(x0 + center_frac * layout.plot_w))
        draw.text((center_x - 22, y1 + 12), config.session_date[5:], fill=(70, 70, 70), font=font)
    draw.text((x0 + layout.plot_w // 2 - 40, y1 + 32), f"session blocks ({session_gap_s:.0f} s gaps)", fill=(45, 45, 45), font=font)

    draw.text((layout.legend_left, layout.legend_top - 20), "tracked units", fill=(20, 20, 20), font=font)
    for idx, row in enumerate(selected_units.itertuples(index=False)):
        yy = layout.legend_top + idx * 26
        draw.rectangle((layout.legend_left, yy, layout.legend_left + 14, yy + 14), fill=(int(row.color_r), int(row.color_g), int(row.color_b)))
        draw.text((layout.legend_left + 22, yy - 1), row.tracked_label, fill=(35, 35, 35), font=font)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_png)


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    selected_units = pd.read_csv(args.selected_tracked_units_csv).copy()
    session_configs = load_session_configs(config, args.session_gap_s)
    time_min = 0.0
    time_max = session_configs[-1].block_start_s + session_configs[-1].duration_s
    depth_min, depth_max, background_counts_by_session = determine_depth_range(session_configs)
    layout = PlotLayout(canvas_w=3200, canvas_h=1650, plot_left=130, plot_top=110, plot_w=2350, plot_h=1280, legend_left=2530, legend_top=150)

    background_counts, overlay_counts_by_unit, overlay_counts_by_session = build_counts_and_overlay_table(
        session_configs=session_configs,
        selected_units=selected_units,
        plot_w=layout.plot_w,
        plot_h=layout.plot_h,
        time_min=time_min,
        time_max=time_max,
        depth_min=depth_min,
        depth_max=depth_max,
        overlay_output_csv_gz=Path(args.overlay_output_csv_gz),
    )

    draw_raster(
        session_configs=session_configs,
        background_counts=background_counts,
        overlay_counts_by_unit=overlay_counts_by_unit,
        selected_units=selected_units,
        depth_min=depth_min,
        depth_max=depth_max,
        time_min=time_min,
        time_max=time_max,
        session_gap_s=args.session_gap_s,
        output_png=Path(args.output_png),
    )

    dump_json(
        Path(args.output_summary_json),
        {
            "created_at": now_iso(),
            "output_png": str(args.output_png),
            "overlay_output_csv_gz": str(args.overlay_output_csv_gz),
            "session_gap_s": float(args.session_gap_s),
            "selected_tracked_unit_ids": [int(v) for v in selected_units["tracked_unit_id"].tolist()],
            "background_spike_counts_by_session": background_counts_by_session,
            "overlay_spike_counts_by_session": overlay_counts_by_session,
            "depth_min_um": float(depth_min),
            "depth_max_um": float(depth_max),
        },
    )


if __name__ == "__main__":
    main()
