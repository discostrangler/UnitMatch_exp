#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks, compute_peak_displacements, estimate_motion

from _pipeline_utils import csv_write, dump_json, now_iso


ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge")
UPSTREAM_ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch")
MANIFEST_PATH = ROOT / "manifests" / "al032_dredge_input_manifest.csv"
DREDGE_ROOT = ROOT / "outputs" / "dredge_per_session"
ATTACHED_SPIKES_ROOT = UPSTREAM_ROOT / "outputs" / "attached_spikes"
SELECTED_UNITS_PATH = UPSTREAM_ROOT / "outputs" / "tracked_tables" / "selected_tracked_units.csv"
DREDGE_CONFIG_PATH = DREDGE_ROOT / "dredge_run_config_phase1to4.json"
LOCALIZED_DTYPES = {
    "source_spike_index": "int64",
    "spike_time_samples": "int64",
    "spike_time_s": "float64",
    "cluster_id": "int32",
    "is_good_cluster": "bool",
    "peak_channel_index": "int32",
    "template_peak_y_um": "float32",
    "template_com_y_um": "float32",
    "processing_chunk_id": "int32",
    "read_window_id": "int32",
    "localization_attempted": "bool",
    "localization_success": "bool",
    "localization_missing": "bool",
    "x_um": "float32",
    "y_um": "float32",
    "amplitude": "float32",
}


DEFAULT_DREDGE_CONFIG = {
    "method": "dredge_ap",
    "direction": "y",
    "rigid": False,
    "win_shape": "gaussian",
    "win_step_um": 200.0,
    "win_scale_um": 300.0,
    "bin_s": 2.0,
    "bin_um": 5.0,
    "max_disp_um": 120.0,
    "time_horizon_s": 400.0,
    "mincorr": 0.1,
    "device": "cpu",
    "progress_bar": False,
    "verbose": False,
}


@dataclass
class SessionSpec:
    session_name: str
    session_date: str
    session_order: int
    localized_spike_table_path: Path
    ks_path: Path
    duration_s: float
    partial_or_exact: str
    sample_rate_hz: float
    channel_positions_path: Path


class LightweightRecording:
    def __init__(self, channel_locations: np.ndarray, sample_rate_hz: float, num_samples: int):
        self._channel_locations = np.asarray(channel_locations, dtype=np.float64)
        self.contact_positions = self._channel_locations
        self._sample_rate_hz = float(sample_rate_hz)
        self._num_samples = int(num_samples)

    def get_num_segments(self) -> int:
        return 1

    def get_num_samples(self) -> int:
        return self._num_samples

    def sample_index_to_time(self, sample_index, segment_index: int = 0):
        sample_index = np.asarray(sample_index, dtype=np.float64)
        return sample_index / self._sample_rate_hz

    def get_probe(self):
        return self

    def get_channel_locations(self):
        return self._channel_locations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-name", action="append", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_manifest(session_names: list[str] | None) -> list[SessionSpec]:
    manifest = pd.read_csv(MANIFEST_PATH).sort_values("session_order")
    if session_names:
        session_names_set = set(session_names)
        manifest = manifest[manifest["session_name"].isin(session_names_set)].copy()
    specs = []
    for row in manifest.itertuples(index=False):
        specs.append(
            SessionSpec(
                session_name=row.session_name,
                session_date=row.session_date,
                session_order=int(row.session_order),
                localized_spike_table_path=Path(row.localized_spike_table_path),
                ks_path=Path(row.ks_path),
                duration_s=float(row.duration_s),
                partial_or_exact=str(row.partial_or_exact),
                sample_rate_hz=float(row.sample_rate_hz),
                channel_positions_path=Path(row.channel_positions_path),
            )
        )
    return specs


def load_selected_units() -> pd.DataFrame:
    return pd.read_csv(SELECTED_UNITS_PATH).sort_values("selection_rank")


def load_session_tracking(session_name: str) -> pd.DataFrame:
    path = ATTACHED_SPIKES_ROOT / session_name / f"{session_name}_tracked_spikes.csv.gz"
    usecols = ["source_spike_index", "cluster_id", "tracked_unit_id", "conflict_flag"]
    df = pd.read_csv(path, compression="gzip", usecols=usecols, low_memory=False)
    return df


def build_motion_inputs(df: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(mask.sum())
    peaks = np.zeros(
        n,
        dtype=[
            ("sample_index", "int64"),
            ("channel_index", "int64"),
            ("amplitude", "float32"),
            ("segment_index", "int64"),
        ],
    )
    peaks["sample_index"] = df.loc[mask, "spike_time_samples"].to_numpy(dtype=np.int64)
    peaks["channel_index"] = df.loc[mask, "peak_channel_index"].to_numpy(dtype=np.int64)
    peaks["amplitude"] = df.loc[mask, "amplitude"].to_numpy(dtype=np.float32)
    peaks["segment_index"] = 0

    peak_locations = np.zeros(n, dtype=[("x", "float32"), ("y", "float32"), ("z", "float32")])
    peak_locations["x"] = np.nan_to_num(df.loc[mask, "x_um"].to_numpy(dtype=np.float32), nan=0.0)
    peak_locations["y"] = df.loc[mask, "y_um"].to_numpy(dtype=np.float32)
    peak_locations["z"] = 0.0
    return peaks, peak_locations


def save_motion_trace(session_root: Path, motion) -> Path:
    motion_root = session_root / "motion"
    if motion_root.exists():
        shutil.rmtree(motion_root)
    motion.save(motion_root)

    rows = []
    for segment_index, (disp, temporal_bins_s) in enumerate(zip(motion.displacement, motion.temporal_bins_s)):
        for t_idx, time_s in enumerate(np.asarray(temporal_bins_s)):
            for s_idx, spatial_bin_um in enumerate(np.asarray(motion.spatial_bins_um)):
                rows.append(
                    {
                        "segment_index": int(segment_index),
                        "time_bin_center_s": float(time_s),
                        "spatial_bin_um": float(spatial_bin_um),
                        "displacement_um": float(disp[t_idx, s_idx]),
                    }
                )
    trace_df = pd.DataFrame(rows)
    trace_path = session_root / "dredge_motion_trace.csv"
    csv_write(trace_df, trace_path)
    return trace_path


def attach_tracking(corrected_df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    tracking_df = load_session_tracking(session_name)
    merged = corrected_df.merge(tracking_df, on=["source_spike_index", "cluster_id"], how="left")
    return merged


def session_palette(session_name: str, annotated_df: pd.DataFrame, selected_units_df: pd.DataFrame, limit: int = 4) -> pd.DataFrame:
    present = annotated_df.loc[
        annotated_df["tracked_unit_id"].notna() & annotated_df["dredge_applied"] & annotated_df["is_good_cluster"].astype(bool),
        "tracked_unit_id",
    ].astype(int)
    palette = selected_units_df[selected_units_df["tracked_unit_id"].isin(set(present.tolist()))].copy()
    return palette.sort_values("selection_rank").head(limit)


def make_before_after_raster(session_spec: SessionSpec, corrected_df: pd.DataFrame, selected_units_df: pd.DataFrame) -> Path:
    annotated = attach_tracking(corrected_df, session_spec.session_name)
    good_mask = corrected_df["is_good_cluster"].astype(bool) & corrected_df["dredge_applied"].astype(bool)
    plot_df = annotated.loc[good_mask].copy()
    if len(plot_df) > 300_000:
        plot_df = plot_df.sample(n=300_000, random_state=0)

    palette = session_palette(session_spec.session_name, annotated, selected_units_df)
    y_values = np.concatenate(
        [
            plot_df["y_um"].to_numpy(dtype=np.float64),
            plot_df["y_um_dredge"].to_numpy(dtype=np.float64),
        ]
    )
    y_lo, y_hi = np.nanquantile(y_values, [0.01, 0.99])
    y_pad = max(10.0, 0.03 * float(y_hi - y_lo))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=170, sharex=True, sharey=True)
    for ax, y_col, title in zip(
        axes,
        ["y_um", "y_um_dredge"],
        ["before DREDge", "after DREDge"],
    ):
        ax.scatter(
            plot_df["spike_time_s"].to_numpy(dtype=np.float64),
            plot_df[y_col].to_numpy(dtype=np.float64),
            s=0.35,
            alpha=0.12,
            color="#7a7a7a",
            rasterized=True,
            linewidths=0,
        )
        for row in palette.itertuples(index=False):
            unit_df = annotated[
                annotated["tracked_unit_id"].fillna(-1).astype(int) == int(row.tracked_unit_id)
            ].copy()
            unit_df = unit_df[unit_df["dredge_applied"] & unit_df["is_good_cluster"].astype(bool)]
            if unit_df.empty:
                continue
            if len(unit_df) > 15_000:
                unit_df = unit_df.sample(n=15_000, random_state=0)
            ax.scatter(
                unit_df["spike_time_s"].to_numpy(dtype=np.float64),
                unit_df[y_col].to_numpy(dtype=np.float64),
                s=0.8,
                alpha=0.40,
                color=row.color_hex,
                linewidths=0,
                rasterized=True,
                label=row.tracked_label,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.08, linewidth=0.5)
    axes[0].set_ylabel("depth (um)")
    axes[0].set_ylim(y_hi + y_pad, y_lo - y_pad)
    if not palette.empty:
        handles, labels = axes[1].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[1].legend(unique.values(), unique.keys(), loc="upper right", fontsize=7, frameon=True)
    fig.suptitle(
        f"{session_spec.session_name} monopolar localized spikes before vs after DREDge",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = DREDGE_ROOT / session_spec.session_name / f"{session_spec.session_name}_before_after_dredge_raster.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def run_session(session_spec: SessionSpec, selected_units_df: pd.DataFrame, force: bool) -> dict[str, object]:
    session_root = DREDGE_ROOT / session_spec.session_name
    corrected_csv = session_root / f"{session_spec.session_name}_corrected_spike_table.csv.gz"
    session_summary_path = session_root / f"{session_spec.session_name}_dredge_session_summary.json"
    if corrected_csv.exists() and session_summary_path.exists() and not force:
        return json.loads(session_summary_path.read_text())

    session_root.mkdir(parents=True, exist_ok=True)
    print(f"[{session_spec.session_name}] loading localized spike table")
    localized_df = pd.read_csv(
        session_spec.localized_spike_table_path,
        compression="gzip",
        low_memory=False,
        dtype=LOCALIZED_DTYPES,
    )

    dredge_mask = (
        localized_df["localization_success"].astype(bool).to_numpy()
        & np.isfinite(localized_df["spike_time_samples"].to_numpy(dtype=np.float64))
        & np.isfinite(localized_df["peak_channel_index"].to_numpy(dtype=np.float64))
        & np.isfinite(localized_df["y_um"].to_numpy(dtype=np.float64))
        & np.isfinite(localized_df["amplitude"].to_numpy(dtype=np.float64))
    )
    n_dredge_input_rows = int(dredge_mask.sum())
    n_good_dredge_input_rows = int(
        (localized_df["is_good_cluster"].astype(bool).to_numpy() & dredge_mask).sum()
    )
    print(f"[{session_spec.session_name}] building motion inputs for {n_dredge_input_rows:,} spikes")
    peaks, peak_locations = build_motion_inputs(localized_df, dredge_mask)

    channel_locations = np.load(session_spec.channel_positions_path)
    num_samples = int(round(session_spec.duration_s * session_spec.sample_rate_hz))
    recording = LightweightRecording(channel_locations, session_spec.sample_rate_hz, num_samples)

    print(f"[{session_spec.session_name}] estimating DREDge motion")
    motion = estimate_motion(
        recording,
        peaks=peaks,
        peak_locations=peak_locations,
        **DEFAULT_DREDGE_CONFIG,
    )
    print(f"[{session_spec.session_name}] correcting peak locations")
    displacements = compute_peak_displacements(peaks, motion, recording, peak_locations=peak_locations)
    corrected_peak_locations = correct_motion_on_peaks(peaks, peak_locations, motion, recording)
    trace_path = save_motion_trace(session_root, motion)

    corrected_df = localized_df.copy()
    corrected_df["y_um_dredge"] = corrected_df["y_um"].to_numpy(dtype=np.float64)
    corrected_df["dredge_peak_displacement_um"] = np.nan
    corrected_df["dredge_applied"] = dredge_mask
    corrected_df.loc[dredge_mask, "y_um_dredge"] = corrected_peak_locations["y"].astype(np.float64)
    corrected_df.loc[dredge_mask, "dredge_peak_displacement_um"] = displacements.astype(np.float64)
    print(f"[{session_spec.session_name}] writing corrected spike table")
    corrected_df.to_csv(corrected_csv, index=False, compression="gzip")

    print(f"[{session_spec.session_name}] rendering before/after raster")
    raster_path = make_before_after_raster(session_spec, corrected_df, selected_units_df)
    displacement_abs = np.abs(displacements.astype(np.float64))
    summary = {
        "created_at": now_iso(),
        "session_name": session_spec.session_name,
        "session_date": session_spec.session_date,
        "session_order": int(session_spec.session_order),
        "partial_or_exact": session_spec.partial_or_exact,
        "input_localized_spike_table": str(session_spec.localized_spike_table_path),
        "corrected_spike_table_csv_gz": str(corrected_csv),
        "motion_trace_csv": str(trace_path),
        "motion_root": str(session_root / "motion"),
        "before_after_raster_png": str(raster_path),
        "dredge_config": DEFAULT_DREDGE_CONFIG,
        "n_input_rows": int(len(localized_df)),
        "n_dredge_input_rows": n_dredge_input_rows,
        "n_good_dredge_input_rows": n_good_dredge_input_rows,
        "mean_abs_displacement_um": float(displacement_abs.mean()),
        "median_abs_displacement_um": float(np.median(displacement_abs)),
        "max_abs_displacement_um": float(displacement_abs.max()),
        "q95_abs_displacement_um": float(np.quantile(displacement_abs, 0.95)),
        "motion_min_um": float(np.min(motion.displacement[0])),
        "motion_max_um": float(np.max(motion.displacement[0])),
        "motion_spatial_bin_count": int(motion.spatial_bins_um.size),
        "motion_temporal_bin_count": int(motion.temporal_bins_s[0].size),
    }
    dump_json(session_summary_path, summary)
    print(f"[{session_spec.session_name}] done")
    return summary


def main() -> None:
    args = parse_args()
    DREDGE_ROOT.mkdir(parents=True, exist_ok=True)
    dump_json(
        DREDGE_CONFIG_PATH,
        {
            "created_at": now_iso(),
            "config": DEFAULT_DREDGE_CONFIG,
            "script": str(Path(__file__)),
        },
    )
    selected_units_df = load_selected_units()
    session_summaries = []
    for session_spec in load_manifest(args.session_name):
        session_summaries.append(run_session(session_spec, selected_units_df, args.force))
    dump_json(DREDGE_ROOT / "dredge_session_summaries_phase1to4.json", session_summaries)


if __name__ == "__main__":
    main()
