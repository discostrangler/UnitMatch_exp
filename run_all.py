#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


EXAMPLE_PAIR_KEY = "AL032_2019-11-21_to_2019-11-22"
MATCH_THRESHOLD = 0.5
BIN_SIZE_SEC = 0.01
REFPOP_MIN_REFS = 6
NEGATIVE_SAMPLE_CAP = 500


@dataclass
class SourceOutputs:
    full_root: Path
    multi_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the clean local_runs_1 UnitMatch package.")
    parser.add_argument("--force-refresh", action="store_true", help="Delete generated outputs inside local_runs_1 and rebuild them.")
    parser.add_argument("--force-source", action="store_true", help="Re-run the shared source scripts under local_runs/* before packaging.")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_runtime(root: Path) -> None:
    mpl_dir = root / ".mplconfig"
    cache_dir = root / ".cache"
    mpl_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    sys.path.insert(0, str(root / "UnitMatchPy" / "tools"))
    sys.path.insert(0, str(root / "UnitMatchPy"))
    sys.path.insert(0, str(root / "UnitMatchPy" / "DeepUnitMatch"))


def reset_outputs(package_dir: Path) -> None:
    for name in ["data", "report", "manifest.json"]:
        path = package_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def ensure_source_outputs(root: Path, force_source: bool) -> SourceOutputs:
    full_root = root / "local_runs" / "full_data_classic"
    multi_root = root / "local_runs" / "multi_session_by_mouse"

    full_ready = (full_root / "within_day_metrics.csv").exists() and (full_root / "pair_tracking.csv").exists()
    multi_ready = (multi_root / "summary.csv").exists() and len(list(multi_root.glob("*/classic/MatchTable.csv"))) >= 5

    if force_source or not full_ready:
        subprocess.run([sys.executable, str(root / "UnitMatchPy" / "tools" / "run_full_data_classic_report.py")], check=True, cwd=root)

    if force_source or not multi_ready:
        import build_paper_structured_report as paper_report

        paper_report.build_multi_session_outputs(root)

    if not ((full_root / "within_day_metrics.csv").exists() and (full_root / "pair_tracking.csv").exists()):
        raise RuntimeError("Missing local_runs/full_data_classic outputs.")
    if not ((multi_root / "summary.csv").exists() and len(list(multi_root.glob("*/classic/MatchTable.csv"))) >= 5):
        raise RuntimeError("Missing local_runs/multi_session_by_mouse outputs.")

    return SourceOutputs(full_root=full_root, multi_root=multi_root)


@lru_cache(maxsize=64)
def parse_sample_rate(ks_dir_str: str) -> float:
    params_path = Path(ks_dir_str) / "params.py"
    sample_rate = 30000.0
    if params_path.exists():
        for line in params_path.read_text().splitlines():
            if "sample_rate" in line and "=" in line:
                try:
                    sample_rate = float(line.split("=", 1)[1].strip())
                    break
                except ValueError:
                    continue
    return sample_rate


@lru_cache(maxsize=64)
def load_session_spikes(ks_dir_str: str) -> tuple[np.ndarray, np.ndarray, float]:
    ks_dir = Path(ks_dir_str)
    spike_times = np.load(ks_dir / "spike_times.npy").astype(float).ravel()
    spike_clusters = np.load(ks_dir / "spike_clusters.npy").astype(int).ravel()
    return spike_times, spike_clusters, parse_sample_rate(ks_dir_str)


def spike_times_for_unit(ks_dir: Path, unit_id: int) -> np.ndarray:
    spike_times, spike_clusters, sample_rate = load_session_spikes(str(ks_dir))
    return spike_times[spike_clusters == int(unit_id)] / sample_rate


def isi_histogram(spike_times_sec: np.ndarray, max_sec: float = 0.1, bin_sec: float = 0.001) -> np.ndarray | None:
    if spike_times_sec.size < 3:
        return None
    intervals = np.diff(np.sort(spike_times_sec))
    intervals = intervals[(intervals >= 0.0) & (intervals <= max_sec)]
    if intervals.size < 2:
        return None
    bins = np.arange(0.0, max_sec + bin_sec, bin_sec)
    hist, _ = np.histogram(intervals, bins=bins)
    hist = hist.astype(float)
    if hist.sum() <= 0:
        return None
    return hist / hist.sum()


def split_half_isi_histograms(spike_times_sec: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if spike_times_sec.size < 6:
        return None, None
    midpoint = 0.5 * (float(spike_times_sec.min()) + float(spike_times_sec.max()))
    return isi_histogram(spike_times_sec[spike_times_sec <= midpoint]), isi_histogram(spike_times_sec[spike_times_sec > midpoint])


def corr_or_nan(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.size != b.size or a.size < 3:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pair_unit_lookup(classic_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    clus_info = pickle.load(open(classic_dir / "ClusInfo.pickle", "rb"))
    return np.asarray(clus_info["original_ids"]).astype(int), np.asarray(clus_info["session_id"]).astype(int)


def reciprocal_match_pairs(classic_dir: Path) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    match_prob = np.load(classic_dir / "MatchProb.npy")
    original_ids, session_id = pair_unit_lookup(classic_dir)
    first_idx = np.flatnonzero(session_id == 0)
    second_idx = np.flatnonzero(session_id == 1)
    cross = match_prob[np.ix_(first_idx, second_idx)]
    cross_rev = match_prob[np.ix_(second_idx, first_idx)]
    row_best = np.argmax(cross, axis=1)
    rev_best = np.argmax(cross_rev, axis=1)

    positives: list[tuple[int, int]] = []
    taken: set[tuple[int, int]] = set()
    for i, j in enumerate(row_best):
        if rev_best[j] == i and cross[i, j] > MATCH_THRESHOLD and cross_rev[j, i] > MATCH_THRESHOLD:
            positives.append((int(original_ids[first_idx[i]]), int(original_ids[second_idx[j]])))
            taken.add((int(i), int(j)))

    negatives: list[tuple[int, int]] = []
    for i in range(len(first_idx)):
        for j in range(len(second_idx)):
            if (i, j) not in taken:
                negatives.append((int(original_ids[first_idx[i]]), int(original_ids[second_idx[j]])))
    return positives, negatives


def nearest_neighbor_map(classic_dir: Path) -> dict[int, int]:
    wave_info = np.load(classic_dir / "WaveformInfo.npz")
    original_ids, _ = pair_unit_lookup(classic_dir)
    avg_centroid = np.nanmean(wave_info["avg_centroid"], axis=2).T
    dist = np.linalg.norm(avg_centroid[:, None, :] - avg_centroid[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    nn = np.argmin(dist, axis=1)
    return {int(original_ids[i]): int(original_ids[j]) for i, j in enumerate(nn)}


@lru_cache(maxsize=128)
def unit_count_matrix_cached(ks_dir_str: str, unit_ids: tuple[int, ...], bin_size_sec: float = BIN_SIZE_SEC) -> tuple[np.ndarray, np.ndarray]:
    spike_times, spike_clusters, sample_rate = load_session_spikes(ks_dir_str)
    times_sec = spike_times / sample_rate
    if times_sec.size == 0:
        return np.zeros((len(unit_ids), 0), dtype=float), np.array([], dtype=float)
    start = float(times_sec.min())
    stop = float(times_sec.max())
    edges = np.arange(start, stop + bin_size_sec, bin_size_sec)
    if edges.size < 3:
        edges = np.linspace(start, stop + bin_size_sec, 3)
    rows = [np.histogram(times_sec[spike_clusters == int(unit_id)], bins=edges)[0].astype(float) for unit_id in unit_ids]
    return np.vstack(rows), edges


def unit_count_matrix(ks_dir: Path, unit_ids: list[int], bin_size_sec: float = BIN_SIZE_SEC) -> tuple[np.ndarray, np.ndarray]:
    return unit_count_matrix_cached(str(ks_dir), tuple(int(x) for x in unit_ids), bin_size_sec)


def correlation_matrix(counts: np.ndarray) -> np.ndarray:
    if counts.shape[0] < 2 or counts.shape[1] < 4:
        return np.full((counts.shape[0], counts.shape[0]), np.nan)
    corr = np.corrcoef(counts)
    corr = np.asarray(corr, dtype=float)
    corr[~np.isfinite(corr)] = np.nan
    np.fill_diagonal(corr, np.nan)
    return corr


def vector_correlation(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 4:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def add_gap_columns(pair_df: pd.DataFrame) -> pd.DataFrame:
    out = pair_df.copy()
    gaps = []
    for row in out.to_dict(orient="records"):
        try:
            gaps.append(int((pd.Timestamp(row["session_b_label"]) - pd.Timestamp(row["session_a_label"])).days))
        except Exception:
            gaps.append(0 if row["pair_type"] == "chronic" else np.nan)
    out["gap_days"] = gaps
    return out


def build_isi_rows(within_df: pd.DataFrame, pair_df: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(0)
    within_rows = []
    for row in within_df.to_dict(orient="records"):
        classic_dir = Path(row["output_dir"]) / "classic"
        unit_ids = [int(x) for x in np.asarray(pickle.load(open(classic_dir / "ClusInfo.pickle", "rb"))["original_ids"]).astype(int)]
        neighbor_lookup = nearest_neighbor_map(classic_dir)
        pos = []
        neg = []
        seen = set()
        for unit_id in unit_ids:
            score = corr_or_nan(*split_half_isi_histograms(spike_times_for_unit(Path(row["ks_dir"]), unit_id)))
            if np.isfinite(score):
                pos.append(score)
            neighbor_id = neighbor_lookup.get(unit_id)
            if neighbor_id is None:
                continue
            pair = tuple(sorted((unit_id, neighbor_id)))
            if pair in seen:
                continue
            seen.add(pair)
            score = corr_or_nan(isi_histogram(spike_times_for_unit(Path(row["ks_dir"]), unit_id)), isi_histogram(spike_times_for_unit(Path(row["ks_dir"]), neighbor_id)))
            if np.isfinite(score):
                neg.append(score)
        auc = float("nan")
        if pos and neg:
            auc = float(roc_auc_score(np.array([1] * len(pos) + [0] * len(neg)), np.array(pos + neg, dtype=float)))
        within_rows.append({"mouse": row["mouse"], "group": row["group"], "session_key": row["session_key"], "session_label": row["session_label"], "ks_dir": row["ks_dir"], "classic_dir": str(classic_dir), "positive_count": len(pos), "negative_count": len(neg), "auc": auc, "positive_scores": pos, "negative_scores": neg})

    across_rows = []
    for row in pair_df.to_dict(orient="records"):
        classic_dir = Path(row["output_dir"]) / "classic"
        ks_dirs = [Path(p) for p in ast.literal_eval(row["ks_dirs"])]
        positives, negatives = reciprocal_match_pairs(classic_dir)
        if len(negatives) > NEGATIVE_SAMPLE_CAP:
            negatives = [negatives[idx] for idx in rng.choice(len(negatives), size=NEGATIVE_SAMPLE_CAP, replace=False)]
        pos = []
        neg = []
        for unit_a, unit_b in positives:
            score = corr_or_nan(isi_histogram(spike_times_for_unit(ks_dirs[0], unit_a)), isi_histogram(spike_times_for_unit(ks_dirs[1], unit_b)))
            if np.isfinite(score):
                pos.append(score)
        for unit_a, unit_b in negatives:
            score = corr_or_nan(isi_histogram(spike_times_for_unit(ks_dirs[0], unit_a)), isi_histogram(spike_times_for_unit(ks_dirs[1], unit_b)))
            if np.isfinite(score):
                neg.append(score)
        auc = float("nan")
        if pos and neg:
            auc = float(roc_auc_score(np.array([1] * len(pos) + [0] * len(neg)), np.array(pos + neg, dtype=float)))
        across_rows.append({"mouse": row["mouse"], "pair_type": row["pair_type"], "pair_key": row["pair_key"], "positive_count": len(pos), "negative_count": len(neg), "auc": auc, "positive_scores": pos, "negative_scores": neg})
    return within_rows, across_rows


def build_refpop_rows(within_df: pd.DataFrame, pair_df: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(1)
    within_rows = []
    for row in within_df.to_dict(orient="records"):
        classic_dir = Path(row["output_dir"]) / "classic"
        unit_ids = [int(x) for x in np.asarray(pickle.load(open(classic_dir / "ClusInfo.pickle", "rb"))["original_ids"]).astype(int)]
        counts, _ = unit_count_matrix(Path(row["ks_dir"]), unit_ids)
        split = counts.shape[1] // 2
        corr_a = correlation_matrix(counts[:, :split])
        corr_b = correlation_matrix(counts[:, split : split * 2])
        neighbor_lookup = nearest_neighbor_map(classic_dir)
        id_to_idx = {unit_id: idx for idx, unit_id in enumerate(unit_ids)}
        pos = []
        neg = []
        for unit_id in unit_ids:
            idx = id_to_idx[unit_id]
            mask = np.ones(len(unit_ids), dtype=bool)
            mask[idx] = False
            score = vector_correlation(corr_a[idx, mask], corr_b[idx, mask])
            if np.isfinite(score):
                pos.append(score)
            neighbor_id = neighbor_lookup.get(unit_id)
            if neighbor_id is None or neighbor_id not in id_to_idx:
                continue
            jdx = id_to_idx[neighbor_id]
            mask = np.ones(len(unit_ids), dtype=bool)
            mask[idx] = False
            mask[jdx] = False
            score = vector_correlation(corr_a[idx, mask], corr_b[jdx, mask])
            if np.isfinite(score):
                neg.append(score)
        auc = float("nan")
        if pos and neg:
            auc = float(roc_auc_score(np.array([1] * len(pos) + [0] * len(neg)), np.array(pos + neg, dtype=float)))
        within_rows.append({"mouse": row["mouse"], "group": row["group"], "session_key": row["session_key"], "positive_count": len(pos), "negative_count": len(neg), "auc": auc, "positive_scores": pos, "negative_scores": neg})

    across_rows = []
    for row in pair_df.to_dict(orient="records"):
        classic_dir = Path(row["output_dir"]) / "classic"
        ks_dirs = [Path(p) for p in ast.literal_eval(row["ks_dirs"])]
        positives, _ = reciprocal_match_pairs(classic_dir)
        if len(positives) < REFPOP_MIN_REFS + 1:
            across_rows.append({"mouse": row["mouse"], "pair_type": row["pair_type"], "pair_key": row["pair_key"], "positive_count": 0, "negative_count": 0, "auc": float("nan"), "positive_scores": [], "negative_scores": []})
            continue
        first_ids = [pair[0] for pair in positives]
        second_ids = [pair[1] for pair in positives]
        corr_a = correlation_matrix(unit_count_matrix(ks_dirs[0], first_ids)[0])
        corr_b = correlation_matrix(unit_count_matrix(ks_dirs[1], second_ids)[0])
        pos = []
        for idx in range(len(positives)):
            mask = np.ones(len(positives), dtype=bool)
            mask[idx] = False
            if int(mask.sum()) < REFPOP_MIN_REFS:
                continue
            score = vector_correlation(corr_a[idx, mask], corr_b[idx, mask])
            if np.isfinite(score):
                pos.append(score)
        max_pairs = len(positives) * (len(positives) - 1)
        if max_pairs <= NEGATIVE_SAMPLE_CAP:
            sampled = [(i, j) for i in range(len(positives)) for j in range(len(positives)) if i != j]
        else:
            sampled = set()
            while len(sampled) < NEGATIVE_SAMPLE_CAP:
                i = int(rng.integers(0, len(positives)))
                j = int(rng.integers(0, len(positives)))
                if i != j:
                    sampled.add((i, j))
            sampled = list(sampled)
        neg = []
        for idx, jdx in sampled:
            mask = np.ones(len(positives), dtype=bool)
            mask[idx] = False
            mask[jdx] = False
            if int(mask.sum()) < REFPOP_MIN_REFS:
                continue
            score = vector_correlation(corr_a[idx, mask], corr_b[jdx, mask])
            if np.isfinite(score):
                neg.append(score)
        auc = float("nan")
        if pos and neg:
            auc = float(roc_auc_score(np.array([1] * len(pos) + [0] * len(neg)), np.array(pos + neg, dtype=float)))
        across_rows.append({"mouse": row["mouse"], "pair_type": row["pair_type"], "pair_key": row["pair_key"], "positive_count": len(pos), "negative_count": len(neg), "auc": auc, "positive_scores": pos, "negative_scores": neg})
    return within_rows, across_rows


def write_compact_tables(package_dir: Path, full_root: Path, within_df: pd.DataFrame, pair_df: pd.DataFrame, isi_within: list[dict[str, object]], isi_across: list[dict[str, object]], ref_within: list[dict[str, object]], ref_across: list[dict[str, object]]) -> dict[str, Path]:
    data_dir = package_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    within_path = data_dir / "within_day_metrics.csv"
    pair_path = data_dir / "pair_tracking.csv"
    within_df.to_csv(within_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    shutil.copy2(full_root / "summary.json", data_dir / "full_data_summary.json")

    table_map = {"within_day_metrics": within_path, "pair_tracking": pair_path, "full_data_summary": data_dir / "full_data_summary.json"}
    for name, rows in [
        ("isi_within_day", isi_within),
        ("isi_across_day", isi_across),
        ("refpop_within_day", ref_within),
        ("refpop_across_day", ref_across),
    ]:
        df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows])
        path = data_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        table_map[name] = path
    return table_map


def build_classifier_figure(report_dir: Path, pair_df: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    score_order = [
        ("spatial_decay_score", "D"),
        ("waveform_score", "W"),
        ("amp_score", "A"),
        ("centroid_overlord_score", "C"),
        ("centroid_dist", "V"),
        ("trajectory_score", "R"),
        ("TotalScore", "T"),
    ]

    pair_row = pair_df.loc[pair_df["pair_key"] == EXAMPLE_PAIR_KEY].iloc[0]
    pair_dir = Path(pair_row["output_dir"])
    classic_dir = pair_dir / "classic"
    match_table = pd.read_csv(classic_dir / "MatchTable.csv")
    wave_info = np.load(classic_dir / "WaveformInfo.npz")
    clus_info = pickle.load(open(classic_dir / "ClusInfo.pickle", "rb"))
    top_pair = pd.read_csv(pair_dir / "top_reciprocal_pairs.csv").iloc[0]
    unit_a = int(top_pair["unit_a"])
    unit_b = int(top_pair["unit_b"])
    original_ids = np.asarray(clus_info["original_ids"]).astype(int)
    session_id = np.asarray(clus_info["session_id"]).astype(int)
    idx_a = int(np.flatnonzero((original_ids == unit_a) & (session_id == 0))[0])
    idx_b = int(np.flatnonzero((original_ids == unit_b) & (session_id == 1))[0])
    within_neighbors = match_table[(match_table["RecSes 1"] == 1) & (match_table["RecSes 2"] == 1) & (match_table["ID1"] == unit_a) & (match_table["ID2"] != unit_a)].sort_values("centroid_overlord_score", ascending=False)
    neighbor_unit = int(within_neighbors.iloc[0]["ID2"])
    idx_neighbor = int(np.flatnonzero((original_ids == neighbor_unit) & (session_id == 0))[0])
    matched_row = match_table[(match_table["RecSes 1"] == 1) & (match_table["RecSes 2"] == 2) & (match_table["ID1"] == unit_a) & (match_table["ID2"] == unit_b)].iloc[0]
    neighbor_row = within_neighbors.iloc[0]

    avg_waveform = wave_info["avg_waveform"]
    avg_centroid_per_tp = wave_info["avg_waveform_per_tp"]
    time_ms = np.arange(avg_waveform.shape[0]) / 30.0

    def mean_wave(idx: int) -> np.ndarray:
        return np.nan_to_num(np.nanmean(avg_waveform[:, idx, :], axis=1), nan=0.0)

    def trajectory(idx: int) -> np.ndarray:
        xy = np.nanmean(avg_centroid_per_tp[1:, idx, :, :], axis=-1).T
        return xy[np.isfinite(xy).all(axis=1)]

    wave_a = mean_wave(idx_a)
    wave_b = mean_wave(idx_b)
    wave_neighbor = mean_wave(idx_neighbor)
    traj_a = trajectory(idx_a)
    traj_b = trajectory(idx_b)
    traj_neighbor = trajectory(idx_neighbor)

    total_score = match_table["TotalScore"].to_numpy().reshape(len(original_ids), len(original_ids))
    normalized_score = total_score.copy()
    if normalized_score.max() > normalized_score.min():
        normalized_score = (normalized_score - normalized_score.min()) / (normalized_score.max() - normalized_score.min())
    else:
        normalized_score = np.zeros_like(normalized_score)
    session_switch = np.asarray(clus_info["session_switch"]).astype(int)
    boundary = int(session_switch[1]) if len(session_switch) > 2 else normalized_score.shape[0] // 2

    centroid_scores = match_table["centroid_overlord_score"].to_numpy().reshape(len(original_ids), len(original_ids))
    centroid_scores = centroid_scores.copy()
    np.fill_diagonal(centroid_scores, -np.inf)
    nn_idx = np.argmax(centroid_scores, axis=1)
    diagonal = np.diag(total_score)
    neighbor_scores = total_score[np.arange(len(original_ids)), nn_idx]
    cross_match = match_table[(match_table["RecSes 1"] != match_table["RecSes 2"]) & (match_table["UM Probabilities"] > MATCH_THRESHOLD)]["TotalScore"].to_numpy()
    cross_nonmatch = match_table[(match_table["RecSes 1"] != match_table["RecSes 2"]) & (match_table["UM Probabilities"] <= MATCH_THRESHOLD)]["TotalScore"].to_numpy()
    prob_matrix = np.load(classic_dir / "MatchProb.npy")

    fig = plt.figure(figsize=(16.2, 10.2), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, width_ratios=[1.1, 1.0, 1.0, 1.35], height_ratios=[1.0, 1.0, 1.15, 0.95])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(time_ms, wave_a, color="black", lw=2)
    ax.plot(time_ms, wave_neighbor, color="#1d4ed8", lw=1.8)
    ax.set_title("a  Weighted-average waveforms", loc="left", fontweight="bold")
    ax.text(0.03, 0.08, "Closest neighbor\nwithin day", transform=ax.transAxes, color="#1d4ed8", fontsize=9)
    ax.set_xlabel("ms")

    ax = fig.add_subplot(gs[0, 1])
    if len(traj_a):
        ax.plot(traj_a[:, 0], traj_a[:, 1], color="black", lw=1.8)
    if len(traj_neighbor):
        ax.plot(traj_neighbor[:, 0], traj_neighbor[:, 1], color="#1d4ed8", lw=1.8)
    ax.set_title("b  Trajectories", loc="left", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    y = np.arange(len(score_order))
    ax.barh(y, [float(neighbor_row[key]) for key, _ in score_order], color="#1d4ed8")
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in score_order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("c  Similarity scores", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(time_ms, wave_a, color="black", lw=2)
    ax.plot(time_ms, wave_b, color="#dc2626", lw=1.8)
    ax.set_title("d", loc="left", fontweight="bold")
    ax.text(0.03, 0.08, "Best match\nacross days", transform=ax.transAxes, color="#dc2626", fontsize=9)
    ax.set_xlabel("ms")

    ax = fig.add_subplot(gs[1, 1])
    if len(traj_a):
        ax.plot(traj_a[:, 0], traj_a[:, 1], color="black", lw=1.8)
    if len(traj_b):
        ax.plot(traj_b[:, 0], traj_b[:, 1], color="#dc2626", lw=1.8)
    ax.set_title("e", loc="left", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 2])
    ax.barh(y, [float(matched_row[key]) for key, _ in score_order], color="#dc2626")
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in score_order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("f", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[:2, 3])
    ax.imshow(normalized_score, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    n_units = normalized_score.shape[0]
    for x0, y0, w, h, color in [
        (0, 0, boundary, boundary, "#1d4ed8"),
        (boundary, 0, n_units - boundary, boundary, "#ef4444"),
        (0, boundary, boundary, n_units - boundary, "#ef4444"),
        (boundary, boundary, n_units - boundary, n_units - boundary, "#1d4ed8"),
    ]:
        ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), w, h, fill=False, edgecolor=color, linewidth=1.5))
    ax.axvline(boundary - 0.5, color="black", lw=0.8)
    ax.axhline(boundary - 0.5, color="black", lw=0.8)
    ax.set_title("g  Total score matrix", loc="left", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    bins = np.linspace(0, 1, 41)
    ax = fig.add_subplot(gs[2, 0:2])
    ax.hist(neighbor_scores, bins=bins, density=True, alpha=0.7, color="#2563eb", label="Neighbors")
    ax.hist(diagonal, bins=bins, density=True, alpha=0.6, color="#16a34a", label="Same units")
    ax.axvline(0.6, color="#6b7280", ls="--", lw=1.4)
    ax.set_title("h  Pairs within days", loc="left", fontweight="bold")
    ax.set_xlabel("T")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[2, 2:4])
    ax.hist(cross_nonmatch, bins=bins, density=True, alpha=0.8, color="#fca5a5", label="Nonmatches")
    ax.hist(cross_match, bins=bins, density=True, alpha=0.6, color="#dc2626", label="Matches")
    ax.axvline(0.6, color="#6b7280", ls="--", lw=1.4)
    ax.set_title("i  Pairs across days", loc="left", fontweight="bold")
    ax.set_xlabel("T")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[3, 0:3])
    ax.hist(np.diag(prob_matrix), bins=bins, density=True, alpha=0.45, color="#16a34a", label="Within-day same")
    ax.hist(prob_matrix[np.arange(len(original_ids)), nn_idx], bins=bins, density=True, alpha=0.45, color="#2563eb", label="Within-day neighbor")
    cross_prob = prob_matrix[np.ix_(np.flatnonzero(session_id == 0), np.flatnonzero(session_id == 1))].ravel()
    ax.hist(cross_prob, bins=bins, density=True, alpha=0.65, color="#dc2626", label="Across days")
    ax.axvline(MATCH_THRESHOLD, color="#6b7280", ls="--", lw=1.4)
    ax.set_title("j  Match probability", loc="left", fontweight="bold")
    ax.set_xlabel("P(match)")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[3, 3])
    ax.text(0.02, 0.90, "Key example", fontsize=11, fontweight="bold")
    ax.text(0.02, 0.68, f"Reference unit: {unit_a}", fontsize=10)
    ax.text(0.02, 0.52, f"Within-day neighbor: {neighbor_unit}", fontsize=10)
    ax.text(0.02, 0.36, f"Across-day match: {unit_b}", fontsize=10)
    ax.text(0.02, 0.16, EXAMPLE_PAIR_KEY, fontsize=9)
    ax.axis("off")

    out_path = report_dir / "figures" / "figure3_like_classifier_overview.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_functional_figure(report_dir: Path, within_df: pd.DataFrame, pair_df: pd.DataFrame, isi_within: list[dict[str, object]], isi_across: list[dict[str, object]], ref_within: list[dict[str, object]], ref_across: list[dict[str, object]]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pair_row = pair_df.loc[pair_df["pair_key"] == EXAMPLE_PAIR_KEY].iloc[0]
    ks_dirs = [Path(p) for p in ast.literal_eval(pair_row["ks_dirs"])]
    pair_dir = Path(pair_row["output_dir"])
    top_pair = pd.read_csv(pair_dir / "top_reciprocal_pairs.csv").iloc[0]
    unit_a = int(top_pair["unit_a"])
    unit_b = int(top_pair["unit_b"])
    neighbor_unit = nearest_neighbor_map(pair_dir / "classic")[unit_a]

    def isi_curve(ks_dir: Path, unit_id: int) -> tuple[np.ndarray, np.ndarray]:
        hist = isi_histogram(spike_times_for_unit(ks_dir, unit_id), max_sec=0.5, bin_sec=0.005)
        if hist is None:
            return np.array([]), np.array([])
        x = np.arange(0.0025, 0.0025 + 0.005 * len(hist), 0.005) * 1000.0
        return x, hist

    def mouse_auc(rows: list[dict[str, object]]) -> pd.Series:
        return pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows]).groupby("mouse")["auc"].mean().sort_index()

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    x_same, y_same = isi_curve(ks_dirs[0], unit_a)
    x_neighbor, y_neighbor = isi_curve(ks_dirs[0], neighbor_unit)
    x_match, y_match = isi_curve(ks_dirs[1], unit_b)
    ax = axes[0, 0]
    ax.plot(x_same, y_same, color="black", lw=1.8, label=f"Unit {unit_a}")
    ax.plot(x_neighbor, y_neighbor, color="#2563eb", lw=1.6, label=f"Neighbor {neighbor_unit}")
    ax.plot(x_match, y_match, color="#dc2626", lw=1.6, label=f"Match {unit_b}")
    ax.set_xscale("log")
    ax.set_title("a  ISIs", loc="left", fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Proportion")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    bins = np.linspace(-0.2, 1.0, 45)
    ax.hist(np.concatenate([row["negative_scores"] for row in isi_within if row["negative_scores"]]), bins=bins, density=True, alpha=0.55, color="#2563eb", label="Within-day different")
    ax.hist(np.concatenate([row["positive_scores"] for row in isi_within if row["positive_scores"]]), bins=bins, density=True, alpha=0.45, color="#111827", label="Within-day same")
    ax.hist(np.concatenate([row["positive_scores"] for row in isi_across if row["positive_scores"]]), bins=bins, density=True, alpha=0.55, color="#dc2626", label="Across-day matches")
    ax.set_title("b  ISI correlation", loc="left", fontweight="bold")
    ax.set_xlabel("Correlation")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 2]
    within_mouse = mouse_auc(isi_within)
    across_mouse = mouse_auc(isi_across)
    x = np.arange(len(across_mouse.index))
    ax.bar(x - 0.17, within_mouse.reindex(across_mouse.index), width=0.34, color="#2563eb", label="Within day")
    ax.bar(x + 0.17, across_mouse.values, width=0.34, color="#dc2626", label="Across days")
    ax.axhline(0.94, color="#1d4ed8", ls="--", lw=1.2)
    ax.axhline(0.88, color="#991b1b", ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(across_mouse.index)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("c  ISI AUC by mouse", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    example_ref = next((row for row in ref_across if row["pair_key"] == EXAMPLE_PAIR_KEY and row["positive_count"] > 0), None)
    if example_ref is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "Reference-population analog\nnot available for example pair", ha="center", va="center")
    else:
        positives, _ = reciprocal_match_pairs(pair_dir / "classic")
        first_ids = [pair[0] for pair in positives]
        second_ids = [pair[1] for pair in positives]
        corr_a = correlation_matrix(unit_count_matrix(ks_dirs[0], first_ids)[0])
        corr_b = correlation_matrix(unit_count_matrix(ks_dirs[1], second_ids)[0])
        idx = next(i for i, pair in enumerate(positives) if pair == (unit_a, unit_b))
        order = np.argsort(np.nan_to_num(corr_a[idx], nan=-np.inf))[::-1]
        order = order[order != idx]
        ax.plot(np.arange(len(order)), corr_a[idx, order], color="black", lw=1.8, label="Day 1")
        ax.plot(np.arange(len(order)), corr_b[idx, order], color="#dc2626", lw=1.6, label="Day 2")
        ax.set_title("d  Correlation with reference population", loc="left", fontweight="bold")
        ax.set_xlabel("Reference unit rank")
        ax.set_ylabel("Correlation")
        ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    ref_pos = [row["positive_scores"] for row in ref_within if row["positive_scores"]]
    ref_neg = [row["negative_scores"] for row in ref_within if row["negative_scores"]]
    ref_across_pos = [row["positive_scores"] for row in ref_across if row["positive_scores"]]
    if ref_pos and ref_neg and ref_across_pos:
        bins = np.linspace(-1.0, 1.0, 45)
        ax.hist(np.concatenate(ref_neg), bins=bins, density=True, alpha=0.55, color="#2563eb", label="Within-day different")
        ax.hist(np.concatenate(ref_pos), bins=bins, density=True, alpha=0.45, color="#111827", label="Within-day same")
        ax.hist(np.concatenate(ref_across_pos), bins=bins, density=True, alpha=0.55, color="#dc2626", label="Across-day matches")
        ax.set_xlabel("Correlation")
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Reference-population\ndistributions unavailable", ha="center", va="center")
    ax.set_title("e  Reference-population fingerprint", loc="left", fontweight="bold")

    ax = axes[1, 2]
    ref_within_mouse = mouse_auc(ref_within)
    ref_across_mouse = mouse_auc(ref_across)
    if len(ref_across_mouse):
        x = np.arange(len(ref_across_mouse.index))
        ax.bar(x - 0.17, ref_within_mouse.reindex(ref_across_mouse.index), width=0.34, color="#2563eb", label="Within day")
        ax.bar(x + 0.17, ref_across_mouse.values, width=0.34, color="#dc2626", label="Across days")
        ax.set_xticks(x)
        ax.set_xticklabels(ref_across_mouse.index)
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Reference-population AUCs\nunavailable", ha="center", va="center")
    ax.set_title("f  Reference-population AUC by mouse", loc="left", fontweight="bold")

    fig.suptitle("Figure 4-like: validation with stable functional properties", fontsize=16)
    out_path = report_dir / "figures" / "figure4_like_functional_validation.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_tracking_figure(report_dir: Path, multi_root: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import build_paper_structured_report as paper_report

    runs = []
    for mouse_dir in sorted(p for p in multi_root.iterdir() if p.is_dir() and (p / "classic" / "MatchTable.csv").exists()):
        summary = json.loads((mouse_dir / "run_summary.json").read_text())
        runs.append({"mouse": mouse_dir.name, "session_labels": summary["session_labels"], "classic_dir": mouse_dir / "classic"})

    class RunProxy:
        def __init__(self, mouse: str, session_labels: list[str], classic_dir: Path) -> None:
            self.mouse = mouse
            self.session_labels = session_labels
            self.classic_dir = classic_dir

    proxies = [RunProxy(row["mouse"], row["session_labels"], row["classic_dir"]) for row in runs]
    gap_rows = []
    algo_rows = []
    for proxy in proxies:
        gap_rows.extend(paper_report.compute_tracking_gap_rows(proxy, "uid_default"))
        algo_rows.extend(paper_report.summarize_algorithms(proxy))
    gap_df = pd.DataFrame(gap_rows)
    algo_df = pd.DataFrame(algo_rows)

    example = next(proxy for proxy in proxies if proxy.mouse == "AL032")
    prob = np.load(example.classic_dir / "MatchProb.npy")
    binary = (prob > MATCH_THRESHOLD).astype(float)
    clus_info = pickle.load(open(example.classic_dir / "ClusInfo.pickle", "rb"))
    unit_df = paper_report.load_unit_table(example.classic_dir)
    presence, _ = paper_report.uid_presence_matrix(unit_df, "uid_default")
    presence, _ = paper_report.sort_presence_matrix(presence, list(range(presence.shape[0])), limit=140)
    labels = paper_report.short_session_labels(example.session_labels)

    fig = plt.figure(figsize=(16.0, 10.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.0, 1.0], width_ratios=[1.15, 1.0, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(binary, cmap="Blues", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    for boundary in np.asarray(clus_info["session_switch"]).astype(int)[1:-1]:
        ax.axvline(boundary - 0.5, color="white", lw=1.0)
        ax.axhline(boundary - 0.5, color="white", lw=1.0)
    ax.set_title("a  Pairs with P(match) > 0.5", loc="left", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_title("b  Liberal / default / conservative", loc="left", fontweight="bold")
    x = [0.14, 0.50, 0.86]
    y_rows = [0.78, 0.48, 0.18]
    for y0, name, color in [(0.78, "Liberal", "#6b7280"), (0.48, "Default", "#16a34a"), (0.18, "Conservative", "#dc2626")]:
        ax.text(0.02, y0, name, fontsize=11, va="center")
        ax.scatter(x, [y0] * 3, s=50, color="black")
        ax.plot([x[0], x[1]], [y0, y0], color=color, lw=2)
        ax.plot([x[1], x[2]], [y0, y0], color=color, lw=2)
    ax.scatter([x[1]], [0.36], s=50, color="#2563eb")
    ax.plot([x[1], x[1]], [0.48, 0.36], color="#16a34a", lw=2)
    ax.add_patch(patches.FancyArrowPatch((x[1], 0.18), (x[2], 0.18), connectionstyle="arc3,rad=-0.3", arrowstyle="-", lw=1.8, color="#dc2626"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[0, 2])
    uid_groups = unit_df.groupby("uid_default")["recording_index"].apply(list)
    palette = plt.cm.tab20(np.linspace(0, 1, min(12, len(uid_groups))))
    for idx, (_, recs) in enumerate(uid_groups.sort_values(key=lambda s: s.apply(lambda vals: (-len(vals), vals))).head(12).items()):
        y = np.full(len(recs), idx)
        ax.scatter(recs, y, s=36, color=palette[idx])
        ax.plot(recs, y, lw=1.2, color=palette[idx], alpha=0.8)
    ax.set_title("c  Example tracked units", loc="left", fontweight="bold")
    ax.set_xlabel("Recording")
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1:, 0:2])
    if presence.size:
        ax.imshow(presence, cmap="Greys", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks([])
        ax.set_title("d  Presence of unique neurons", loc="left", fontweight="bold")
        ax.set_xlabel("Recording")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No multi-recording groups available", ha="center", va="center")

    gap_summary = gap_df.groupby("gap")["p_track"].agg(["mean", "count", "std"]).reset_index().rename(columns={"mean": "p_track_mean", "std": "p_track_std"})
    gap_summary["sem"] = gap_summary["p_track_std"] / np.sqrt(gap_summary["count"].clip(lower=1))
    ax = fig.add_subplot(gs[1, 2])
    ax.errorbar(gap_summary["gap"], gap_summary["p_track_mean"], yerr=gap_summary["sem"].fillna(0.0), marker="o", lw=2, capsize=3, color="#111827")
    ax.axvline(0, color="#9ca3af", ls="--", lw=1.0)
    ax.set_ylim(0, 1.02)
    ax.set_title("e  P(track) vs gap", loc="left", fontweight="bold")
    ax.set_xlabel("Recording gap")
    ax.set_ylabel("P(track)")

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(gap_summary["gap"], gap_summary["count"], marker="o", lw=2, color="#2563eb")
    ax.set_title("f  Dataset count vs gap", loc="left", fontweight="bold")
    ax.set_xlabel("Recording gap")
    ax.set_ylabel("Comparisons")

    fig.suptitle("Figure 5-like: tracking neurons over many recordings", fontsize=16)
    out_path = report_dir / "figures" / "figure5_like_tracking_overview.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path, algo_df, gap_df


def write_markdown(report_dir: Path, figures: dict[str, Path], tables: dict[str, Path], summary_rows: list[tuple[str, str]]) -> Path:
    lines = [
        "# Tracking neurons across days with high-density probes",
        "",
        "## Local Full-Data UnitMatch Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "### Abstract",
        "",
        "This package consolidates the full bundled UnitMatch data into a clean paper-style report.",
        "It emphasizes the requested figure families for classifier construction, functional validation, and multi-recording tracking.",
        "The bundled checkout supports waveform matching, ISI validation, and a spike-time-derived reference-population analog, but not natural-image-response validation.",
        "",
        "### Key Summary",
        "",
    ]
    for label, value in summary_rows:
        lines.append(f"- {label}: {value}")
    lines.extend([
        "",
        "### Results",
        "",
        "#### Computing similarity scores and setting up the classifier",
        "",
        f"![classifier](figures/{figures['classifier'].name})",
        "",
        "#### Validation with stable functional properties",
        "",
        f"![functional](figures/{figures['functional'].name})",
        "",
        "#### Tracking neurons over many recordings",
        "",
        f"![tracking](figures/{figures['tracking'].name})",
        "",
        "### Data Files",
        "",
    ])
    for label, path in tables.items():
        lines.append(f"- `{label}`: `{path.name}`")
    path = report_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


def write_html(report_dir: Path, figures: dict[str, Path], tables: dict[str, Path], summary_rows: list[tuple[str, str]]) -> Path:
    summary_html = "\n".join(f"<li><strong>{label}:</strong> {value}</li>" for label, value in summary_rows)
    table_html = "\n".join(f"<li><code>{label}</code>: <a href=\"../data/{path.name}\">{path.name}</a></li>" for label, path in tables.items())
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tracking neurons across days with high-density probes | local report</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      max-width: 920px;
      margin: 36px auto;
      padding: 0 24px;
      line-height: 1.65;
      color: #111827;
      background: #fafaf9;
    }}
    .kicker {{ text-transform: uppercase; letter-spacing: 0.12em; font-size: 11px; color: #6b7280; margin-bottom: 8px; }}
    h1 {{ font-size: 36px; line-height: 1.08; margin: 0 0 10px; }}
    h2 {{ font-size: 22px; margin-top: 34px; border-top: 1px solid #d6d3d1; padding-top: 18px; }}
    h3 {{ font-size: 18px; margin-top: 22px; }}
    p, li {{ font-size: 15px; }}
    .meta {{ color: #4b5563; font-size: 14px; margin-bottom: 20px; }}
    .abstract {{ border-left: 3px solid #111827; padding-left: 16px; margin: 18px 0 28px; }}
    figure {{ margin: 28px 0 34px; }}
    img {{ width: 100%; height: auto; border: 1px solid #d6d3d1; background: white; }}
    figcaption {{ font-size: 13px; color: #374151; margin-top: 8px; }}
  </style>
</head>
<body>
  <div class="kicker">Nature Methods Style Local Reproduction</div>
  <h1>Tracking neurons across days with high-density probes</h1>
  <div class="meta">Local full-data UnitMatch report generated {date.today().isoformat()}</div>
  <div class="abstract"><strong>Abstract.</strong> This local report packages a full-data UnitMatch run across the bundled recordings in a paper-like format. It focuses on the classifier-construction, functional-validation, and multi-recording tracking results requested for review. The local bundle supports waveform-based matching, ISI validation, and a spike-time-derived reference-population analog, but not the natural-image-response validation dataset used in the paper.</div>
  <h2>Key Summary</h2>
  <ul>{summary_html}</ul>
  <h2>Results</h2>
  <h3>Computing similarity scores and setting up the classifier</h3>
  <figure>
    <img src="figures/{figures['classifier'].name}" alt="classifier overview">
    <figcaption>Figure 3-like local analog built from the AL032 consecutive-day example pair.</figcaption>
  </figure>
  <h3>Validation with stable functional properties</h3>
  <figure>
    <img src="figures/{figures['functional'].name}" alt="functional validation">
    <figcaption>Figure 4-like local analog using the functional fingerprints supported by the bundled spike-time data.</figcaption>
  </figure>
  <h3>Tracking neurons over many recordings</h3>
  <figure>
    <img src="figures/{figures['tracking'].name}" alt="tracking overview">
    <figcaption>Figure 5-like local analog using the saved multi-session classic runs across the bundled mice.</figcaption>
  </figure>
  <h2>Data Files</h2>
  <ul>{table_html}</ul>
</body>
</html>
"""
    path = report_dir / "report.html"
    path.write_text(html)
    return path


def add_text_page(pdf, title: str, paragraphs: list[str], summary_rows: list[tuple[str, str]] | None = None) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.05, 0.95, title, fontsize=24, fontweight="bold", va="top")
    y = 0.87
    for paragraph in paragraphs:
        wrapped = "\n".join(textwrap.wrap(paragraph, width=94))
        ax.text(0.05, y, wrapped, fontsize=12, va="top")
        y -= 0.1 + 0.018 * wrapped.count("\n")
    if summary_rows:
        table = ax.table(cellText=[[label, value] for label, value in summary_rows], colLabels=["Metric", "Value"], colLoc="left", cellLoc="left", bbox=[0.05, 0.08, 0.90, 0.36])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.35)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf, title: str, image_path: Path, caption: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.82])
    ax.axis("off")
    ax.set_title(title, fontsize=18, pad=12)
    ax.imshow(mpimg.imread(image_path))
    fig.text(0.05, 0.04, "\n".join(textwrap.wrap(caption, width=110)), fontsize=10)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf(report_dir: Path, figures: dict[str, Path], summary_rows: list[tuple[str, str]]) -> Path:
    from matplotlib.backends.backend_pdf import PdfPages

    path = report_dir / "report.pdf"
    with PdfPages(path) as pdf:
        add_text_page(
            pdf,
            "Tracking neurons across days with high-density probes",
            [
                "This is a local full-data UnitMatch report assembled in a paper-like format from the bundled recordings.",
                "It consolidates the existing exploratory outputs into a single clean package and focuses on the figure families requested for review: classifier construction, functional validation, and tracking over many recordings.",
                "Natural-image-response validation from the published paper is not reproducible from the files present in this checkout, so the functional-validation section replaces that portion with a spike-time-derived reference-population analog.",
            ],
            summary_rows,
        )
        add_image_page(pdf, "Figure 3-like: computing similarity scores and setting up the classifier", figures["classifier"], "Local analog built from the AL032 consecutive-day pair.")
        add_image_page(pdf, "Figure 4-like: validation with stable functional properties", figures["functional"], "Local analog using ISI histograms and a spike-time-derived reference-population fingerprint.")
        add_image_page(pdf, "Figure 5-like: tracking neurons over many recordings", figures["tracking"], "Local analog using the saved multi-session classic runs across the bundled mice.")
    return path


def build_summary_rows(full_summary: dict[str, object], within_df: pd.DataFrame, pair_df: pd.DataFrame, isi_across: list[dict[str, object]], ref_across: list[dict[str, object]]) -> list[tuple[str, str]]:
    isi_df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in isi_across])
    ref_df = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in ref_across])
    return [
        ("Sessions processed", str(int(full_summary["session_count"]))),
        ("Pairs processed", str(int(full_summary["pair_count"]))),
        ("Mice", ", ".join(sorted(within_df["mouse"].unique()))),
        ("Acute mean tracked fraction", f"{100.0 * pair_df.loc[pair_df['pair_type'] == 'acute', 'tracked_fraction_of_smaller_session'].mean():.1f}%"),
        ("Chronic mean tracked fraction", f"{100.0 * pair_df.loc[pair_df['pair_type'] == 'chronic', 'tracked_fraction_of_smaller_session'].mean():.1f}%"),
        ("Within-day false-positive median", f"{float(within_df['false_positive_percent'].median()):.2f}%"),
        ("Within-day false-negative median", f"{float(within_df['false_negative_rate_percent'].median()):.2f}%"),
        ("ISI across-day AUC mean", f"{float(isi_df['auc'].dropna().mean()):.3f}" if not isi_df.empty else "n/a"),
        ("Reference-population across-day AUC mean", f"{float(ref_df['auc'].dropna().mean()):.3f}" if not ref_df.empty else "n/a"),
    ]


def main() -> int:
    args = parse_args()
    root = repo_root()
    configure_runtime(root)

    package_dir = root / "local_runs_1"
    package_dir.mkdir(parents=True, exist_ok=True)
    if args.force_refresh:
        reset_outputs(package_dir)

    source = ensure_source_outputs(root, args.force_source)
    full_root = source.full_root
    multi_root = source.multi_root
    full_summary = json.loads((full_root / "summary.json").read_text())
    within_df = pd.read_csv(full_root / "within_day_metrics.csv")
    pair_df = add_gap_columns(pd.read_csv(full_root / "pair_tracking.csv"))

    report_dir = package_dir / "report"
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("Building ISI tables...", flush=True)
    isi_within, isi_across = build_isi_rows(within_df, pair_df)
    print("Building reference-population tables...", flush=True)
    ref_within, ref_across = build_refpop_rows(within_df, pair_df)

    tables = write_compact_tables(package_dir, full_root, within_df, pair_df, isi_within, isi_across, ref_within, ref_across)
    print("Rendering tracking figure...", flush=True)
    tracking_path, algo_df, gap_df = build_tracking_figure(report_dir, multi_root)
    algo_path = package_dir / "data" / "tracking_algorithm_summary.csv"
    gap_path = package_dir / "data" / "tracking_gap_table.csv"
    algo_df.to_csv(algo_path, index=False)
    gap_df.to_csv(gap_path, index=False)
    tables["tracking_algorithm_summary"] = algo_path
    tables["tracking_gap_table"] = gap_path

    figures = {
        "classifier": build_classifier_figure(report_dir, pair_df),
        "functional": build_functional_figure(report_dir, within_df, pair_df, isi_within, isi_across, ref_within, ref_across),
        "tracking": tracking_path,
    }
    summary_rows = build_summary_rows(full_summary, within_df, pair_df, isi_across, ref_across)

    print("Writing report files...", flush=True)
    report_md = write_markdown(report_dir, figures, tables, summary_rows)
    report_html = write_html(report_dir, figures, tables, summary_rows)
    report_pdf = write_pdf(report_dir, figures, summary_rows)

    manifest = {
        "generated_on": date.today().isoformat(),
        "source_roots": {"full_data_classic": str(full_root), "multi_session_by_mouse": str(multi_root)},
        "report_markdown": str(report_md),
        "report_html": str(report_html),
        "report_pdf": str(report_pdf),
        "figures": {key: str(path) for key, path in figures.items()},
        "tables": {key: str(path) for key, path in tables.items()},
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
