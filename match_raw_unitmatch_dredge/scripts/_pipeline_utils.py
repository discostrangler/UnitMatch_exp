from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AL032_SESSION_RE = re.compile(r"^AL032_(\d{4}-\d{2}-\d{2})$")
RAW_WAVEFORM_RE = re.compile(r"^Unit(\d+)_RawSpikes\.npy$")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_session_date(session_name: str) -> str:
    match = AL032_SESSION_RE.match(session_name)
    if not match:
        raise ValueError(f"Unexpected AL032 session name: {session_name}")
    return match.group(1)


def discover_al032_sessions(raw_root: Path) -> list[Path]:
    sessions = []
    for path in raw_root.iterdir():
        if path.is_dir() and AL032_SESSION_RE.match(path.name):
            sessions.append(path)
    return sorted(sessions, key=lambda p: parse_session_date(p.name))


def parse_params_py(path: Path) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if not path.exists():
        return params
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            params[key] = ast.literal_eval(value)
        except Exception:
            params[key] = value.strip("'\"")
    return params


def parse_spikeglx_meta(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not path.exists():
        return meta
    for raw_line in path.read_text(errors="ignore").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def find_raw_waveforms_path(ks_path: Path) -> Path | None:
    return find_first_existing(
        [
            ks_path / "RawWaveforms",
            ks_path / "qMetrics" / "RawWaveforms",
            ks_path / "bombcell" / "RawWaveforms",
        ]
    )


def read_cluster_group(cluster_group_path: Path) -> pd.DataFrame:
    if not cluster_group_path.exists():
        return pd.DataFrame(columns=["cluster_id", "group"])
    df = pd.read_csv(cluster_group_path, sep="\t")
    normalized = {str(col).strip().lower(): col for col in df.columns}
    cluster_col = normalized.get("cluster_id")
    group_col = normalized.get("group")
    if cluster_col is None or group_col is None:
        raise ValueError(f"Unexpected cluster_group.tsv columns in {cluster_group_path}: {df.columns.tolist()}")
    out = df[[cluster_col, group_col]].copy()
    out.columns = ["cluster_id", "group"]
    out["cluster_id"] = out["cluster_id"].astype(int)
    out["group"] = out["group"].astype(str)
    return out


def get_good_cluster_ids(cluster_group_path: Path) -> list[int]:
    df = read_cluster_group(cluster_group_path)
    if df.empty:
        return []
    good = df.loc[df["group"].str.lower() == "good", "cluster_id"].astype(int).tolist()
    return sorted(good)


def get_raw_waveform_unit_ids(raw_waveforms_path: Path) -> list[int]:
    if not raw_waveforms_path.exists():
        return []
    unit_ids = []
    for file_path in raw_waveforms_path.iterdir():
        match = RAW_WAVEFORM_RE.match(file_path.name)
        if match:
            unit_ids.append(int(match.group(1)))
    return sorted(unit_ids)


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    return value


def read_session_validation_report(localization_root: Path, session_name: str) -> dict[str, Any]:
    report_path = localization_root / session_name / f"{session_name}_validation_report.json"
    report = load_json(report_path)
    return report if isinstance(report, dict) else {}


def count_unique_clusters(spike_clusters_path: Path) -> int:
    clusters = np.load(spike_clusters_path, mmap_mode="r")
    return int(np.unique(np.asarray(clusters)).size)


def csv_write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
