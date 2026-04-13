#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from _pipeline_utils import dump_json, now_iso


ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge")
UPSTREAM_ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch")
INPUT_MANIFEST_CSV = UPSTREAM_ROOT / "manifests" / "unitmatch_input_manifest.csv"
OUTPUT_CONFIG_JSON = ROOT / "configs" / "unitmatch_dredge_run_config.json"


def main() -> None:
    df = pd.read_csv(INPUT_MANIFEST_CSV).sort_values("session_order")
    invalid = df.loc[~df["unitmatch_ready"], ["session_name", "issues"]]
    if not invalid.empty:
        raise SystemExit(f"Cannot prepare DREDge UnitMatch run; invalid sessions present: {invalid.to_dict(orient='records')}")

    baseline_config = json.loads((UPSTREAM_ROOT / "configs" / "unitmatch_run_config.json").read_text())
    session_names = df["session_name"].tolist()
    config = {
        "created_at": now_iso(),
        "mouse": "AL032",
        "run_name": "al032_dredge_12session_unitmatchpy",
        "unitmatch_root": baseline_config["unitmatch_root"],
        "baseline_config_path": str(UPSTREAM_ROOT / "configs" / "unitmatch_run_config.json"),
        "baseline_output_root": str(UPSTREAM_ROOT / "outputs" / "unitmatch_raw_12session"),
        "input_manifest_csv": str(INPUT_MANIFEST_CSV),
        "dredge_manifest_csv": str(ROOT / "manifests" / "al032_dredge_input_manifest.csv"),
        "session_names": session_names,
        "session_order": session_names,
        "ks_dirs": df["ks_path"].tolist(),
        "custom_raw_waveform_paths": df["raw_waveforms_path"].tolist(),
        "unit_label_paths": df["unit_label_path"].tolist(),
        "output_root": str(ROOT / "outputs" / "unitmatch_dredge_12session"),
        "tracked_tables_root": str(ROOT / "outputs" / "tracked_tables"),
        "corrected_spikes_root": str(ROOT / "outputs" / "corrected_spikes"),
        "attached_spikes_root": str(ROOT / "outputs" / "corrected_spikes"),
        "figures_root": str(ROOT / "outputs" / "figures"),
        "logs_root": str(ROOT / "logs"),
        "dredge_per_session_root": str(ROOT / "outputs" / "dredge_per_session"),
        "corrected_unit_features_csv": str(ROOT / "outputs" / "corrected_unit_features" / "al032_corrected_unit_feature_table.csv"),
        "uncorrected_unit_features_csv": str(ROOT / "outputs" / "corrected_unit_features" / "al032_uncorrected_unit_feature_table.csv"),
        "geometry_shift_by_unit_csv": str(ROOT / "outputs" / "corrected_unit_features" / "al032_geometry_shift_by_unit.csv"),
        "tracked_id_mode": baseline_config.get("tracked_id_mode", "intermediate"),
        "localization_method_for_attachment": "monopolar_triangulation_dredge_corrected",
        "unitmatch_parameters": baseline_config["unitmatch_parameters"],
        "dredge_geometry_override": {
            "enabled": True,
            "override_fields": ["avg_centroid_y", "avg_waveform_per_tp_y"],
            "shift_source": "median_y_shift_um",
        },
    }
    dump_json(OUTPUT_CONFIG_JSON, config)


if __name__ == "__main__":
    main()
