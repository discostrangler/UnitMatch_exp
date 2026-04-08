#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _pipeline_utils import dump_json, now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/manifests/unitmatch_input_manifest.csv",
    )
    parser.add_argument(
        "--output-config-json",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/configs/unitmatch_run_config.json",
    )
    parser.add_argument(
        "--output-root",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/unitmatch_raw_12session",
    )
    parser.add_argument(
        "--localization-root",
        default="/scratch/am15577/UnitMatch/spike_rosters_raw/al032_rosters_monopolar_triangulation",
    )
    parser.add_argument(
        "--unitmatch-root",
        default="/scratch/am15577/UnitMatch/UnitMatch-main-2",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_manifest_csv).sort_values("session_order")
    invalid = df.loc[~df["unitmatch_ready"], ["session_name", "issues"]]
    if not invalid.empty:
        raise SystemExit(f"Cannot prepare run; invalid sessions present: {invalid.to_dict(orient='records')}")

    session_names = df["session_name"].tolist()
    ks_dirs = df["ks_path"].tolist()
    raw_waveform_paths = df["raw_waveforms_path"].tolist()
    unit_label_paths = df["unit_label_path"].tolist()

    config = {
        "created_at": now_iso(),
        "mouse": "AL032",
        "run_name": "al032_raw_12session_unitmatchpy",
        "unitmatch_root": args.unitmatch_root,
        "input_manifest_csv": args.input_manifest_csv,
        "session_names": session_names,
        "session_order": session_names,
        "ks_dirs": ks_dirs,
        "custom_raw_waveform_paths": raw_waveform_paths,
        "unit_label_paths": unit_label_paths,
        "output_root": args.output_root,
        "tracked_tables_root": "/scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/tracked_tables",
        "attached_spikes_root": "/scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/attached_spikes",
        "figures_root": "/scratch/am15577/UnitMatch/match_raw_unitmatch/outputs/figures",
        "logs_root": "/scratch/am15577/UnitMatch/match_raw_unitmatch/logs",
        "localization_method_for_attachment": "monopolar_triangulation",
        "localization_root_for_attachment": args.localization_root,
        "tracked_id_mode": "intermediate",
        "unitmatch_parameters": {
            "good_units_only": True,
            "match_threshold": 0.5,
            "use_data_driven_prob_thrs": False,
            "channel_radius": 150,
            "max_dist": 100,
            "neighbour_dist": 50,
            "min_new_shank_distance": 100,
            "units_per_shank_thrs": 15,
            "curve_fit_maxfev": 10000,
            "niter": 2,
            "save_match_table": True,
        },
    }
    dump_json(Path(args.output_config_json), config)

    plot_config = {
        "created_at": now_iso(),
        "mouse": "AL032",
        "preferred_localization_method": "monopolar_triangulation",
        "background_source": "all good localized spikes",
        "overlay_source": "selected tracked units",
        "default_gap_s": 60.0,
    }
    dump_json(Path("/scratch/am15577/UnitMatch/match_raw_unitmatch/configs/plot_config.json"), plot_config)


if __name__ == "__main__":
    main()
