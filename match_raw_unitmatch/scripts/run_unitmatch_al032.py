#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/configs/unitmatch_run_config.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    save_dir = Path(config["output_root"])
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "MPLCONFIGDIR",
        "/scratch/am15577/UnitMatch/match_raw_unitmatch/tmp/matplotlib",
    )

    match_table_path = save_dir / "MatchTable.csv"
    if match_table_path.exists() and not args.force:
        print(f"UnitMatch output already exists at {save_dir}; skipping. Use --force to rerun.")
        return

    unitmatch_root = Path(config["unitmatch_root"])
    sys.path.insert(0, str(unitmatch_root / "UnitMatchPy"))

    import UnitMatchPy.assign_unique_id as aid
    import UnitMatchPy.bayes_functions as bf
    import UnitMatchPy.default_params as default_params
    import UnitMatchPy.overlord as ov
    import UnitMatchPy.save_utils as su
    import UnitMatchPy.utils as util

    param = default_params.get_default_param(config.get("unitmatch_parameters", {}))
    ks_dirs = config["ks_dirs"]
    param["KS_dirs"] = ks_dirs
    param["KSdirs"] = ks_dirs

    wave_paths, unit_label_paths, channel_pos = util.paths_from_KS(
        ks_dirs,
        custom_raw_waveform_paths=config.get("custom_raw_waveform_paths"),
    )
    param = util.get_probe_geometry(channel_pos[0], param)

    waveform, session_id, session_switch, within_session, good_units, param = util.load_good_waveforms(
        wave_paths,
        unit_label_paths,
        param,
        good_units_only=bool(config["unitmatch_parameters"]["good_units_only"]),
    )

    clus_info = {
        "good_units": good_units,
        "session_switch": session_switch,
        "session_id": session_id,
        "original_ids": np.concatenate(good_units),
        "session_names": np.array(config["session_names"], dtype=object),
    }

    print(f"Loaded {param['n_units']} good units across {param['n_sessions']} sessions")
    extracted_wave_properties = ov.extract_parameters(waveform, channel_pos, clus_info, param)
    total_score, candidate_pairs, scores_to_include, predictors = ov.extract_metric_scores(
        extracted_wave_properties,
        session_switch,
        within_session,
        param,
        niter=int(config["unitmatch_parameters"]["niter"]),
    )

    prior_match = 1 - (param["n_expected_matches"] / param["n_units"] ** 2)
    priors = np.array((prior_match, 1 - prior_match))
    labels = candidate_pairs.astype(int)
    cond = np.unique(labels)
    parameter_kernels = bf.get_parameter_kernels(scores_to_include, labels, cond, param, add_one=1)
    probability = bf.apply_naive_bayes(parameter_kernels, priors, predictors, param, cond)
    output_prob_matrix = probability[:, 1].reshape(param["n_units"], param["n_units"])

    match_threshold = float(config["unitmatch_parameters"]["match_threshold"])
    output_threshold = (output_prob_matrix > match_threshold).astype(np.int8)
    matches = np.argwhere(output_threshold == 1)
    UIDs = aid.assign_unique_id(output_prob_matrix, param, clus_info)

    su.save_to_output(
        str(save_dir),
        scores_to_include,
        matches,
        output_prob_matrix,
        extracted_wave_properties["avg_centroid"],
        extracted_wave_properties["avg_waveform"],
        extracted_wave_properties["avg_waveform_per_tp"],
        extracted_wave_properties["max_site"],
        total_score,
        output_threshold,
        clus_info,
        param,
        UIDs=UIDs,
        matches_curated=None,
        save_match_table=bool(config["unitmatch_parameters"]["save_match_table"]),
    )

    run_summary = {
        "config_path": str(config_path),
        "output_root": str(save_dir),
        "n_units": int(param["n_units"]),
        "n_sessions": int(param["n_sessions"]),
        "match_threshold": match_threshold,
        "n_matches_thresholded": int(matches.shape[0]),
    }
    (save_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
