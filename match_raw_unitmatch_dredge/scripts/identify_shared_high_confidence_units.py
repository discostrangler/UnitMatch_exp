#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _pipeline_utils import csv_write, dump_json, now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/configs/unitmatch_dredge_run_config.json",
    )
    parser.add_argument("--min-sessions-present", type=int, default=6)
    parser.add_argument("--min-good-tracked-spikes", type=int, default=1000)
    parser.add_argument("--min-mean-probability", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    tracked_root = Path(config["tracked_tables_root"])
    coverage = pd.read_csv(tracked_root / "tracked_unit_coverage_summary.csv")
    summary = pd.read_csv(tracked_root / "tracked_unit_summary.csv")

    merged = coverage.merge(
        summary[
            [
                "tracked_unit_id",
                "sessions_present",
                "cluster_ids_by_session",
                "conflict_free_validity_flag",
                "mean_cross_session_probability",
                "max_cross_session_probability",
            ]
        ],
        on="tracked_unit_id",
        suffixes=("", "_summary"),
        how="left",
    )

    merged["alignment_eligible"] = (
        merged["conflict_free_validity_flag"].fillna(False).astype(bool)
        & (merged["n_sessions_present"] >= int(args.min_sessions_present))
        & (merged["min_good_tracked_spikes"] >= int(args.min_good_tracked_spikes))
        & (merged["mean_cross_session_probability"] >= float(args.min_mean_probability))
    )

    eligible = merged.loc[merged["alignment_eligible"]].copy()
    eligible = eligible.sort_values(
        ["n_sessions_present", "min_good_tracked_spikes", "mean_cross_session_probability", "tracked_unit_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    output_csv = tracked_root / "high_confidence_shared_units_for_alignment.csv"
    csv_write(eligible, output_csv)
    dump_json(
        tracked_root / "high_confidence_shared_units_for_alignment_summary.json",
        {
            "created_at": now_iso(),
            "min_sessions_present": int(args.min_sessions_present),
            "min_good_tracked_spikes": int(args.min_good_tracked_spikes),
            "min_mean_probability": float(args.min_mean_probability),
            "output_csv": str(output_csv),
            "n_total_tracked_units": int(merged.shape[0]),
            "n_alignment_eligible_units": int(eligible.shape[0]),
            "tracked_unit_ids": [int(v) for v in eligible["tracked_unit_id"].tolist()],
        },
    )


if __name__ == "__main__":
    main()
