#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline_utils import csv_write, dump_json, now_iso


TRACKED_MODE_TO_COLUMNS = {
    "intermediate": ("UID int 1", "UM UID int 2"),
    "liberal": ("UID Liberal 1", "UID Liberal 2"),
    "conservative": ("UID Conservative 1", "UID Conservative 2"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/configs/unitmatch_run_config.json",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    output_root = Path(config["output_root"])
    tracked_root = Path(config["tracked_tables_root"])
    tracked_root.mkdir(parents=True, exist_ok=True)

    match_table = pd.read_csv(output_root / "MatchTable.csv")
    with open(output_root / "ClusInfo.pickle", "rb") as handle:
        clus_info = pickle.load(handle)

    session_names = config["session_names"]
    tracked_mode = config.get("tracked_id_mode", "intermediate")
    uid_col_1, uid_col_2 = TRACKED_MODE_TO_COLUMNS[tracked_mode]

    diag = match_table.loc[
        (match_table["ID1"] == match_table["ID2"])
        & (match_table["RecSes 1"] == match_table["RecSes 2"])
    ].copy()
    diag["session_index"] = diag["RecSes 1"].astype(int) - 1
    diag["session_name"] = diag["session_index"].map(lambda idx: session_names[idx])
    diag["cluster_id"] = diag["ID1"].astype(int)
    diag["tracked_unit_id"] = diag[uid_col_1].astype(int)
    diag["tracked_unit_id_liberal"] = diag["UID Liberal 1"].astype(int)
    diag["tracked_unit_id_intermediate"] = diag["UID int 1"].astype(int)
    diag["tracked_unit_id_conservative"] = diag["UID Conservative 1"].astype(int)
    diag["self_match_probability"] = diag["UM Probabilities"].astype(float)
    diag["self_total_score"] = diag["TotalScore"].astype(float)

    cross = match_table.loc[
        (match_table["RecSes 1"] != match_table["RecSes 2"])
        & (match_table[uid_col_1] == match_table[uid_col_2])
    ].copy()

    if cross.empty:
        diag["max_cross_session_probability"] = np.nan
        diag["mean_cross_session_probability"] = np.nan
        diag["cross_session_pair_count"] = 0
    else:
        pair_summary = (
            cross.groupby(["RecSes 1", "ID1"], as_index=False)
            .agg(
                max_cross_session_probability=("UM Probabilities", "max"),
                mean_cross_session_probability=("UM Probabilities", "mean"),
                cross_session_pair_count=("UM Probabilities", "size"),
            )
            .rename(columns={"RecSes 1": "session_index_one_based", "ID1": "cluster_id"})
        )
        pair_summary["session_index"] = pair_summary["session_index_one_based"].astype(int) - 1
        pair_summary["session_name"] = pair_summary["session_index"].map(lambda idx: session_names[idx])
        diag = diag.merge(
            pair_summary[
                [
                    "session_name",
                    "cluster_id",
                    "max_cross_session_probability",
                    "mean_cross_session_probability",
                    "cross_session_pair_count",
                ]
            ],
            on=["session_name", "cluster_id"],
            how="left",
        )
        diag["cross_session_pair_count"] = diag["cross_session_pair_count"].fillna(0).astype(int)

    counts = diag.groupby(["tracked_unit_id", "session_name"]).size().rename("session_cluster_count").reset_index()
    diag = diag.merge(counts, on=["tracked_unit_id", "session_name"], how="left")
    diag["conflict_flag"] = diag["session_cluster_count"] > 1

    cluster_to_tracked = diag[
        [
            "session_name",
            "session_index",
            "cluster_id",
            "tracked_unit_id",
            "tracked_unit_id_liberal",
            "tracked_unit_id_intermediate",
            "tracked_unit_id_conservative",
            "self_match_probability",
            "self_total_score",
            "max_cross_session_probability",
            "mean_cross_session_probability",
            "cross_session_pair_count",
            "conflict_flag",
        ]
    ].sort_values(["session_index", "cluster_id"])
    csv_write(cluster_to_tracked, tracked_root / "cluster_to_tracked_unit.csv")

    summary_rows = []
    for tracked_unit_id, group in cluster_to_tracked.groupby("tracked_unit_id"):
        sessions_present = group["session_name"].tolist()
        cluster_map = {
            row.session_name: int(row.cluster_id)
            for row in group.itertuples(index=False)
        }
        summary_rows.append(
            {
                "tracked_unit_id": int(tracked_unit_id),
                "sessions_present": json.dumps(sessions_present),
                "n_sessions_present": int(group["session_name"].nunique()),
                "cluster_ids_by_session": json.dumps(cluster_map),
                "max_cross_session_probability": float(group["max_cross_session_probability"].max())
                if group["max_cross_session_probability"].notna().any()
                else np.nan,
                "mean_cross_session_probability": float(group["mean_cross_session_probability"].mean())
                if group["mean_cross_session_probability"].notna().any()
                else np.nan,
                "conflict_free_validity_flag": bool((~group["conflict_flag"]).all()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["n_sessions_present", "tracked_unit_id"], ascending=[False, True])
    csv_write(summary_df, tracked_root / "tracked_unit_summary.csv")

    dump_json(
        tracked_root / "tracked_tables_summary.json",
        {
            "created_at": now_iso(),
            "tracked_id_mode": tracked_mode,
            "cluster_to_tracked_path": str(tracked_root / "cluster_to_tracked_unit.csv"),
            "tracked_summary_path": str(tracked_root / "tracked_unit_summary.csv"),
            "n_cluster_rows": int(cluster_to_tracked.shape[0]),
            "n_tracked_units": int(summary_df.shape[0]),
        },
    )


if __name__ == "__main__":
    main()
