#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from _pipeline_utils import dump_json, now_iso


def write_chunked_join(
    input_csv_gz: Path,
    mapping_df: pd.DataFrame,
    session_name: str,
    output_csv_gz: Path,
) -> dict:
    output_csv_gz.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    attached_rows = 0
    good_attached_rows = 0
    first = True
    with gzip.open(output_csv_gz, "wt") as handle:
        for chunk in pd.read_csv(input_csv_gz, chunksize=250_000):
            chunk["session_name"] = session_name
            merged = chunk.merge(mapping_df, on="cluster_id", how="left")
            merged.to_csv(handle, index=False, header=first)
            first = False
            total_rows += int(merged.shape[0])
            attached_mask = merged["tracked_unit_id"].notna()
            attached_rows += int(attached_mask.sum())
            if "is_good_cluster" in merged.columns:
                good_attached_rows += int((attached_mask & merged["is_good_cluster"].fillna(False)).sum())
    return {
        "session_name": session_name,
        "total_rows": total_rows,
        "attached_rows": attached_rows,
        "good_attached_rows": good_attached_rows,
        "output_csv_gz": str(output_csv_gz),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch/configs/unitmatch_run_config.json",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    mapping = pd.read_csv(Path(config["tracked_tables_root"]) / "cluster_to_tracked_unit.csv")
    localization_root = Path(config["localization_root_for_attachment"])
    attached_root = Path(config["attached_spikes_root"])
    attached_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    merged_output = attached_root / "al032_tracked_spikes_all_sessions.csv.gz"
    first_merged = True
    with gzip.open(merged_output, "wt") as merged_handle:
        for session_name in config["session_names"]:
            session_mapping = mapping.loc[mapping["session_name"] == session_name, ["cluster_id", "tracked_unit_id", "conflict_flag"]].copy()
            localized_path = localization_root / session_name / f"{session_name}_localized_spike_table.csv.gz"
            output_path = attached_root / session_name / f"{session_name}_tracked_spikes.csv.gz"
            summary = write_chunked_join(localized_path, session_mapping, session_name, output_path)
            all_summaries.append(summary)
            for chunk in pd.read_csv(output_path, compression="gzip", chunksize=250_000):
                chunk.to_csv(merged_handle, index=False, header=first_merged)
                first_merged = False

    dump_json(
        attached_root / "attach_tracked_ids_summary.json",
        {
            "created_at": now_iso(),
            "merged_output_csv_gz": str(merged_output),
            "sessions": all_summaries,
        },
    )


if __name__ == "__main__":
    main()
