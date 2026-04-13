#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from _pipeline_utils import dump_json, now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/configs/unitmatch_dredge_run_config.json",
    )
    parser.add_argument(
        "--offsets-csv",
        default="/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/outputs/corrected_spikes/session_alignment_offsets.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    corrected_spikes_root = Path(config["corrected_spikes_root"])
    offsets = pd.read_csv(args.offsets_csv)
    offset_map = {str(row.session_name): float(row.session_alignment_offset_um) for row in offsets.itertuples(index=False)}

    merged_output = corrected_spikes_root / "al032_tracked_spikes_all_sessions_dredge_session_aligned.csv.gz"
    first_merged = True
    session_summaries = []
    with gzip.open(merged_output, "wt") as merged_handle:
        for session_name in config["session_names"]:
            input_csv_gz = corrected_spikes_root / session_name / f"{session_name}_tracked_spikes.csv.gz"
            output_csv_gz = corrected_spikes_root / session_name / f"{session_name}_tracked_spikes_session_aligned.csv.gz"
            offset = float(offset_map.get(session_name, 0.0))
            first = True
            total_rows = 0
            with gzip.open(output_csv_gz, "wt") as handle:
                for chunk in pd.read_csv(input_csv_gz, compression="gzip", chunksize=250_000, low_memory=False):
                    chunk["session_alignment_offset_um"] = offset
                    chunk["y_um_dredge_session_aligned"] = chunk["y_um_dredge"] + offset
                    chunk.to_csv(handle, index=False, header=first)
                    chunk.to_csv(merged_handle, index=False, header=first_merged)
                    first = False
                    first_merged = False
                    total_rows += int(chunk.shape[0])
            session_summaries.append(
                {
                    "session_name": session_name,
                    "session_alignment_offset_um": offset,
                    "total_rows": total_rows,
                    "output_csv_gz": str(output_csv_gz),
                }
            )

    dump_json(
        corrected_spikes_root / "apply_session_alignment_summary.json",
        {
            "created_at": now_iso(),
            "offsets_csv": str(args.offsets_csv),
            "merged_output_csv_gz": str(merged_output),
            "sessions": session_summaries,
        },
    )


if __name__ == "__main__":
    main()
