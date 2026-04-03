#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import struct
from collections import Counter
from pathlib import Path


EXPECTED_SPIKE_TABLE_COLUMNS = [
    "source_spike_index",
    "spike_time_samples",
    "spike_time_s",
    "cluster_id",
    "is_good_cluster",
    "peak_channel_index",
    "template_peak_y_um",
    "template_com_y_um",
    "processing_chunk_id",
    "read_window_id",
    "localization_attempted",
    "localization_success",
    "localization_missing",
    "skip_reason",
    "x_um",
    "y_um",
    "amplitude",
]

EXPECTED_GOOD_CLUSTER_SUMMARY_COLUMNS = [
    "cluster_id",
    "total_spike_count",
    "localized_spike_count",
    "localized_x_median_um",
    "localized_y_median_um",
    "localized_y_mean_um",
    "localized_y_std_um",
    "localized_y_min_um",
    "localized_y_max_um",
    "amplitude_median",
    "skipped_spike_count",
    "localized_fraction",
    "primary_template_id",
    "template_peak_y_um",
    "template_com_y_um",
    "peak_channel_index",
    "delta_median_vs_template_peak_um",
    "delta_median_vs_template_com_um",
    "Amplitude",
    "firing_rate_hz",
    "contamination_pct",
    "cluster_metrics_spike_count",
]

EXPECTED_VALIDATION_REPORT_KEYS = [
    "session_name",
    "run_mode",
    "total_spikes",
    "total_good_cluster_spikes",
    "localized_spikes",
    "localized_good_cluster_spikes",
    "skipped_corrupted_spikes",
    "skipped_read_failure_spikes",
    "skipped_invalid_peak_spikes",
    "skipped_other_spikes",
    "unresolved_pipeline_gap_spikes",
    "localized_fraction_overall",
    "localized_fraction_good_clusters",
    "processing_chunk_count",
    "readable_window_count",
    "failed_window_count",
    "plotting_status",
    "plotting_excluded_nonfinite_localized_spikes",
    "plotting_excluded_nonfinite_good_cluster_spikes",
    "join_integrity",
    "corrupted_region_summary_path",
    "selected_cluster_depth_examples",
    "outputs",
]

SKIP_REASON_VOCABULARY = [
    "",
    "corrupted_raw_chunk",
    "read_failure",
    "invalid_peak",
    "other",
    "unresolved_pipeline_gap",
]

REQUIRED_ARTIFACT_SUFFIXES = [
    "localized_spike_table.csv.gz",
    "validation_report.json",
    "good_cluster_summary.csv",
    "exact_depth_raster.png",
]

OPTIONAL_ARTIFACT_SUFFIXES = [
    "corrupted_region_summary.json",
    "localized_depth_histogram.png",
    "cluster_depth_spread.png",
    "example_cluster_scatter.png",
    "notes.md",
    "processing_windows.csv",
    "selected_good_clusters.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and freeze AL032 Phase 4 localization outputs.")
    parser.add_argument(
        "--session-manifest",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_session_manifest.csv"),
        help="Existing per-session manifest produced before Phase 4 completion.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_session_outputs"),
        help="Per-session output root.",
    )
    parser.add_argument(
        "--phase4-manifest-csv",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_phase4_manifest.csv"),
        help="Mouse-level Phase 4 manifest CSV output path.",
    )
    parser.add_argument(
        "--phase4-manifest-json",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_phase4_manifest.json"),
        help="Mouse-level Phase 4 manifest JSON output path.",
    )
    parser.add_argument(
        "--schema-report-json",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_phase4_schema_report.json"),
        help="Schema and quality audit report JSON output path.",
    )
    parser.add_argument(
        "--completion-checklist-md",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_phase4_completion_checklist.md"),
        help="Completion checklist Markdown output path.",
    )
    parser.add_argument(
        "--contract-json",
        type=Path,
        default=Path("/scratch/am15577/UnitMatch/spike_rosters_raw/al032_phase4_contract.json"),
        help="Frozen Phase 4 contract JSON output path.",
    )
    parser.add_argument(
        "--rewrite-validation-reports",
        action="store_true",
        help="Rewrite per-session validation reports to the frozen Phase 4 schema.",
    )
    return parser.parse_args()


def load_session_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(text: str) -> bool:
    return text.strip().lower() == "true"


def parse_float(text: str) -> float:
    value = text.strip()
    if value == "":
        return float("nan")
    return float(value)


def parse_int(text: str) -> int:
    return int(text.strip())


def is_true(text: str) -> bool:
    return text.strip().lower() == "true"


def read_csv_header(path: Path, compressed: bool = False) -> list[str]:
    opener = gzip.open if compressed else open
    with opener(path, "rt", newline="") as f:
        return next(csv.reader(f))


def read_png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        signature = f.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{path} is not a PNG")
        ihdr_length = f.read(4)
        ihdr_type = f.read(4)
        if ihdr_length != b"\x00\x00\x00\r" or ihdr_type != b"IHDR":
            raise ValueError(f"{path} has an invalid PNG header")
        width, height = struct.unpack(">II", f.read(8))
    return width, height


def read_selected_cluster_examples(selected_clusters_path: Path) -> list[dict[str, object]]:
    if not selected_clusters_path.exists():
        return []

    examples: list[dict[str, object]] = []
    with selected_clusters_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(
                {
                    "cluster_id": int(row["cluster_id"]),
                    "localized_spike_count": int(float(row["localized_spike_count"])),
                    "localized_y_median_um": float(row["localized_y_median_um"]),
                    "localized_y_std_um": float(row["localized_y_std_um"]),
                    "localized_y_min_um": float(row["localized_y_min_um"]),
                    "localized_y_max_um": float(row["localized_y_max_um"]),
                }
            )
    return examples


def audit_spike_table(path: Path) -> dict[str, object]:
    header = read_csv_header(path, compressed=True)

    total_spikes = 0
    total_good_cluster_spikes = 0
    localized_spikes = 0
    localized_good_cluster_spikes = 0
    nonfinite_localized_spikes = 0
    nonfinite_good_cluster_spikes = 0
    skip_reason_counts: Counter[str] = Counter()
    skip_reason_vocabulary_observed: set[str] = set()
    source_index_sequence_ok = True
    source_index_expected = 0

    with gzip.open(path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_spikes += 1
            source_index = parse_int(row["source_spike_index"])
            if source_index != source_index_expected:
                source_index_sequence_ok = False
            source_index_expected += 1

            is_good_cluster = is_true(row["is_good_cluster"])
            localization_success = is_true(row["localization_success"])
            skip_reason = row["skip_reason"]

            skip_reason_counts[skip_reason] += 1
            skip_reason_vocabulary_observed.add(skip_reason)

            if is_good_cluster:
                total_good_cluster_spikes += 1

            if localization_success:
                localized_spikes += 1
                if is_good_cluster:
                    localized_good_cluster_spikes += 1

                x_um = parse_float(row["x_um"])
                y_um = parse_float(row["y_um"])
                if not (math.isfinite(x_um) and math.isfinite(y_um)):
                    nonfinite_localized_spikes += 1
                    if is_good_cluster:
                        nonfinite_good_cluster_spikes += 1

    return {
        "header": header,
        "total_spikes": total_spikes,
        "total_good_cluster_spikes": total_good_cluster_spikes,
        "localized_spikes": localized_spikes,
        "localized_good_cluster_spikes": localized_good_cluster_spikes,
        "nonfinite_localized_spikes": nonfinite_localized_spikes,
        "nonfinite_good_cluster_spikes": nonfinite_good_cluster_spikes,
        "skip_reason_counts": dict(skip_reason_counts),
        "skip_reason_vocabulary_observed": sorted(skip_reason_vocabulary_observed),
        "source_index_sequence_ok": source_index_sequence_ok,
    }


def audit_good_cluster_summary(path: Path) -> dict[str, object]:
    header = read_csv_header(path, compressed=False)

    row_count = 0
    localized_row_count = 0
    finite_localized_depth_rows = 0
    nonzero_depth_spread_rows = 0
    depth_order_violations = 0
    depth_min = float("inf")
    depth_max = float("-inf")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            localized_spike_count = int(float(row["localized_spike_count"]))
            if localized_spike_count <= 0:
                continue
            localized_row_count += 1

            depth_median = parse_float(row["localized_y_median_um"])
            depth_std = parse_float(row["localized_y_std_um"])
            depth_row_min = parse_float(row["localized_y_min_um"])
            depth_row_max = parse_float(row["localized_y_max_um"])

            if math.isfinite(depth_median) and math.isfinite(depth_std) and math.isfinite(depth_row_min) and math.isfinite(depth_row_max):
                finite_localized_depth_rows += 1
                depth_min = min(depth_min, depth_row_min)
                depth_max = max(depth_max, depth_row_max)
                if depth_std > 0.0:
                    nonzero_depth_spread_rows += 1
                if not (depth_row_min <= depth_median <= depth_row_max):
                    depth_order_violations += 1

    if depth_min == float("inf"):
        depth_min = float("nan")
    if depth_max == float("-inf"):
        depth_max = float("nan")

    return {
        "header": header,
        "row_count": row_count,
        "localized_row_count": localized_row_count,
        "finite_localized_depth_rows": finite_localized_depth_rows,
        "nonzero_depth_spread_rows": nonzero_depth_spread_rows,
        "depth_order_violations": depth_order_violations,
        "depth_min_um": depth_min,
        "depth_max_um": depth_max,
    }


def audit_processing_windows(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "processing_chunk_count": 0,
            "readable_window_count": 0,
            "failed_window_count": 0,
            "status_counts": {},
        }

    status_counts: Counter[str] = Counter()
    processing_chunk_count = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processing_chunk_count += 1
            status_counts[row["status"]] += 1

    failed_window_count = sum(count for status, count in status_counts.items() if "fail" in status.lower())
    readable_window_count = processing_chunk_count - failed_window_count
    return {
        "processing_chunk_count": processing_chunk_count,
        "readable_window_count": readable_window_count,
        "failed_window_count": failed_window_count,
        "status_counts": dict(status_counts),
    }


def build_validation_report(
    session_name: str,
    existing_report: dict[str, object],
    row: dict[str, str],
    spike_audit: dict[str, object],
    summary_audit: dict[str, object],
    window_audit: dict[str, object],
    corrupted_region_summary_path: Path,
    raster_path: Path,
    histogram_path: Path,
    cluster_spread_path: Path,
    example_scatter_path: Path,
    selected_clusters_path: Path,
    spike_table_path: Path,
    summary_path: Path,
    processing_windows_path: Path,
) -> dict[str, object]:
    skip_reason_counts = spike_audit["skip_reason_counts"]

    skipped_corrupted_spikes = int(skip_reason_counts.get("corrupted_raw_chunk", 0))
    skipped_read_failure_spikes = int(skip_reason_counts.get("read_failure", 0))
    skipped_invalid_peak_spikes = int(skip_reason_counts.get("invalid_peak", 0))
    skipped_other_spikes = int(skip_reason_counts.get("other", 0))
    unresolved_pipeline_gap_spikes = int(skip_reason_counts.get("unresolved_pipeline_gap", 0))

    total_spikes = int(spike_audit["total_spikes"])
    total_good_cluster_spikes = int(spike_audit["total_good_cluster_spikes"])
    localized_spikes = int(spike_audit["localized_spikes"])
    localized_good_cluster_spikes = int(spike_audit["localized_good_cluster_spikes"])

    run_mode = "partial" if parse_bool(row["partial_mode"]) or skipped_corrupted_spikes > 0 or corrupted_region_summary_path.exists() else "exact"
    plotting_status = "complete" if raster_path.exists() else "missing"
    selected_examples = read_selected_cluster_examples(selected_clusters_path)
    if not selected_examples:
        selected_examples = existing_report.get("selected_cluster_depth_examples", [])

    report = {
        "session_name": session_name,
        "run_mode": run_mode,
        "total_spikes": total_spikes,
        "total_good_cluster_spikes": total_good_cluster_spikes,
        "localized_spikes": localized_spikes,
        "localized_good_cluster_spikes": localized_good_cluster_spikes,
        "skipped_corrupted_spikes": skipped_corrupted_spikes,
        "skipped_read_failure_spikes": skipped_read_failure_spikes,
        "skipped_invalid_peak_spikes": skipped_invalid_peak_spikes,
        "skipped_other_spikes": skipped_other_spikes,
        "unresolved_pipeline_gap_spikes": unresolved_pipeline_gap_spikes,
        "localized_fraction_overall": float(localized_spikes / total_spikes) if total_spikes else 0.0,
        "localized_fraction_good_clusters": float(localized_good_cluster_spikes / total_good_cluster_spikes) if total_good_cluster_spikes else 0.0,
        "processing_chunk_count": int(window_audit["processing_chunk_count"] or existing_report.get("processing_chunk_count", 0)),
        "readable_window_count": int(window_audit["readable_window_count"] or existing_report.get("readable_window_count", 0)),
        "failed_window_count": int(window_audit["failed_window_count"]),
        "plotting_status": plotting_status,
        "plotting_excluded_nonfinite_localized_spikes": int(spike_audit["nonfinite_localized_spikes"]),
        "plotting_excluded_nonfinite_good_cluster_spikes": int(spike_audit["nonfinite_good_cluster_spikes"]),
        "join_integrity": {
            "row_count_matches_input": total_spikes == int(row["total_spikes"]),
            "source_spike_index_unique": bool(spike_audit["source_index_sequence_ok"]),
            "localized_plus_skipped_equals_total": localized_spikes
            + skipped_corrupted_spikes
            + skipped_read_failure_spikes
            + skipped_invalid_peak_spikes
            + skipped_other_spikes
            + unresolved_pipeline_gap_spikes
            == total_spikes,
        },
        "corrupted_region_summary_path": str(corrupted_region_summary_path) if corrupted_region_summary_path.exists() else "",
        "selected_cluster_depth_examples": selected_examples,
        "outputs": {
            "spike_table_csv_gz": str(spike_table_path),
            "good_cluster_summary_csv": str(summary_path),
            "selected_clusters_csv": str(selected_clusters_path),
            "processing_windows_csv": str(processing_windows_path),
            "corrupted_region_summary_json": str(corrupted_region_summary_path) if corrupted_region_summary_path.exists() else "",
            "raster_png": str(raster_path),
            "histogram_png": str(histogram_path),
            "cluster_spread_png": str(cluster_spread_path),
            "example_cluster_scatter_png": str(example_scatter_path),
        },
    }
    return report


def build_quality_review(
    report: dict[str, object],
    summary_audit: dict[str, object],
    raster_dimensions: tuple[int, int] | None,
) -> tuple[str, list[str]]:
    warnings: list[str] = []

    localized_spikes = int(report["localized_spikes"])
    overall_coverage = float(report["localized_fraction_overall"])
    good_coverage = float(report["localized_fraction_good_clusters"])
    run_mode = str(report["run_mode"])

    if localized_spikes <= 0:
        warnings.append("localized_spike_count_zero")
    if run_mode == "exact" and overall_coverage < 0.999:
        warnings.append("exact_coverage_below_0.999")
    if run_mode == "partial" and overall_coverage < 0.95:
        warnings.append("partial_coverage_below_0.95")
    if good_coverage < 0.95:
        warnings.append("good_cluster_coverage_below_0.95")
    if summary_audit["localized_row_count"] <= 0:
        warnings.append("no_localized_good_clusters")
    if summary_audit["finite_localized_depth_rows"] != summary_audit["localized_row_count"]:
        warnings.append("nonfinite_good_cluster_depth_summary_rows")
    if summary_audit["nonzero_depth_spread_rows"] <= 0:
        warnings.append("no_nonzero_depth_spread_rows")
    if summary_audit["depth_order_violations"] > 0:
        warnings.append("depth_order_violations_present")
    if raster_dimensions is None:
        warnings.append("raster_missing_or_invalid")

    return ("pass" if not warnings else "warn"), warnings


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def write_completion_checklist(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# AL032 Phase 4 Completion Checklist",
        "",
        "| Session | Status | Mode | Coverage | Good-cluster coverage | Required outputs | Corruption summary |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        required_outputs = "ok" if row["required_outputs_present"] else "missing"
        corruption_summary = "present" if row["corrupted_region_summary_present"] else "n/a"
        lines.append(
            f"| {row['session_name']} | {row['status']} | {row['mode']} | "
            f"{float(row['localization_coverage']):.6f} | {float(row['good_cluster_coverage']):.6f} | "
            f"{required_outputs} | {corruption_summary} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_contract(schema_report: dict[str, object]) -> dict[str, object]:
    return {
        "phase": "AL032 Phase 4",
        "status": "frozen",
        "output_directory_pattern": "/scratch/am15577/UnitMatch/spike_rosters_raw/al032_session_outputs/<session_name>",
        "required_artifacts": REQUIRED_ARTIFACT_SUFFIXES,
        "optional_artifacts": OPTIONAL_ARTIFACT_SUFFIXES,
        "spike_table_columns": EXPECTED_SPIKE_TABLE_COLUMNS,
        "good_cluster_summary_columns": EXPECTED_GOOD_CLUSTER_SUMMARY_COLUMNS,
        "validation_report_keys": EXPECTED_VALIDATION_REPORT_KEYS,
        "localization_status_vocabulary": SKIP_REASON_VOCABULARY,
        "run_modes": ["exact", "partial"],
        "partial_session_rule": {
            "description": "Partial mode is allowed only when raw localization encounters known corrupted input regions and every skipped spike is explicitly labeled.",
            "required_skip_reason": "corrupted_raw_chunk",
            "required_artifact": "corrupted_region_summary.json",
        },
        "schema_consistency": {
            "spike_table_columns_consistent": schema_report["spike_table_schema_consistent"],
            "good_cluster_summary_columns_consistent": schema_report["good_cluster_summary_schema_consistent"],
            "validation_report_keys_consistent": schema_report["validation_report_schema_consistent"],
        },
    }


def main() -> None:
    args = parse_args()
    session_rows = load_session_rows(args.session_manifest.resolve())
    output_root = args.output_root.resolve()

    manifest_rows: list[dict[str, object]] = []
    spike_schema_by_session: dict[str, list[str]] = {}
    summary_schema_by_session: dict[str, list[str]] = {}
    validation_schema_by_session: dict[str, list[str]] = {}
    quality_flags: dict[str, dict[str, object]] = {}
    observed_skip_reasons: set[str] = set()
    failed_sessions: list[str] = []

    for row in session_rows:
        session_name = row["session_name"]
        session_output_dir = output_root / session_name
        spike_table_path = session_output_dir / f"{session_name}_localized_spike_table.csv.gz"
        validation_report_path = session_output_dir / f"{session_name}_validation_report.json"
        good_cluster_summary_path = session_output_dir / f"{session_name}_good_cluster_summary.csv"
        raster_path = session_output_dir / f"{session_name}_exact_depth_raster.png"
        histogram_path = session_output_dir / f"{session_name}_localized_depth_histogram.png"
        cluster_spread_path = session_output_dir / f"{session_name}_cluster_depth_spread.png"
        example_scatter_path = session_output_dir / f"{session_name}_example_cluster_scatter.png"
        corrupted_region_summary_path = session_output_dir / f"{session_name}_corrupted_region_summary.json"
        processing_windows_path = session_output_dir / f"{session_name}_processing_windows.csv"
        selected_clusters_path = session_output_dir / f"{session_name}_selected_good_clusters.csv"

        required_outputs_present = all(
            (session_output_dir / f"{session_name}_{suffix}").exists() for suffix in REQUIRED_ARTIFACT_SUFFIXES
        )

        if not required_outputs_present:
            failed_sessions.append(session_name)
            manifest_rows.append(
                {
                    "session_name": session_name,
                    "status": "failed",
                    "mode": "unknown",
                    "total_spikes": "",
                    "localized_spikes": "",
                    "skipped_spikes": "",
                    "localization_coverage": "",
                    "good_cluster_coverage": "",
                    "corrupted_interval_flag": parse_bool(row["corruption_known"]),
                    "corrupted_region_summary_present": False,
                    "required_outputs_present": False,
                    "quality_status": "fail",
                    "quality_warnings": "missing_required_outputs",
                    "localized_spike_table_path": str(spike_table_path),
                    "validation_report_path": str(validation_report_path),
                    "good_cluster_summary_path": str(good_cluster_summary_path),
                    "raster_path": str(raster_path),
                    "corrupted_region_summary_path": "",
                    "output_dir": str(session_output_dir),
                }
            )
            continue

        existing_report = json.loads(validation_report_path.read_text()) if validation_report_path.exists() else {}
        spike_audit = audit_spike_table(spike_table_path)
        summary_audit = audit_good_cluster_summary(good_cluster_summary_path)
        window_audit = audit_processing_windows(processing_windows_path)
        observed_skip_reasons.update(spike_audit["skip_reason_vocabulary_observed"])

        normalized_report = build_validation_report(
            session_name=session_name,
            existing_report=existing_report,
            row=row,
            spike_audit=spike_audit,
            summary_audit=summary_audit,
            window_audit=window_audit,
            corrupted_region_summary_path=corrupted_region_summary_path,
            raster_path=raster_path,
            histogram_path=histogram_path,
            cluster_spread_path=cluster_spread_path,
            example_scatter_path=example_scatter_path,
            selected_clusters_path=selected_clusters_path,
            spike_table_path=spike_table_path,
            summary_path=good_cluster_summary_path,
            processing_windows_path=processing_windows_path,
        )

        if args.rewrite_validation_reports:
            validation_report_path.write_text(json.dumps(normalized_report, indent=2))

        validation_schema_by_session[session_name] = list(normalized_report.keys())
        spike_schema_by_session[session_name] = spike_audit["header"]
        summary_schema_by_session[session_name] = summary_audit["header"]

        raster_dimensions: tuple[int, int] | None
        try:
            raster_dimensions = read_png_dimensions(raster_path)
        except Exception:
            raster_dimensions = None

        quality_status, quality_warnings = build_quality_review(
            report=normalized_report,
            summary_audit=summary_audit,
            raster_dimensions=raster_dimensions,
        )

        quality_flags[session_name] = {
            "quality_status": quality_status,
            "quality_warnings": quality_warnings,
            "raster_dimensions": raster_dimensions,
            "summary_depth_range_um": [summary_audit["depth_min_um"], summary_audit["depth_max_um"]],
            "observed_skip_reasons": spike_audit["skip_reason_vocabulary_observed"],
        }

        status = "complete" if quality_status == "pass" else "complete_with_warnings"
        skipped_spikes = (
            int(normalized_report["skipped_corrupted_spikes"])
            + int(normalized_report["skipped_read_failure_spikes"])
            + int(normalized_report["skipped_invalid_peak_spikes"])
            + int(normalized_report["skipped_other_spikes"])
            + int(normalized_report["unresolved_pipeline_gap_spikes"])
        )

        manifest_rows.append(
            {
                "session_name": session_name,
                "status": status,
                "mode": normalized_report["run_mode"],
                "total_spikes": int(normalized_report["total_spikes"]),
                "localized_spikes": int(normalized_report["localized_spikes"]),
                "skipped_spikes": skipped_spikes,
                "localization_coverage": float(normalized_report["localized_fraction_overall"]),
                "good_cluster_coverage": float(normalized_report["localized_fraction_good_clusters"]),
                "corrupted_interval_flag": bool(normalized_report["run_mode"] == "partial"),
                "corrupted_region_summary_present": corrupted_region_summary_path.exists(),
                "required_outputs_present": required_outputs_present,
                "quality_status": quality_status,
                "quality_warnings": ";".join(quality_warnings),
                "localized_spike_table_path": str(spike_table_path),
                "validation_report_path": str(validation_report_path),
                "good_cluster_summary_path": str(good_cluster_summary_path),
                "raster_path": str(raster_path),
                "corrupted_region_summary_path": str(corrupted_region_summary_path) if corrupted_region_summary_path.exists() else "",
                "output_dir": str(session_output_dir),
            }
        )

    spike_table_schema_consistent = all(cols == EXPECTED_SPIKE_TABLE_COLUMNS for cols in spike_schema_by_session.values())
    good_cluster_summary_schema_consistent = all(
        cols == EXPECTED_GOOD_CLUSTER_SUMMARY_COLUMNS for cols in summary_schema_by_session.values()
    )
    validation_report_schema_consistent = all(
        cols == EXPECTED_VALIDATION_REPORT_KEYS for cols in validation_schema_by_session.values()
    )

    schema_report = {
        "session_count": len(session_rows),
        "completed_session_count": sum(1 for row in manifest_rows if str(row["status"]).startswith("complete")),
        "failed_session_count": len(failed_sessions),
        "failed_sessions": failed_sessions,
        "spike_table_schema_consistent": spike_table_schema_consistent,
        "good_cluster_summary_schema_consistent": good_cluster_summary_schema_consistent,
        "validation_report_schema_consistent": validation_report_schema_consistent,
        "expected_spike_table_columns": EXPECTED_SPIKE_TABLE_COLUMNS,
        "expected_good_cluster_summary_columns": EXPECTED_GOOD_CLUSTER_SUMMARY_COLUMNS,
        "expected_validation_report_keys": EXPECTED_VALIDATION_REPORT_KEYS,
        "spike_table_columns_by_session": spike_schema_by_session,
        "good_cluster_summary_columns_by_session": summary_schema_by_session,
        "validation_report_keys_by_session": validation_schema_by_session,
        "skip_reason_vocabulary_contract": SKIP_REASON_VOCABULARY,
        "skip_reason_vocabulary_observed": sorted(observed_skip_reasons),
        "quality_flags_by_session": quality_flags,
    }

    contract = build_contract(schema_report)

    write_csv(args.phase4_manifest_csv.resolve(), manifest_rows)
    write_json(args.phase4_manifest_json.resolve(), manifest_rows)
    write_json(args.schema_report_json.resolve(), schema_report)
    write_completion_checklist(args.completion_checklist_md.resolve(), manifest_rows)
    write_json(args.contract_json.resolve(), contract)

    print(f"Wrote Phase 4 manifest CSV to {args.phase4_manifest_csv}")
    print(f"Wrote Phase 4 manifest JSON to {args.phase4_manifest_json}")
    print(f"Wrote schema report to {args.schema_report_json}")
    print(f"Wrote completion checklist to {args.completion_checklist_md}")
    print(f"Wrote frozen contract to {args.contract_json}")


if __name__ == "__main__":
    main()
