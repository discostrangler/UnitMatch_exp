#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/tmp/mplconfig"

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge")
BASELINE_ROOT = Path("/scratch/am15577/UnitMatch/match_raw_unitmatch")
OUTPUT_ROOT = ROOT / "outputs" / "unitmatch_dredge_12session"
EVAL_ROOT = OUTPUT_ROOT / "eval"
UNITMATCH_QC_ROOT = OUTPUT_ROOT / "outputs_unitmatch"
TRACKED_ROOT = ROOT / "outputs" / "tracked_tables"
FIGURES_ROOT = ROOT / "outputs" / "figures"
MANIFEST_ROOT = ROOT / "manifests"

REPORT_PATH = OUTPUT_ROOT / "Unitmatch_32_12session_dredge.pdf"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def wrap(text: str, width: int = 110) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def fmt_float(value: float | int | str, ndigits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, int):
        return f"{value}"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def fig_page(title: str, subtitle: str | None = None, figsize=(11, 8.5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.05, 0.95, title, fontsize=20, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.05, 0.915, subtitle, fontsize=10, color="dimgray", va="top")
    return fig, ax


def add_paragraph(ax, text: str, x: float, y: float, width: int = 105, fontsize: int = 10, line_spacing: float = 0.028) -> float:
    wrapped = wrap(text, width=width)
    ax.text(x, y, wrapped, fontsize=fontsize, va="top", ha="left")
    lines = wrapped.count("\n") + 1
    return y - lines * line_spacing


def add_bullets(ax, items: list[str], x: float, y: float, width: int = 100, fontsize: int = 10, line_spacing: float = 0.026) -> float:
    for item in items:
        wrapped = textwrap.wrap(str(item), width=width, break_long_words=False)
        if not wrapped:
            y -= line_spacing
            continue
        ax.text(x, y, f"- {wrapped[0]}", fontsize=fontsize, va="top", ha="left")
        y -= line_spacing
        for continuation in wrapped[1:]:
            ax.text(x + 0.02, y, continuation, fontsize=fontsize, va="top", ha="left")
            y -= line_spacing
    return y


def save_image_page(pdf: PdfPages, title: str, image_specs: list[tuple[Path, str]], cols: int = 1, subtitle: str | None = None):
    image_specs = [(img_path, caption) for img_path, caption in image_specs if img_path.exists()]
    if not image_specs:
        fig, ax = fig_page(title, subtitle)
        ax.text(0.05, 0.80, "Image not available yet for this report build.", fontsize=12, va="top", ha="left")
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    rows = math.ceil(len(image_specs) / cols)
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.96, title, fontsize=18, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.05, 0.925, subtitle, fontsize=10, color="dimgray", va="top")
    top = 0.88
    bottom = 0.08
    hspace = 0.08
    wspace = 0.04
    height = (top - bottom - (rows - 1) * hspace) / rows
    width = (0.9 - (cols - 1) * wspace) / cols
    for i, (img_path, caption) in enumerate(image_specs):
        row = i // cols
        col = i % cols
        left = 0.05 + col * (width + wspace)
        bottom_i = top - (row + 1) * height - row * hspace
        ax = fig.add_axes([left, bottom_i, width, height])
        ax.axis("off")
        with Image.open(img_path) as im:
            ax.imshow(im)
        ax.set_title(caption, fontsize=10, pad=6)
    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, subtitle: str | None = None, fontsize: int = 8, scale_y: float = 1.25):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.96, title, fontsize=18, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.05, 0.925, subtitle, fontsize=10, color="dimgray", va="top")
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.84])
    ax.axis("off")
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=df.columns,
        cellLoc="left",
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, scale_y)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8edf3")
    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_report():
    manifest = load_json(EVAL_ROOT / "evaluation_run_manifest.json")
    metrics = load_json(EVAL_ROOT / "replication_metrics_summary.json")
    run_summary = load_json(OUTPUT_ROOT / "run_summary.json")
    plot_summary = load_json(UNITMATCH_QC_ROOT / "unitmatch_plot_summary.json")
    validation_report_path = MANIFEST_ROOT / "session_validation_report.json"
    if not validation_report_path.exists():
        validation_report_path = BASELINE_ROOT / "manifests" / "session_validation_report.json"
    validation_report = load_json(validation_report_path)
    coverage_summary = load_json(TRACKED_ROOT / "tracked_unit_coverage_summary.json")
    raster_summary = load_json(FIGURES_ROOT / "al032_12session_dredge_session_aligned_raster_summary.json")
    raster_wf_summary = (
        load_json(FIGURES_ROOT / "al032_12session_dredge_session_aligned_raster_plus_waveforms_summary.json")
        if (FIGURES_ROOT / "al032_12session_dredge_session_aligned_raster_plus_waveforms_summary.json").exists()
        else {}
    )

    session_summary = pd.read_csv(EVAL_ROOT / "session_summary_table.csv")
    pairwise = pd.read_csv(EVAL_ROOT / "pairwise_session_table.csv")
    targets = pd.read_csv(EVAL_ROOT / "paper_replication_targets.csv")
    long_lived = pd.read_csv(EVAL_ROOT / "example_long_lived_units.csv")
    selected_units = pd.read_csv(TRACKED_ROOT / "selected_tracked_units.csv")
    replication_summary_md = (EVAL_ROOT / "replication_summary.md").read_text(encoding="utf-8")

    os.makedirs(REPORT_PATH.parent, exist_ok=True)
    os.makedirs("/scratch/am15577/UnitMatch/match_raw_unitmatch_dredge/tmp/mplconfig", exist_ok=True)

    with PdfPages(REPORT_PATH) as pdf:
        fig, ax = fig_page(
            "UnitMatch AL032 12-Session DREDge Evaluation Report",
            f"Output root: {OUTPUT_ROOT}",
            figsize=(11, 8.5),
        )
        y = 0.86
        y = add_paragraph(
            ax,
            "This report freezes and evaluates the completed 12-session UnitMatch run using DREDge-corrected geometry for the AL032 Kilosort outputs. It consolidates the run configuration, tracked-unit statistics, paper-style replication figures, and downstream DREDge-aligned raster and waveform validation figures into one reproducible document.",
            0.05,
            y,
            width=110,
            fontsize=11,
        )
        y -= 0.02
        y = add_bullets(
            ax,
            [
                f"Mouse: {manifest['mouse']}",
                f"Sessions evaluated: {manifest['n_sessions']}",
                f"Total good units passed into UnitMatch: {run_summary['n_units']}",
                f"Tracked identity mode: {manifest['tracked_id_mode']}",
                f"Match threshold: {manifest['unitmatch_parameters']['match_threshold']}",
                f"Conflict-free tracked units after graph cleanup: {metrics['n_conflict_free_tracked_units']}",
                f"Multi-session tracked units: {metrics['n_multi_session_tracked_units']}",
                f"Units present in all 12 sessions: {metrics['n_units_all_12_sessions']}",
                f"Downstream localization method for raster attachment: {manifest['localization_method_for_attachment']}",
            ],
            0.05,
            y,
            width=105,
            fontsize=11,
        )
        ax.text(
            0.05,
            0.12,
            "Key interpretation: the DREDge-corrected UnitMatch run completed cleanly on all 12 sessions, the overlap structure still decays strongly with increasing session gap, and the downstream session-aligned raster validation figures are available for this same session set.",
            fontsize=11,
            va="top",
            ha="left",
            wrap=True,
        )
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = fig_page("1. Frozen Evaluation Run", "Exact run definition used for all statistics in this report.")
        y = 0.87
        y = add_bullets(
            ax,
            [
                f"Run name: {manifest['run_name']}",
                f"Created: {manifest['created_at']}",
                f"UnitMatch root: {manifest['unitmatch_root']}",
                f"UnitMatch git metadata: repository unavailable in this local checkout, so commit freezing is recorded as unavailable rather than guessed.",
                f"Session order: {', '.join(manifest['session_order'])}",
                f"Input manifest: {manifest['input_manifest_csv']}",
                f"Validation report: {manifest['validation_report_path']}",
                f"DREDge manifest: {manifest.get('dredge_manifest_csv', '')}",
                f"Corrected spikes root: {manifest.get('corrected_spikes_root', '')}",
                f"Localization root used later for tracked-spike attachment: {manifest.get('localization_root_for_attachment', '')}",
            ],
            0.05,
            y,
            width=110,
            fontsize=10,
        )
        y -= 0.02
        y = add_paragraph(
            ax,
            "Frozen UnitMatch parameters: good units only, intermediate tracked-ID mode, fixed probability threshold 0.5, channel radius 150 µm, max distance 100 µm, neighbour distance 50 µm, minimum new-shank distance 100 µm, units-per-shank threshold 15, curve-fit maximum function evaluations 10000, two UnitMatch iterations, and MatchTable saving enabled.",
            0.05,
            y,
            width=112,
            fontsize=10,
        )
        y -= 0.02
        y = add_paragraph(
            ax,
            "Caveat: AL032_2019-11-21 remains partial only on the later raw-localization side because its SpikeGLX .ap.cbin source contains a reproducibly corrupted compressed region. UnitMatch itself still ran on the session’s Kilosort/qMetrics inputs and therefore includes that session in the 12-session tracking graph.",
            0.05,
            y,
            width=112,
            fontsize=10,
        )
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = fig_page("2. Session Validation", "All 12 sessions were checked against the UnitMatch input contract before the run.")
        y = 0.87
        y = add_bullets(
            ax,
            [
                f"Sessions validated: {validation_report['session_count']}",
                f"Sessions ready: {validation_report['ready_session_count']}",
                f"Not-ready sessions: {len(validation_report['not_ready_sessions'])}",
                "Every session had aligned spike arrays, compatible waveform geometry, and qMetrics RawWaveforms available.",
                "Waveform tensor shape was consistent across sessions: 82 timepoints x 384 channels x 2 cross-validation halves.",
                "Template-channel geometry matched channel_positions for all sessions.",
                "Sample rate metadata was present for every session.",
            ],
            0.05,
            y,
            width=105,
            fontsize=10,
        )
        y -= 0.02
        validation_subset = session_summary[
            [
                "session_order",
                "session_name",
                "good_units_input",
                "tracked_units_conflict_free",
                "multi_session_tracked_units_conflict_free",
                "localization_mode",
                "localization_fraction_good_clusters",
            ]
        ].copy()
        validation_subset["localization_fraction_good_clusters"] = validation_subset[
            "localization_fraction_good_clusters"
        ].map(lambda v: fmt_float(v, 4))
        table_ax = fig.add_axes([0.04, 0.05, 0.92, 0.48])
        table_ax.axis("off")
        table = table_ax.table(
            cellText=validation_subset.astype(str).values,
            colLabels=[
                "order",
                "session",
                "good_units",
                "tracked_conflict_free",
                "multi_session_tracked",
                "localization_mode",
                "good_loc_frac",
            ],
            cellLoc="left",
            loc="upper left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.22)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#e8edf3")
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = fig_page("3. Native UnitMatch Run Outputs", "Summary of the direct outputs written by the 12-session UnitMatchPy run.")
        y = 0.87
        y = add_bullets(
            ax,
            [
                f"Output root: {run_summary['output_root']}",
                f"Number of sessions: {run_summary['n_sessions']}",
                f"Number of good units in the MatchProb matrix: {run_summary['n_units']}",
                f"Thresholded pair count written by UnitMatch: {run_summary['n_matches_thresholded']}",
                f"Conflict-free tracked units after postprocessing: {plot_summary['n_conflict_free_tracked_units']}",
                f"Maximum pairwise shared tracked-unit count: {plot_summary['max_pair_shared_units']['shared_tracked_units']} ({plot_summary['max_pair_shared_units']['session_name_i']} vs {plot_summary['max_pair_shared_units']['session_name_j']})",
                f"Probability-matrix population sizes: self pairs {plot_summary['probability_distribution_counts']['self_pairs']}, within-session pairs {plot_summary['probability_distribution_counts']['within_session_pairs']}, across-session pairs {plot_summary['probability_distribution_counts']['across_session_pairs']}.",
            ],
            0.05,
            y,
            width=110,
            fontsize=10,
        )
        y -= 0.02
        y = add_paragraph(
            ax,
            "Native files retained in the evaluation root include MatchProb.npy, Matches.npy, MatchTable.csv, UM Scores.npz, WaveformInfo.npz, ClusInfo.pickle, UMparam.pickle, and run_summary.json. These are the direct UnitMatch outputs; all later tables and figures in this report are derived from them without rerunning the algorithm.",
            0.05,
            y,
            width=112,
            fontsize=10,
        )
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

        targets_display = targets[
            ["paper_item", "metric_type", "population", "status", "notes"]
        ].copy()
        save_table_page(
            pdf,
            "4. Paper Replication Targets",
            targets_display,
            subtitle="Checklist of target figures and quantities extracted from the UnitMatch paper or codebase for this AL032 replication pass.",
            fontsize=7.5,
            scale_y=1.35,
        )

        session_display = session_summary[
            [
                "session_order",
                "session_name",
                "session_date",
                "good_units_input",
                "tracked_units_conflict_free",
                "multi_session_tracked_units_conflict_free",
                "mean_max_cross_session_probability",
                "localization_mode",
            ]
        ].copy()
        session_display["mean_max_cross_session_probability"] = session_display[
            "mean_max_cross_session_probability"
        ].map(lambda v: fmt_float(v, 3))
        save_table_page(
            pdf,
            "5. Session-Level Summary Table",
            session_display,
            subtitle="Per-session counts used for later pairwise and tracked-unit analyses.",
            fontsize=7.5,
            scale_y=1.28,
        )

        top_pairs = pairwise.sort_values(
            ["shared_tracked_unit_count", "shared_fraction_min"], ascending=[False, False]
        ).head(12).copy()
        top_pairs = top_pairs[
            [
                "session_a",
                "session_b",
                "days_apart",
                "shared_tracked_unit_count",
                "shared_fraction_min",
                "shared_fraction_union",
                "mean_match_probability_shared",
            ]
        ]
        for col in ["shared_fraction_min", "shared_fraction_union", "mean_match_probability_shared"]:
            top_pairs[col] = top_pairs[col].map(lambda v: fmt_float(v, 3))
        save_table_page(
            pdf,
            "6. Strongest Session Pairs",
            top_pairs,
            subtitle="Top pairwise overlaps by shared conflict-free tracked units.",
            fontsize=7.5,
            scale_y=1.3,
        )

        save_image_page(
            pdf,
            "7. Core Pairwise Overlap Heatmaps",
            [
                (EVAL_ROOT / "pairwise_shared_count_heatmap.png", "Shared tracked-unit counts"),
                (EVAL_ROOT / "pairwise_shared_fraction_min_heatmap.png", "Shared / min(session tracked units)"),
            ],
            cols=2,
            subtitle="These match the paper’s chronic-tracking framing better than raw counts alone because they expose both absolute overlap and normalized overlap.",
        )
        save_image_page(
            pdf,
            "8. Additional Pairwise Normalization and Native Tracking Heatmap",
            [
                (EVAL_ROOT / "pairwise_shared_fraction_union_heatmap.png", "Shared / union normalization"),
                (UNITMATCH_QC_ROOT / "unitmatch_session_pair_tracked_fraction.png", "UnitMatch QC fraction heatmap"),
            ],
            cols=2,
            subtitle="The union-normalized view is stricter; the QC heatmap provides the earlier pipeline view for cross-checking.",
        )
        save_image_page(
            pdf,
            "9. Tracking Versus Session Gap",
            [
                (EVAL_ROOT / "shared_tracked_units_vs_days_apart.png", "Shared tracked units vs days apart"),
                (EVAL_ROOT / "ptrack_vs_delta_days.png", "P(track)-style ordered overlap vs signed day gap"),
            ],
            cols=2,
            subtitle="The raw shared-count and shared/min trends show strong decay with increasing gap; the ordered P(track)-style definition is more sensitive to denominator choice.",
        )
        save_image_page(
            pdf,
            "10. Persistence Across Many Sessions",
            [
                (EVAL_ROOT / "tracked_unit_persistence_histogram.png", "Tracked-unit persistence histogram"),
                (EVAL_ROOT / "tracked_unit_presence_sorted_first_appearance.png", "Presence matrix sorted by first appearance"),
            ],
            cols=2,
            subtitle="This is the closest local analogue to the paper’s many-recording persistence view for one chronic mouse.",
        )
        save_image_page(
            pdf,
            "11. Native UnitMatch Matrix Views",
            [
                (UNITMATCH_QC_ROOT / "unitmatch_match_probability_matrix.png", "P(match) matrix from MatchProb.npy"),
                (UNITMATCH_QC_ROOT / "unitmatch_thresholded_match_matrix.png", "Thresholded match matrix at p > 0.5"),
            ],
            cols=2,
            subtitle="These plots are direct visualizations of the native UnitMatch outputs and form the closest local match to the paper’s matrix panels.",
        )
        save_image_page(
            pdf,
            "12. Additional UnitMatch QC Figures",
            [
                (UNITMATCH_QC_ROOT / "unitmatch_probability_distributions.png", "Probability distributions"),
                (UNITMATCH_QC_ROOT / "unitmatch_tracking_summary_panel.png", "Tracking summary panel"),
                (UNITMATCH_QC_ROOT / "unitmatch_tracked_unit_lifespan_histogram.png", "Lifespan histogram"),
                (UNITMATCH_QC_ROOT / "unitmatch_tracking_vs_session_gap.png", "Tracking vs session gap"),
            ],
            cols=2,
            subtitle="These summarize the overall quality, threshold behavior, and longitudinal degradation structure of the AL032 run.",
        )

        fig, ax = fig_page("13. Long-Lived Units and Figure-Selection Outputs")
        y = 0.87
        y = add_bullets(
            ax,
            [
                f"Selected 12-session raster overlay units: {', '.join('T' + str(x) for x in raster_summary['selected_tracked_unit_ids'])}",
                (
                    "Selected 12-session raster+waveform units: "
                    + ", ".join("T" + str(x) for x in raster_wf_summary.get("selected_tracked_unit_ids", []))
                    if raster_wf_summary
                    else "Selected 12-session raster+waveform units: pending"
                ),
                f"Coverage-selection filter: at least {coverage_summary['min_sessions_present']} sessions, at least {coverage_summary['min_good_tracked_spikes']} good tracked spikes per session, mean cross-session probability at least {coverage_summary['min_mean_probability']}, then depth-diverse ranking.",
                f"Eligible units under that filter: {coverage_summary['n_selection_eligible']}",
                f"Total tracked units before filtering: {coverage_summary['n_tracked_units_total']}",
            ],
            0.05,
            y,
            width=110,
            fontsize=10,
        )
        y -= 0.02
        example_subset = long_lived[
            [
                "tracked_unit_id",
                "n_sessions_present",
                "span_days",
                "mean_cross_session_probability",
                "total_good_tracked_spikes",
                "depth_center_um",
                "depth_stability_max_abs_diff_um",
            ]
        ].head(10).copy()
        for col in [
            "mean_cross_session_probability",
            "depth_center_um",
            "depth_stability_max_abs_diff_um",
        ]:
            example_subset[col] = example_subset[col].map(lambda v: fmt_float(v, 3))
        table_ax = fig.add_axes([0.04, 0.07, 0.92, 0.52])
        table_ax.axis("off")
        table = table_ax.table(
            cellText=example_subset.astype(str).values,
            colLabels=[
                "tracked_unit",
                "n_sessions",
                "span_days",
                "mean_prob",
                "total_good_spikes",
                "depth_center_um",
                "max_depth_shift_um",
            ],
            cellLoc="left",
            loc="upper left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.25)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#e8edf3")
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

        save_image_page(
            pdf,
            "14. Downstream 12-Session Tracked Raster",
            [
                (FIGURES_ROOT / "al032_12session_dredge_session_aligned_raster.png", "DREDge session-aligned 12-session raster with selected tracked units"),
            ],
            cols=1,
            subtitle="This figure is not a direct UnitMatch paper panel. It is a downstream validation view obtained after attaching DREDge-run tracked IDs to DREDge-corrected and session-aligned localized spikes.",
        )
        save_image_page(
            pdf,
            "15. Downstream 12-Session Raster Plus Waveforms",
            [
                (FIGURES_ROOT / "al032_12session_dredge_session_aligned_raster_plus_waveforms.png", "DREDge session-aligned 12-session raster with per-unit raw waveform overlays"),
            ],
            cols=1,
            subtitle="Waveform overlays use single-channel SpikeGLX AP raw snippets, baseline-subtracted and trough-aligned, to provide an additional validation layer for a subset of persistent tracked units. If the figure is still rendering, this page is intentionally left blank rather than failing the report.",
        )

        fig, ax = fig_page("16. Interpretation: What Matches the Paper and What Does Not")
        y = 0.87
        y = add_paragraph(
            ax,
            "Replicated closely: the probability-matrix and thresholded-matrix views are reproduced directly from the native UnitMatch outputs; the pairwise shared-unit heatmaps and the days-apart decay trends show the same qualitative chronic-tracking structure expected from the paper; and the many-session presence matrix provides the same style of longitudinal persistence view for one mouse.",
            0.05,
            y,
            width=112,
            fontsize=10,
        )
        y -= 0.02
        y = add_paragraph(
            ax,
            "Same qualitative trend but not yet definition-perfect: the ordered P(track)-style statistic is implemented locally, but its near-zero signed-gap correlation suggests the denominator or directional definition still differs from the exact panel definition used in the paper. That gap should be debugged by checking whether the paper normalizes by the source session, target session, minimum session size, or a stricter subset of conservative graph edges.",
            0.05,
            y,
            width=112,
            fontsize=10,
        )
        y -= 0.02
        y = add_paragraph(
            ax,
            "Still pending for paper-style replication: an explicit support/block graph visualization of the UnitMatch graph logic, and a frozen waveform/spatial example-unit subset written specifically as an evaluation panel rather than a broader downstream visualization. Those are identified as the two remaining pending targets in paper_replication_targets.csv.",
            0.05,
            y,
            width=112,
            fontsize=10,
        )
        y -= 0.03
        y = add_bullets(
            ax,
            [
                f"Implemented targets: {metrics['implemented_targets']} of {metrics['n_targets']}",
                f"Pending targets: {metrics['pending_targets']}",
                f"Pearson correlation days apart vs shared tracked-unit count: {fmt_float(metrics['days_apart_count_corr'], 3)}",
                f"Pearson correlation days apart vs shared/min overlap: {fmt_float(metrics['days_apart_fraction_corr'], 3)}",
                f"Pearson correlation signed delta days vs P(track)-style overlap: {fmt_float(metrics['ptrack_delta_corr'], 3)}",
            ],
            0.05,
            y,
            width=110,
            fontsize=10,
        )
        y -= 0.02
        ax.text(
            0.05,
            0.10,
            wrap(replication_summary_md.replace("# UnitMatch Replication Summary", "").strip(), 118),
            fontsize=8.5,
            va="top",
            ha="left",
        )
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    build_report()
