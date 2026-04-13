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
FEATURE_VIZ_ROOT = ROOT / "outputs" / "unitmatch_dredge_12session" / "feature_viz"
REPORT_PATH = FEATURE_VIZ_ROOT / "feature_report.pdf"


PLOT_SPECS = [
    {
        "filename": "feature_stability_summary_spatial.png",
        "title": "Spatial Feature Stability Summary",
        "subtitle": "Population summary only: median normalized drift with IQR shading.",
        "kind": "summary",
        "family": "spatial",
    },
    {
        "filename": "feature_stability_summary_waveform.png",
        "title": "Waveform Feature Stability Summary",
        "subtitle": "Population summary only: median normalized drift with IQR shading.",
        "kind": "summary",
        "family": "waveform",
    },
    {
        "filename": "feature_stability_summary_trajectory.png",
        "title": "Trajectory Feature Stability Summary",
        "subtitle": "Population summary only: median normalized drift with IQR shading.",
        "kind": "summary",
        "family": "trajectory",
    },
    {
        "filename": "feature_example_units_spatial.png",
        "title": "Example Units: Spatial Features",
        "subtitle": "Four long-lived tracked units only, shown as normalized-drift trajectories.",
        "kind": "example",
        "family": "spatial",
    },
    {
        "filename": "feature_example_units_waveform.png",
        "title": "Example Units: Waveform Features",
        "subtitle": "Four long-lived tracked units only, shown as normalized-drift trajectories.",
        "kind": "example",
        "family": "waveform",
    },
    {
        "filename": "feature_example_units_trajectory.png",
        "title": "Example Units: Trajectory Features",
        "subtitle": "Four long-lived tracked units only, shown as normalized-drift trajectories.",
        "kind": "example",
        "family": "trajectory",
    },
    {
        "filename": "feature_volatility_ranking.png",
        "title": "Feature Volatility Ranking",
        "subtitle": "Features sorted by median absolute normalized drift across selected long-lived units.",
        "kind": "ranking",
        "family": None,
    },
]


def wrap(text: str, width: int = 72) -> list[str]:
    return textwrap.wrap(str(text), width=width, break_long_words=False, replace_whitespace=False)


def add_wrapped_text(
    fig: plt.Figure,
    text: str,
    x: float,
    y: float,
    width: int = 72,
    fontsize: int = 9,
    line_spacing: float = 0.023,
    weight: str | None = None,
    color: str = "black",
) -> float:
    lines = wrap(text, width=width) or [""]
    for line in lines:
        fig.text(x, y, line, fontsize=fontsize, va="top", ha="left", fontweight=weight, color=color)
        y -= line_spacing
    return y


def add_bullets(
    fig: plt.Figure,
    items: list[str],
    x: float,
    y: float,
    width: int = 68,
    fontsize: int = 9,
    line_spacing: float = 0.022,
) -> float:
    for item in items:
        lines = wrap(item, width=width) or [""]
        fig.text(x, y, f"- {lines[0]}", fontsize=fontsize, va="top", ha="left")
        y -= line_spacing
        for line in lines[1:]:
            fig.text(x + 0.02, y, line, fontsize=fontsize, va="top", ha="left")
            y -= line_spacing
    return y


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    panel_guide = pd.read_csv(FEATURE_VIZ_ROOT / "feature_panel_guide.csv")
    filter_summary = pd.read_csv(FEATURE_VIZ_ROOT / "feature_filter_summary.csv")
    example_units = pd.read_csv(FEATURE_VIZ_ROOT / "example_units_selected.csv")
    with (FEATURE_VIZ_ROOT / "feature_timecourse_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return panel_guide, filter_summary, example_units, summary


def get_family_features(panel_guide: pd.DataFrame, filter_summary: pd.DataFrame, family: str) -> list[dict]:
    kept = filter_summary[filter_summary["keep_for_plots"]].copy()
    fam = panel_guide[panel_guide["family"] == family].copy()
    fam = fam.merge(
        kept[["feature_name", "median_abs_normalized_drift", "short_note"]],
        on="feature_name",
        how="inner",
    )
    return fam.to_dict("records")


def place_image(fig: plt.Figure, image_path: Path, box: tuple[float, float, float, float]) -> None:
    left, bottom, width, height = box
    with Image.open(image_path) as im:
        img_w, img_h = im.size
        img_ratio = img_w / img_h
        box_ratio = width / height
        if img_ratio > box_ratio:
            draw_w = width
            draw_h = width / img_ratio
        else:
            draw_h = height
            draw_w = height * img_ratio
        x = left + (width - draw_w) / 2
        y = bottom + (height - draw_h) / 2
        ax = fig.add_axes([x, y, draw_w, draw_h])
        ax.axis("off")
        ax.imshow(im)


def render_cover(pdf: PdfPages, summary: dict, filter_summary: pd.DataFrame, example_units: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, "UnitMatch Feature Stability Report", fontsize=22, fontweight="bold", va="top")
    fig.text(
        0.05,
        0.91,
        "AL032 12-session raw UnitMatch run: refined feature-trajectory visualization",
        fontsize=12,
        color="dimgray",
        va="top",
    )
    y = 0.84
    y = add_wrapped_text(
        fig,
        "This report summarizes the refined feature-trajectory plots built from the native UnitMatch outputs. "
        "The goal is to show feature stability over time without the clutter of plotting every tracked unit.",
        0.05,
        y,
        width=95,
        fontsize=11,
        line_spacing=0.028,
    )
    y -= 0.015
    y = add_bullets(
        fig,
        [
            f"Total derived per-unit features inventoried: {summary['n_total_features']}",
            f"Features kept for refined plots: {summary['n_kept_features']}",
            f"Dropped features: {', '.join(summary['dropped_features'])}",
            f"Normalization used: {summary['normalization']}",
            "Summary plots show only the population median and IQR.",
            f"Example plots show only four long-lived tracked units: {', '.join('T' + str(x) for x in summary['selected_example_units'])}.",
            "Robust y-limits were set from the 5th to 95th percentile of the plotted values.",
        ],
        0.05,
        y,
        width=90,
        fontsize=10,
        line_spacing=0.026,
    )
    y -= 0.01
    fig.text(0.05, y, "Selected example units", fontsize=12, fontweight="bold", va="top")
    y -= 0.03
    for row in example_units.itertuples(index=False):
        fig.text(
            0.05,
            y,
            f"T{int(row.tracked_unit_id)}: present in {int(row.n_sessions_present)} sessions, "
            f"mean cross-session probability {float(row.mean_cross_session_probability):.3f}, "
            f"depth center {float(row.depth_center_um):.1f} um",
            fontsize=10,
            va="top",
            ha="left",
        )
        y -= 0.03
    fig.text(
        0.05,
        0.08,
        f"Output folder: {FEATURE_VIZ_ROOT}",
        fontsize=9,
        color="dimgray",
        va="bottom",
        ha="left",
    )
    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_methods(pdf: PdfPages, filter_summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, "Methods and Feature Filtering", fontsize=20, fontweight="bold", va="top")
    y = 0.89
    y = add_wrapped_text(
        fig,
        "Feature trajectories were built from per-unit observables derived from WaveformInfo.npz. "
        "Pairwise UnitMatch score matrices in UM Scores.npz and MatchTable.csv were inventoried but excluded from these timecourse plots because they are pairwise, not per-unit.",
        0.05,
        y,
        width=104,
        fontsize=10,
    )
    y -= 0.02
    fig.text(0.05, y, "Filtering rules", fontsize=12, fontweight="bold", va="top")
    y -= 0.03
    y = add_bullets(
        fig,
        [
            "Drop features that are effectively constant or numerically degenerate.",
            "Drop discrete index-proxy features that are not informative as continuous timecourses.",
            "Normalize drift as (feature_value - first_value) / feature_global_IQR.",
            "Separate figures into spatial, waveform, and trajectory feature families.",
            "For example-unit plots, keep only four depth-diverse long-lived tracked units.",
        ],
        0.05,
        y,
        width=95,
        fontsize=10,
    )
    kept = filter_summary[filter_summary["keep_for_plots"]]
    dropped = filter_summary[~filter_summary["keep_for_plots"]]
    y -= 0.02
    fig.text(0.05, y, "Dropped features", fontsize=12, fontweight="bold", va="top")
    y -= 0.03
    y = add_bullets(
        fig,
        [
            f"{row.label}: {row.drop_reason}"
            for row in dropped.itertuples(index=False)
        ],
        0.05,
        y,
        width=95,
        fontsize=10,
    )
    y -= 0.02
    fig.text(0.05, y, "Kept feature families", fontsize=12, fontweight="bold", va="top")
    y -= 0.03
    family_counts = kept.groupby("family")["feature_name"].nunique().to_dict()
    y = add_bullets(
        fig,
        [
            f"Spatial: {family_counts.get('spatial', 0)} features",
            f"Waveform: {family_counts.get('waveform', 0)} features",
            f"Trajectory / agreement: {family_counts.get('trajectory', 0)} features",
        ],
        0.05,
        y,
        width=70,
        fontsize=10,
    )
    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_plot_page(
    pdf: PdfPages,
    image_path: Path,
    title: str,
    subtitle: str,
    text_blocks: list[tuple[str, list[str]]],
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, title, fontsize=18, fontweight="bold", va="top")
    fig.text(0.05, 0.92, subtitle, fontsize=10, color="dimgray", va="top")
    place_image(fig, image_path, (0.04, 0.09, 0.56, 0.78))

    y = 0.88
    for heading, bullets in text_blocks:
        fig.text(0.64, y, heading, fontsize=11, fontweight="bold", va="top")
        y -= 0.03
        y = add_bullets(fig, bullets, 0.64, y, width=42, fontsize=8.8, line_spacing=0.0205)
        y -= 0.02
    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


def feature_bullets(features: list[dict]) -> list[str]:
    bullets = []
    for feature in features:
        bullets.append(
            f"{feature['label']}: {feature['definition']} Interpretation: {feature['interpretation']} "
            f"Median |normalized drift| = {float(feature['median_abs_normalized_drift']):.3f}."
        )
    return bullets


def build_pdf() -> None:
    panel_guide, filter_summary, example_units, summary = load_inputs()
    os.makedirs(FEATURE_VIZ_ROOT, exist_ok=True)
    os.makedirs(ROOT / "tmp" / "mplconfig", exist_ok=True)

    with PdfPages(REPORT_PATH) as pdf:
        render_cover(pdf, summary, filter_summary, example_units)
        render_methods(pdf, filter_summary)

        for spec in PLOT_SPECS:
            image_path = FEATURE_VIZ_ROOT / spec["filename"]
            if not image_path.exists():
                continue

            if spec["kind"] == "ranking":
                blocks = [
                    (
                        "How to read this plot",
                        [
                            "Each bar is one retained feature.",
                            "Bar height is median absolute normalized drift across the selected long-lived tracked units.",
                            "Higher bars indicate more temporal volatility after scaling by the feature-wide IQR.",
                            "Colors denote feature family: spatial, waveform, or trajectory.",
                        ],
                    ),
                    (
                        "What is being ranked",
                        [
                            "The ranking compresses the whole timecourse analysis into one stability metric per feature.",
                            "It is useful for identifying which UnitMatch features are comparatively stable and which are comparatively noisy over long session spans.",
                        ],
                    ),
                ]
            else:
                features = get_family_features(panel_guide, filter_summary, spec["family"])
                general_bullets = [
                    "X-axis is days since first appearance of a tracked unit.",
                    "Y-axis is normalized drift, defined as (feature value - first value) / feature global IQR.",
                ]
                if spec["kind"] == "summary":
                    general_bullets.extend(
                        [
                            "Thick black line is the across-unit median at each elapsed-time point.",
                            "Gray shading is the interquartile range (25th to 75th percentile).",
                            "No per-unit colored trajectories are shown in these summary panels.",
                        ]
                    )
                else:
                    unit_labels = ", ".join(f"T{int(x)}" for x in summary["selected_example_units"])
                    general_bullets.extend(
                        [
                            f"Only four example tracked units are shown: {unit_labels}.",
                            "Each colored line is one tracked unit trajectory through time.",
                            "These units were chosen to be long-lived and depth-diverse rather than simply the highest-count units.",
                        ]
                    )

                blocks = [
                    ("How to read this plot", general_bullets),
                    ("Features tracked in this figure", feature_bullets(features)),
                ]

            render_plot_page(pdf, image_path, spec["title"], spec["subtitle"], blocks)


def main() -> None:
    build_pdf()
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
