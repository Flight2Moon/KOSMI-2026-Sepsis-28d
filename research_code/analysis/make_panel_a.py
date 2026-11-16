"""Panel A: flat 3-class versus death-first cascade, discrimination forest plot.

Reads the paired bootstrap output produced by
`paired_bootstrap_flat_vs_cascade.py` and renders a near-square, publication
ready forest plot of macro-AUROC by evaluation setting.

The paired difference and its confidence interval are deliberately left out of
the figure; they belong in the table or the caption. Pass `--delta-column` if a
standalone version carrying those numbers is needed.

Layout is defined in physical units (millimetres) and type sizes are absolute
points, so the figure can be placed at 100% scale without rescaling and the
labels keep their intended size. Both a high-resolution raster and a vector PDF
are written; prefer the PDF wherever the target allows it.

Usage
-----
    python analysis/make_panel_a.py                 # 160 mm wide, 600 dpi
    python analysis/make_panel_a.py --width-mm 100 --aspect 0.98   # near-square
    python analysis/make_panel_a.py --dpi 1200 --model lightgbm
    python analysis/make_panel_a.py --delta-column  # restore the right column
    python analysis/make_panel_a.py --results other.csv --out figures/panel_a
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
# Paths resolve from the project root, so the script runs from any working
# directory and on any machine without edits.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "analysis/results/flat_vs_cascade_paired_bootstrap.csv"
DEFAULT_OUT_STEM = PROJECT_ROOT / "analysis/figures/PanelA_flat_vs_cascade"

# Internal development validation. Cross-validation based, so a point estimate
# only -- drawn as a reference line rather than a row with an interval.
DEVELOPMENT_AUROC = 0.832

SITE_ORDER = ["mimic_temporal", "eicu_alt", "eicu_primary", "sicdb", "nwicu"]
SITE_LABELS = {
    "mimic_temporal": ("MIMIC-IV temporal", "2017-2022, n=5,732"),
    "eicu_alt": ("eICU-CRD alt", "n=13,198"),
    "eicu_primary": ("eICU-CRD primary", "n=2,367"),
    "sicdb": ("SICdb", "n=9,122"),
    "nwicu": ("NWICU", "n=3,829"),
}

LABEL_FLAT = "Multiclass model"
LABEL_CASCADE = "Death-first two-stage model"

COLOR_FLAT = "#22456F"
COLOR_CASCADE = "#C0392B"
COLOR_GRID = "#D8D8D8"
COLOR_MUTED = "#707070"

X_MIN, X_MAX = 0.712, 0.848
X_TICKS = [0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84]
MM_PER_INCH = 25.4

# Axes box as a fraction of the canvas. Fixed, because the figure is saved at
# its exact requested size rather than cropped to its ink.
LEFT_MARGIN = 0.235
BOTTOM_MARGIN = 0.255

# Absolute type sizes in points; independent of the chosen figure width.
FONT_SITE = 7.4
FONT_TICK = 7.0
FONT_AXIS = 8.0
FONT_NOTE = 6.4
FONT_LEGEND = 7.2
FONT_DELTA = 7.0
FONT_DELTA_CI = 6.4
FONT_HEADER = 6.8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the flat-versus-cascade discrimination forest plot."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_STEM,
        help="Output path stem; .png and .pdf are appended.",
    )
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--width-mm", type=float, default=160.0,
                        help="Printed width in mm.")
    parser.add_argument("--aspect", type=float, default=0.46,
                        help="Height divided by width. 1.0 is square.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--delta-column", action="store_true",
        help="Add a right-hand column with the paired difference and its 95% CI.",
    )
    parser.add_argument(
        "--title", default=None,
        help="Optional title. Omitted by default so the caption carries it.",
    )
    return parser


def draw_delta_column(annotation, frame) -> None:
    """Right-hand column carrying the paired difference and its 95% CI."""
    annotation.set_xlim(0, 1)
    annotation.set_ylim(-0.95, len(SITE_ORDER) - 0.15)
    annotation.axis("off")
    annotation.text(
        0.0, len(SITE_ORDER) - 0.42, "\u0394 Multiclass \u2212 Two-stage",
        fontsize=FONT_HEADER, fontweight="bold", va="bottom", ha="left",
    )
    for position, site in zip(list(range(len(SITE_ORDER)))[::-1], SITE_ORDER):
        row = frame.loc[site]
        significant = row["delta_ci_low"] > 0 or row["delta_ci_high"] < 0
        color = "#1A1A1A" if significant else COLOR_MUTED
        annotation.text(
            0.0, position + 0.16, f"{row['delta']:+.3f}",
            fontsize=FONT_DELTA, va="center", ha="left", color=color,
            fontweight="bold" if significant else "normal",
        )
        annotation.text(
            0.0, position - 0.19,
            f"({row['delta_ci_low']:+.3f}, {row['delta_ci_high']:+.3f})",
            fontsize=FONT_DELTA_CI, va="center", ha="left", color=color,
        )
    annotation.text(
        0.0, -0.66, "Bold: 95% CI\nexcludes zero",
        fontsize=FONT_NOTE, va="center", ha="left",
        color=COLOR_MUTED, linespacing=1.3,
    )


def main() -> int:
    arguments = build_parser().parse_args()

    width_inch = arguments.width_mm / MM_PER_INCH
    height_inch = width_inch * arguments.aspect

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42  # embed TrueType so text stays selectable
    plt.rcParams["ps.fonttype"] = 42

    frame = pd.read_csv(arguments.results)
    frame = frame.loc[frame["model"] == arguments.model].set_index("site")
    missing = [site for site in SITE_ORDER if site not in frame.index]
    if missing:
        raise SystemExit(f"[ERROR] Results are missing sites: {missing}")

    if arguments.delta_column:
        figure, (plot, annotation) = plt.subplots(
            1, 2, figsize=(width_inch, height_inch),
            gridspec_kw={"width_ratios": [1.0, 0.46], "wspace": 0.04},
        )
    else:
        figure, plot = plt.subplots(1, 1, figsize=(width_inch, height_inch))
        annotation = None

    positions = list(range(len(SITE_ORDER)))[::-1]
    offset = 0.20

    for position, site in zip(positions, SITE_ORDER):
        row = frame.loc[site]

        # Cascade interval, lighter and drawn behind.
        plot.plot(
            [row["cascade_ci_low"], row["cascade_ci_high"]],
            [position - offset, position - offset],
            color=COLOR_CASCADE, linewidth=1.3, alpha=0.55,
            solid_capstyle="butt", zorder=2,
        )
        plot.plot(
            row["cascade_macro_auroc"], position - offset,
            marker="D", markersize=4.8, color=COLOR_CASCADE,
            markerfacecolor="white", markeredgewidth=1.3, zorder=4,
        )

        # Flat interval.
        plot.plot(
            [row["flat_ci_low"], row["flat_ci_high"]],
            [position + offset, position + offset],
            color=COLOR_FLAT, linewidth=1.7, solid_capstyle="butt", zorder=3,
        )
        plot.plot(
            row["flat_macro_auroc"], position + offset,
            marker="o", markersize=6.0, color=COLOR_FLAT, zorder=5,
        )

        if position != positions[-1]:
            plot.axhline(position - 0.5, color="#EDEDED", linewidth=0.7, zorder=0)

    plot.axvline(
        DEVELOPMENT_AUROC, color="#7A7A7A", linestyle=(0, (4, 2.5)),
        linewidth=1.0, zorder=1,
    )
    plot.text(
        DEVELOPMENT_AUROC - 0.0035, len(SITE_ORDER) / 2.0 - 0.5,
        "Development validation 0.832",
        ha="center", va="center", rotation=90, fontsize=FONT_NOTE,
        color=COLOR_MUTED, style="italic",
    )

    plot.set_yticks(positions)
    plot.set_yticklabels(
        [f"{SITE_LABELS[site][0]}\n{SITE_LABELS[site][1]}" for site in SITE_ORDER],
        fontsize=FONT_SITE, linespacing=1.35,
    )
    plot.set_xlabel("Macro AUROC (95% CI)", fontsize=FONT_AXIS, labelpad=3.0)
    plot.set_xticks(X_TICKS)
    plot.set_xlim(X_MIN, X_MAX)
    plot.set_ylim(-0.72, len(SITE_ORDER) - 0.28)
    plot.tick_params(axis="x", labelsize=FONT_TICK, length=2.4, width=0.7, pad=2.0)
    plot.tick_params(axis="y", length=0, pad=2.0)
    plot.grid(axis="x", linestyle=":", linewidth=0.6, color=COLOR_GRID)
    plot.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        plot.spines[spine].set_visible(False)
    plot.spines["bottom"].set_linewidth(0.8)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="-", color=COLOR_FLAT,
                   linewidth=1.7, markersize=5.8, label=LABEL_FLAT),
        plt.Line2D([], [], marker="D", linestyle="-", color=COLOR_CASCADE,
                   linewidth=1.3, alpha=0.75, markerfacecolor="white",
                   markeredgewidth=1.3, markersize=4.8, label=LABEL_CASCADE),
    ]
    plot.legend(
        handles=handles, fontsize=FONT_LEGEND, loc="upper left",
        bbox_to_anchor=(0.0, -0.095), frameon=False, ncol=1,
        handletextpad=0.6, labelspacing=0.45, handlelength=2.0,
    )

    if annotation is not None:
        plot.set_ylim(-0.95, len(SITE_ORDER) - 0.15)
        draw_delta_column(annotation, frame)

    if arguments.title:
        figure.suptitle(arguments.title, fontsize=8.8, x=0.01, ha="left", y=0.995)

    right_margin = 0.995 if annotation is not None else 0.985
    figure.subplots_adjust(left=LEFT_MARGIN, right=right_margin, top=0.965, bottom=BOTTOM_MARGIN)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        destination = arguments.out.with_suffix(f".{suffix}")
        figure.savefig(destination, dpi=arguments.dpi, facecolor="white")
        print(f"[OK] {destination}")

    print(
        f"[INFO] printed size approx {arguments.width_mm:.0f} x "
        f"{arguments.width_mm * arguments.aspect:.0f} mm at {arguments.dpi} dpi"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
