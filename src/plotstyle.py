"""Shared figure styling.

One place defines the palette, type scale and chrome so every figure in the
report and the slide deck reads as one system.  The categorical hues are used in
a fixed slot order and are never cycled or reassigned by rank, so a series keeps
its colour across figures: hindsight relabeling is blue everywhere, its control
is orange everywhere.

The palette was checked for colour-vision deficiency separation rather than
chosen by eye (worst adjacent pair dE 9.1 under protanopia, 22.9 normal vision).
Two of the hues fall below 3:1 contrast on the light surface, so every figure
also carries direct labels or an accompanying table — identity is never conveyed
by colour alone.
"""

from __future__ import annotations

import matplotlib as mpl

# --- surface and ink ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e8e7e4"

# --- categorical slots, in fixed order --------------------------------------
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Stable colour per condition, so a reader who learns a colour in one figure
# keeps it in every other.
CONDITION_COLOR = {
    "her_sparse": SERIES[0],
    "her_sparse_hard": SERIES[0],
    "noher_sparse": SERIES[1],
    "noher_sparse_hard": SERIES[1],
    "her_dense": SERIES[2],
    "noher_dense": SERIES[3],
    "her_sparse_dr": SERIES[6],
    "scripted": INK_MUTED,
    "random": INK_MUTED,
}

CONDITION_LABEL = {
    "her_sparse": "SAC + HER (sparse)",
    "noher_sparse": "SAC, no HER (sparse)",
    "her_dense": "SAC + HER (dense)",
    "noher_dense": "SAC, no HER (dense)",
    "her_sparse_dr": "SAC + HER + domain randomization",
    "her_sparse_hard": "SAC + HER (sparse, hard task)",
    "noher_sparse_hard": "SAC, no HER (sparse, hard task)",
    "scripted": "Scripted controller (reference)",
    "random": "Random policy (floor)",
}


def apply_style() -> None:
    """Install the house style as matplotlib defaults."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "figure.dpi": 110,

        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,

        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,

        # Recessive chrome: solid hairlines one shade off the surface, no dashes.
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.autolayout": False,
    })


def finish(ax, title: str | None = None, subtitle: str | None = None,
           xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Apply title/subtitle/labels consistently, with the subtitle as quiet ink."""
    if title:
        ax.set_title(title, loc="left", pad=18 if subtitle else 10, fontweight="600")
    if subtitle:
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=INK_SECONDARY, va="bottom", ha="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def label_bars(ax, bars, values, fmt: str = "{:.2f}", offset: float = 0.012) -> None:
    """Direct-label bars in text ink, never in the series colour."""
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                fmt.format(value), ha="center", va="bottom",
                fontsize=9, color=INK_SECONDARY)
