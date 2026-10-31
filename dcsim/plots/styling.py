"""Publication and presentation plotting styles and Matplotlib theme helpers."""

from __future__ import annotations
import matplotlib.pyplot as plt


def apply_lab_style(dark_mode: bool = False) -> None:
    """Configure Matplotlib defaults for crisp presentation and report figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.autolayout": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })

    if dark_mode:
        plt.style.use("dark_background")
        plt.rcParams["axes.edgecolor"] = "#4B5563"
        plt.rcParams["grid.color"] = "#374151"
    else:
        plt.rcParams["figure.facecolor"] = "#FFFFFF"
        plt.rcParams["axes.facecolor"] = "#F8FAFC"
        plt.rcParams["axes.edgecolor"] = "#CBD5E1"
        plt.rcParams["grid.color"] = "#E2E8F0"
