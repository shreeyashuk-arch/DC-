"""Eye diagram and constellation rendering utilities."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np


def plot_eye_diagram(
    t_trace: np.ndarray,
    traces: np.ndarray,
    title: str = "Eye Diagram",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multi-trace overlapping eye diagram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    else:
        fig = ax.get_figure()

    for row in traces:
        ax.plot(t_trace, row, color="#0284C7", alpha=0.3, linewidth=1.2)

    ax.axvline(0.0, color="#DC2626", linestyle=":", linewidth=1.5, label="Optimal Sampling Instant")
    ax.axhline(0.0, color="#4B5563", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Normalized Time (Symbol Periods)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    return fig, ax


def plot_constellation(
    points: np.ndarray,
    title: str = "Signal-Space Constellation Diagram",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot 2D complex signal space constellation."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    else:
        fig = ax.get_figure()

    real_part = np.real(points)
    imag_part = np.imag(points)

    ax.scatter(real_part, imag_part, color="#2563EB", alpha=0.5, s=25, edgecolors="none", label="Received Symbols")
    ax.axhline(0, color="#9CA3AF", linestyle="--", linewidth=1.0)
    ax.axvline(0, color="#9CA3AF", linestyle="--", linewidth=1.0)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    return fig, ax
