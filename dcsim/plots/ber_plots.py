"""BER Waterfall plotting utilities overlaying simulation points with theoretical curves."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np


def plot_ber_curves(
    eb_n0_db_sim: np.ndarray,
    ber_sim: np.ndarray,
    eb_n0_db_theo: np.ndarray,
    ber_theo: np.ndarray,
    label: str = "BPSK",
    title: str = "Bit Error Rate Performance",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot simulated BER points against continuous theoretical benchmark curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=120)
    else:
        fig = ax.get_figure()

    # Theoretical continuous curve
    ax.semilogy(
        eb_n0_db_theo,
        ber_theo,
        linestyle="-",
        linewidth=2.2,
        label=f"{label} (Theory)",
        color="#2563EB",
    )

    # Simulated discrete points
    # Filter out zero errors for semilogy display
    valid_idx = ber_sim > 0.0
    if np.any(valid_idx):
        ax.semilogy(
            eb_n0_db_sim[valid_idx],
            ber_sim[valid_idx],
            linestyle="",
            marker="o",
            markersize=7,
            markerfacecolor="#EF4444",
            markeredgecolor="#B91C1C",
            markeredgewidth=1.5,
            label=f"{label} (Simulated)",
        )

    ax.set_xlabel(r"$E_b / N_0$ (dB)")
    ax.set_ylabel(r"Bit Error Rate ($P_b$)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_ylim([1e-6, 1.0])
    ax.legend(loc="upper right", frameon=True)

    return fig, ax
