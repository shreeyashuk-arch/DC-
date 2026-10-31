"""Channel impairments module: AWGN and Flat Rayleigh Fading models."""

from .fading import (
    RayleighChannel,
    simulate_rayleigh_bpsk,
    theoretical_rayleigh_bpsk_ber,
)

__all__ = [
    "RayleighChannel",
    "simulate_rayleigh_bpsk",
    "theoretical_rayleigh_bpsk_ber",
]
