"""Comprehensive Modulation Scheme Comparison, Benchmarks, and Selection Guidelines.

Owner: Student 2 (Module B: Modulation and Detection Lab)

Consolidates:
1. Spectral efficiency (bps/Hz) and bits per symbol across all 8 supported modulation schemes.
2. Required Eb/N0 in dB to achieve BER = 1e-4.
3. Multi-scheme composite BER waterfall generation.
4. Intelligent "Which Scheme to Use When" recommendation engine based on power vs bandwidth constraints.
"""

from __future__ import annotations
import numpy as np
from dcsim.core.conventions import q_function
from dcsim.core.passband_link import theoretical_ber
from dcsim.modulation.advanced_mod import theoretical_advanced_ber

ALL_SCHEMES = [
    "BPSK",
    "QPSK",
    "8-PSK",
    "16-QAM",
    "BFSK (Coherent)",
    "BFSK (Non-coherent)",
    "DBPSK",
    "OOK",
]


def get_modulation_comparison_metrics() -> list[dict[str, str | float]]:
    """Return comparative engineering metrics for all modulation schemes."""
    return [
        {
            "Scheme": "BPSK",
            "Bits/Symbol": 1,
            "Bandwidth Efficiency (bps/Hz)": 1.0,
            "Detection": "Coherent (Carrier Phase Locked)",
            "Eb/N0 for BER 1e-4 (dB)": 8.4,
            "Primary Strength": "Maximum power efficiency, simplest constellation",
            "Primary Limitation": "Low bandwidth efficiency (1 bps/Hz)",
        },
        {
            "Scheme": "QPSK",
            "Bits/Symbol": 2,
            "Bandwidth Efficiency (bps/Hz)": 2.0,
            "Detection": "Coherent (I/Q Demodulation)",
            "Eb/N0 for BER 1e-4 (dB)": 8.4,
            "Primary Strength": "Double throughput of BPSK at identical power efficiency",
            "Primary Limitation": "Requires precise quadrature carrier tracking",
        },
        {
            "Scheme": "8-PSK",
            "Bits/Symbol": 3,
            "Bandwidth Efficiency (bps/Hz)": 3.0,
            "Detection": "Coherent",
            "Eb/N0 for BER 1e-4 (dB)": 11.8,
            "Primary Strength": "Higher spectral efficiency, constant envelope (amplifier friendly)",
            "Primary Limitation": "3.4 dB power penalty over QPSK; sensitive to phase noise",
        },
        {
            "Scheme": "16-QAM",
            "Bits/Symbol": 4,
            "Bandwidth Efficiency (bps/Hz)": 4.0,
            "Detection": "Coherent (I/Q Slicing)",
            "Eb/N0 for BER 1e-4 (dB)": 12.2,
            "Primary Strength": "High spectral efficiency (4 bps/Hz), square decision regions",
            "Primary Limitation": "Non-constant envelope (high PAPR, requires linear PA)",
        },
        {
            "Scheme": "BFSK (Coherent)",
            "Bits/Symbol": 1,
            "Bandwidth Efficiency (bps/Hz)": 0.5,
            "Detection": "Coherent Correlator",
            "Eb/N0 for BER 1e-4 (dB)": 11.4,
            "Primary Strength": "Constant envelope, simple frequency synthesis",
            "Primary Limitation": "3 dB poorer power efficiency than BPSK; wide bandwidth",
        },
        {
            "Scheme": "BFSK (Non-coherent)",
            "Bits/Symbol": 1,
            "Bandwidth Efficiency (bps/Hz)": 0.5,
            "Detection": "Envelope / Energy Detector",
            "Eb/N0 for BER 1e-4 (dB)": 12.3,
            "Primary Strength": "No carrier phase synchronization required whatsoever",
            "Primary Limitation": "3.9 dB poorer than BPSK; low spectral efficiency",
        },
        {
            "Scheme": "DBPSK",
            "Bits/Symbol": 1,
            "Bandwidth Efficiency (bps/Hz)": 1.0,
            "Detection": "Differential / Delay Product",
            "Eb/N0 for BER 1e-4 (dB)": 9.3,
            "Primary Strength": "Resistant to carrier phase ambiguity and slow Doppler shift",
            "Primary Limitation": "0.9 dB penalty over BPSK due to noisy reference symbol",
        },
        {
            "Scheme": "OOK",
            "Bits/Symbol": 1,
            "Bandwidth Efficiency (bps/Hz)": 1.0,
            "Detection": "Coherent / Threshold Slicing",
            "Eb/N0 for BER 1e-4 (dB)": 11.4,
            "Primary Strength": "Ultra-low cost transmitter (on/off keying)",
            "Primary Limitation": "Requires adaptive threshold under fading; 3 dB penalty vs BPSK",
        },
    ]


def evaluate_all_schemes_ber(eb_n0_db_range: np.ndarray) -> dict[str, np.ndarray]:
    """Compute theoretical BER waterfall curves across an SNR range for all schemes."""
    db = np.asarray(eb_n0_db_range, dtype=np.float64)
    results: dict[str, np.ndarray] = {
        "BPSK": theoretical_ber("BPSK", db),
        "QPSK": theoretical_ber("QPSK", db),
        "8-PSK": theoretical_advanced_ber("8-PSK", db),
        "16-QAM": theoretical_advanced_ber("16-QAM", db),
        "BFSK (Coherent)": theoretical_ber("BFSK", db),
        "BFSK (Non-coherent)": theoretical_advanced_ber("Non-coherent BFSK", db),
        "DBPSK": theoretical_advanced_ber("DBPSK", db),
        "OOK": theoretical_ber("OOK", db),
    }
    return results


def get_scheme_recommendation(
    bandwidth_limited: bool,
    power_limited: bool,
    phase_jitter_severe: bool,
    constant_envelope_required: bool,
) -> dict[str, str]:
    """Recommend optimal modulation scheme based on physical link constraints."""
    if phase_jitter_severe:
        if bandwidth_limited:
            rec = "DBPSK"
            reason = "Eliminates carrier phase tracking failures while preserving 1 bps/Hz bandwidth efficiency."
        else:
            rec = "BFSK (Non-coherent)"
            reason = "Operates with pure envelope detection, completely immune to carrier phase offsets and rapid Doppler shifts."
    elif bandwidth_limited:
        if constant_envelope_required:
            rec = "8-PSK"
            reason = "Achieves 3 bps/Hz spectral efficiency with a strictly constant envelope, permitting non-linear high-efficiency RF power amplifiers (e.g. satellite transponders)."
        else:
            rec = "16-QAM"
            reason = "Maximizes throughput (4 bps/Hz) within tight spectrum allocations; ideal for terrestrial microwave links, cable modems, and cellular basebands."
    elif power_limited:
        rec = "BPSK / QPSK"
        reason = "Delivers the highest power efficiency of all linear modulations (BER = 1e-4 at Eb/N0 = 8.4 dB). QPSK is favored as it achieves double the bit rate in the identical RF bandwidth."
    else:
        rec = "QPSK"
        reason = "The benchmark sweet spot of digital communications: optimal power efficiency (8.4 dB @ 1e-4) paired with 2.0 bps/Hz spectral efficiency."

    return {
        "recommended_scheme": rec,
        "engineering_justification": reason,
    }
