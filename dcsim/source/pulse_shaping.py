"""Pulse Shaping, Root-Raised-Cosine (RRC) Matched Filtering, Eye Diagrams, and ISI Analysis.

Owner: Student 1 (Module A: Source and Baseband Lab)

Covers:
- Analytical discrete RRC filter design with roll-off factor alpha in [0.0, 1.0].
- Strict unit-energy normalization: sum(|h|^2) = 1.0.
- TX and RX matched RRC filter pair producing zero-ISI Nyquist Raised Cosine response.
- Eye diagram extraction and timing jitter / noise margin analysis.
- ISI demonstration: matched RRC vs poorly band-limited / rectangular filtering.
- AWGN BER simulation with matched RRC filtering verifying theoretical Q(sqrt(2Eb/N0)).
"""

from __future__ import annotations
import numpy as np
import scipy.signal
from dcsim.core.conventions import q_function, add_awgn_noise, calculate_ber, measure_eb


def rrc_filter_taps(
    alpha: float,
    samples_per_symbol: int,
    num_symbols_span: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Design discrete Root-Raised Cosine (RRC) filter impulse response.

    Strictly normalized to unit energy: sum(h^2) == 1.0.

    Parameters
    ----------
    alpha : float
        Roll-off factor in [0.0, 1.0].
    samples_per_symbol : int
        Oversampling factor (sps).
    num_symbols_span : int
        Filter span in symbols on each side (total symbols = 2 * num_symbols_span + 1).

    Returns
    -------
    t_symbols : np.ndarray
        Time axis in symbol durations T.
    h : np.ndarray
        Unit-energy normalized RRC filter impulse response.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Roll-off factor alpha must be in [0.0, 1.0].")

    sps = samples_per_symbol
    total_samples = 2 * num_symbols_span * sps + 1
    t = np.arange(-num_symbols_span * sps, num_symbols_span * sps + 1) / float(sps)

    h = np.zeros(total_samples, dtype=np.float64)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif alpha > 0.0 and np.isclose(abs(ti), 1.0 / (4.0 * alpha), atol=1e-6):
            # Special limit for singularity at t = +/- T / (4*alpha)
            val1 = (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
            val2 = (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            h[i] = (alpha / np.sqrt(2.0)) * (val1 + val2)
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            denom = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / denom

    # Enforce unit energy constraint
    energy = np.sum(h**2)
    h_unit = h / np.sqrt(energy)
    return t, h_unit


def rc_filter_taps(
    alpha: float,
    samples_per_symbol: int,
    num_symbols_span: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Design discrete full Raised Cosine (RC) filter impulse response.

    Formula:
        g(t) = sinc(t/T) * cos(pi * alpha * t/T) / (1 - (2*alpha*t/T)^2)
    """
    sps = samples_per_symbol
    total_samples = 2 * num_symbols_span * sps + 1
    t = np.arange(-num_symbols_span * sps, num_symbols_span * sps + 1) / float(sps)
    g = np.zeros(total_samples, dtype=np.float64)

    for i, ti in enumerate(t):
        if ti == 0.0:
            g[i] = 1.0
        elif alpha > 0.0 and np.isclose(abs(ti), 1.0 / (2.0 * alpha), atol=1e-6):
            # L'Hopital's limit at singularity
            g[i] = (np.pi / 4.0) * np.sin(np.pi / (2.0 * alpha))
        else:
            sinc_val = np.sin(np.pi * ti) / (np.pi * ti)
            num = sinc_val * np.cos(np.pi * alpha * ti)
            denom = 1.0 - (2.0 * alpha * ti) ** 2
            g[i] = num / denom

    # Normalize peak to 1.0 for zero-ISI reference
    g_norm = g / np.max(np.abs(g))
    return t, g_norm


class MatchedRRCFilter:
    """Implements paired TX pulse shaping and RX matched filtering."""

    def __init__(self, alpha: float = 0.35, sps: int = 8, span: int = 6) -> None:
        self.alpha = alpha
        self.sps = sps
        self.span = span
        _, self.h = rrc_filter_taps(alpha, sps, span)
        self.filter_delay = span * sps

    def tx_pulse_shape(self, symbols: np.ndarray) -> np.ndarray:
        """Upsample symbols and convolve with TX RRC filter."""
        # Upsample by sps
        upsampled = np.zeros(len(symbols) * self.sps, dtype=np.float64)
        upsampled[:: self.sps] = symbols
        # Convolve with TX RRC
        return np.convolve(upsampled, self.h, mode="full")

    def rx_matched_filter(self, received_signal: np.ndarray) -> np.ndarray:
        """Convolve received signal with matched RX RRC filter."""
        return np.convolve(received_signal, self.h, mode="full")

    def sample_matched_output(self, rx_filtered: np.ndarray, num_symbols: int) -> np.ndarray:
        """Extract symbol-rate samples at the zero-ISI optimum timing instances."""
        # The cascade of TX and RX filters incurs 2 * filter_delay
        total_delay = 2 * self.filter_delay
        indices = total_delay + np.arange(num_symbols) * self.sps
        valid_indices = indices[indices < len(rx_filtered)]
        return rx_filtered[valid_indices]


def generate_eye_diagram_data(
    signal: np.ndarray,
    samples_per_symbol: int,
    traces_span_symbols: int = 2,
    num_traces: int = 100,
    discard_initial_symbols: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract overlapping multi-symbol segments for rendering an eye diagram.

    Parameters
    ----------
    signal : np.ndarray
        Continuous filtered discrete signal.
    samples_per_symbol : int
        Oversampling factor (sps).
    traces_span_symbols : int
        Number of symbol durations spanned per trace (default: 2 symbols).
    num_traces : int
        Maximum number of overlapping traces to extract.
    discard_initial_symbols : int
        Initial symbols to skip past transient filter startup.

    Returns
    -------
    t_trace : np.ndarray
        Time axis across traces_span_symbols normalized to [-1, +1] or [0, span].
    traces : np.ndarray
        2D array of shape (num_traces, trace_samples).
    """
    sps = samples_per_symbol
    trace_len = traces_span_symbols * sps
    t_trace = np.linspace(-float(traces_span_symbols) / 2.0, float(traces_span_symbols) / 2.0, trace_len)

    start_idx = discard_initial_symbols * sps
    available_samples = len(signal) - start_idx - trace_len
    if available_samples <= 0:
        return t_trace, np.empty((0, trace_len))

    # Step by 1 symbol per trace
    step = sps
    possible_traces = available_samples // step
    actual_traces = min(num_traces, possible_traces)

    traces = np.zeros((actual_traces, trace_len), dtype=np.float64)
    for i in range(actual_traces):
        idx = start_idx + i * step
        traces[i, :] = signal[idx : idx + trace_len]

    return t_trace, traces


def simulate_rrc_link(
    num_bits: int,
    eb_n0_db: float,
    alpha: float = 0.35,
    sps: int = 8,
    span: int = 6,
    poor_bandlimit: bool = False,
    rng: np.random.Generator | None = None,
) -> dict[str, float | np.ndarray]:
    """Simulate BPSK transmission with RRC pulse shaping through AWGN channel.

    Demonstrates that matched RRC filtering achieves the theoretical AWGN BER,
    while a poorly band-limited filter induces severe ISI and eye closure.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    bits = rng.integers(0, 2, size=num_bits, dtype=np.int8)
    symbols = np.where(bits == 1, 1.0, -1.0)

    matched_filter = MatchedRRCFilter(alpha=alpha, sps=sps, span=span)
    tx_wave = matched_filter.tx_pulse_shape(symbols)

    if poor_bandlimit:
        # Pass through narrow 4th-order lowpass filter violating Nyquist criterion
        sos = scipy.signal.butter(4, 0.4 / sps, output="sos")
        tx_wave = scipy.signal.sosfilt(sos, tx_wave)

    # Add AWGN noise according to global convention
    noisy_wave, eb, n0 = add_awgn_noise(tx_wave, eb_n0_db=eb_n0_db, num_info_bits=num_bits, rng=rng)

    # Matched RX filtering
    rx_filtered = matched_filter.rx_matched_filter(noisy_wave)

    # Sample at optimal zero-ISI points
    sampled_symbols = matched_filter.sample_matched_output(rx_filtered, num_symbols=num_bits)

    # Threshold at 0.0
    rx_bits = (sampled_symbols > 0.0).astype(np.int8)
    ber, errors, _ = calculate_ber(bits[: len(rx_bits)], rx_bits)
    ber_theo = float(q_function(np.sqrt(2.0 * (10.0 ** (eb_n0_db / 10.0)))))

    # Extract eye diagram segments
    t_eye, eye_traces = generate_eye_diagram_data(rx_filtered, samples_per_symbol=sps)

    return {
        "ber": ber,
        "ber_theoretical": ber_theo,
        "errors": errors,
        "num_bits": len(rx_bits),
        "sampled_symbols": sampled_symbols,
        "tx_waveform": tx_wave,
        "rx_filtered": rx_filtered,
        "t_eye": t_eye,
        "eye_traces": eye_traces,
    }
