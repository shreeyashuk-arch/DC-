"""Publication Figure Generator for College Project Lab Report and Presentations.

Executes real simulations and saves high-resolution vector and PNG figures to outputs/figures/.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dcsim.plots.styling import apply_lab_style
from dcsim.core.conventions import q_function
from dcsim.core.passband_link import simulate_passband_link, theoretical_ber
from dcsim.source.pcm import (
    PCMCodec,
    generate_synthetic_tone,
    theoretical_sqnr_sine,
    demonstrate_aliasing,
)
from dcsim.source.line_coding import encode_line_code, compute_line_code_psd
from dcsim.source.pulse_shaping import (
    rrc_filter_taps,
    rc_filter_taps,
    generate_eye_diagram_data,
    simulate_rrc_link,
)
from dcsim.modulation.advanced_mod import (
    simulate_advanced_mod,
    theoretical_advanced_ber,
)
from dcsim.modulation.signal_space import (
    gram_schmidt_orthogonalize,
    get_signal_space_preset,
)
from dcsim.modulation.phase_offset import (
    simulate_phase_offset_impact,
    theoretical_bpsk_phase_offset_ber,
)
from dcsim.modulation.comparison import evaluate_all_schemes_ber
from dcsim.channel.fading import simulate_rayleigh_bpsk, theoretical_rayleigh_bpsk_ber
from dcsim.coding.system import simulate_coded_bpsk_link
from dcsim.transceiver import TransceiverPipeline

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_fig1_pcm_sqnr():
    """Figure 1: PCM SQNR vs bit depth & Companding dynamic range advantage."""
    print("Generating Figure 1: PCM SQNR & Companding...")
    apply_lab_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Full-scale sine SQNR vs bit depth
    n_bits_arr = np.arange(2, 11)
    meas_sqnr = []
    theo_sqnr = []

    _, tone = generate_synthetic_tone(freq_hz=103.5, duration_s=0.5, sampling_rate_hz=10000.0, amplitude=1.0)
    for n in n_bits_arr:
        codec = PCMCodec(n_bits=n, companding=False)
        res = codec.encode_and_decode(tone, v_max=1.0)
        meas_sqnr.append(res.measured_sqnr_db)
        theo_sqnr.append(theoretical_sqnr_sine(n))

    ax1.plot(n_bits_arr, theo_sqnr, "b-", linewidth=2.0, label=r"Theory: $6.02n + 1.76$ dB")
    ax1.plot(n_bits_arr, meas_sqnr, "ro", markersize=7, label="Measured (Uniform)")
    ax1.set_xlabel("Quantization Bits ($n$)")
    ax1.set_ylabel("SQNR (dB)")
    ax1.set_title("Uniform Quantizer SQNR vs. Bit Depth")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    # 2. Dynamic range: SQNR vs input amplitude (Uniform vs mu-law)
    amplitudes = np.logspace(-2.5, 0.0, 15)  # From ~0.003 to 1.0 (-50 dB to 0 dB)
    sqnr_uniform = []
    sqnr_companded = []

    codec_uni = PCMCodec(n_bits=8, companding=False)
    codec_comp = PCMCodec(n_bits=8, companding=True, mu=255.0)

    for amp in amplitudes:
        _, a_tone = generate_synthetic_tone(freq_hz=103.5, duration_s=0.3, sampling_rate_hz=10000.0, amplitude=amp)
        r_u = codec_uni.encode_and_decode(a_tone, v_max=1.0)
        r_c = codec_comp.encode_and_decode(a_tone, v_max=1.0)
        sqnr_uniform.append(r_u.measured_sqnr_db)
        sqnr_companded.append(r_c.measured_sqnr_db)

    amp_db = 20.0 * np.log10(amplitudes)
    ax2.plot(amp_db, sqnr_companded, "g-^", linewidth=2.0, label=r"$\mu$-Law Companded ($\mu=255$)")
    ax2.plot(amp_db, sqnr_uniform, "r--s", linewidth=2.0, label="Uniform Quantizer")
    ax2.set_xlabel("Input Signal Amplitude (dB relative to Full Scale)")
    ax2.set_ylabel("Measured SQNR (dB)")
    ax2.set_title(r"Dynamic Range: Uniform vs $\mu$-Law Companding ($n=8$)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_pcm_sqnr.png")
    plt.close(fig)


def generate_fig2_line_codes():
    """Figure 2: Time-domain line code waveforms and Welch PSD curves."""
    print("Generating Figure 2: Line Codes Waveforms & Spectra...")
    apply_lab_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150)

    test_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=np.int8)
    schemes = ["NRZ-L", "RZ", "Manchester", "AMI"]
    colors = ["#2563EB", "#DC2626", "#059669", "#7C3AED"]

    for i, sc in enumerate(schemes):
        t, wave = encode_line_code(test_bits, sc, samples_per_bit=32, voltage=1.0)  # type: ignore
        offset = (3 - i) * 2.5
        ax1.plot(t, wave + offset, color=colors[i], linewidth=1.8, label=sc)
        ax1.axhline(offset, color="#9CA3AF", linestyle=":", linewidth=0.8)

    ax1.set_xlabel("Time (Normalized to Bit Duration $T_b$)")
    ax1.set_ylabel("Offset Waveforms")
    ax1.set_title(f"Line Coding Waveforms for Bit Pattern: {list(test_bits)}")
    ax1.set_xlim([0, len(test_bits)])
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper right")

    # Welch PSD comparison
    for i, sc in enumerate(schemes):
        freqs, psd = compute_line_code_psd(sc, num_bits=10000, samples_per_bit=32)  # type: ignore
        # Plot up to 3*Rb
        valid = freqs <= 3.0
        ax2.plot(freqs[valid], 10.0 * np.log10(psd[valid] + 1e-12), color=colors[i], linewidth=2.0, label=sc)

    ax2.set_xlabel(r"Normalized Frequency ($f / R_b$)")
    ax2.set_ylabel("Power Spectral Density (dB/Hz)")
    ax2.set_title("Power Spectral Density (Welch Periodogram)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_line_codes.png")
    plt.close(fig)


def generate_fig3_rrc_eye_and_isi():
    """Figure 3: Matched RRC Eye Diagram (Zero ISI) vs Poorly Band-limited (High ISI)."""
    print("Generating Figure 3: RRC Eye Diagram & ISI...")
    apply_lab_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Matched RRC
    res_clean = simulate_rrc_link(num_bits=2000, eb_n0_db=25.0, alpha=0.35, sps=8, span=6, poor_bandlimit=False)
    t_eye = res_clean["t_eye"]
    traces_clean = res_clean["eye_traces"]

    for tr in traces_clean[:80]:
        ax1.plot(t_eye, tr, color="#0284C7", alpha=0.25, linewidth=1.1)
    ax1.axvline(0.0, color="#DC2626", linestyle=":", linewidth=1.5, label="Sampling Instant")
    ax1.set_xlabel("Time (Symbol Periods $T_s$)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(r"Matched RRC Eye Diagram ($\alpha = 0.35$, Open Eye)")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend()

    # 2. Poorly Bandlimited ISI
    res_isi = simulate_rrc_link(num_bits=2000, eb_n0_db=25.0, alpha=0.35, sps=8, span=6, poor_bandlimit=True)
    traces_isi = res_isi["eye_traces"]

    for tr in traces_isi[:80]:
        ax2.plot(t_eye, tr, color="#DC2626", alpha=0.25, linewidth=1.1)
    ax2.axvline(0.0, color="#1E293B", linestyle=":", linewidth=1.5, label="Sampling Instant")
    ax2.set_xlabel("Time (Symbol Periods $T_s$)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Violated Nyquist Bandwidth (Severe ISI & Closed Eye)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_rrc_eye_and_isi.png")
    plt.close(fig)


def generate_fig4_passband_ber():
    """Figure 4: Core Passband Link BER (BPSK, QPSK, BFSK, OOK)."""
    print("Generating Figure 4: Core Passband BER Waterfall...")
    apply_lab_style()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    eb_n0_db_sim = np.arange(0.0, 9.0, 1.5)
    eb_n0_db_theo = np.linspace(0.0, 9.0, 100)

    schemes = ["BPSK", "QPSK", "BFSK", "OOK"]
    colors = ["#2563EB", "#059669", "#DC2626", "#D97706"]
    markers = ["o", "s", "^", "D"]

    rng = np.random.default_rng(42)

    for i, sc in enumerate(schemes):
        # Theoretical curve
        theo_ber = theoretical_ber(sc, eb_n0_db_theo)  # type: ignore
        ax.semilogy(eb_n0_db_theo, theo_ber, color=colors[i], linestyle="-", linewidth=2.0, label=f"{sc} (Theory)")

        # Simulated points
        sim_bers = []
        for snr in eb_n0_db_sim:
            bits = rng.integers(0, 2, size=50000, dtype=np.int8)
            ber, _, _, _, _ = simulate_passband_link(bits, sc, snr, samples_per_bit=16, carrier_cycles_per_bit=2, rng=rng)  # type: ignore
            sim_bers.append(ber)
        sim_bers = np.array(sim_bers)
        valid = sim_bers > 0
        ax.semilogy(eb_n0_db_sim[valid], sim_bers[valid], color=colors[i], marker=markers[i], linestyle="", markersize=7, label=f"{sc} (Sim)")

    ax.set_xlabel(r"$E_b / N_0$ (dB)")
    ax.set_ylabel(r"Bit Error Rate ($P_b$)")
    ax.set_title("Core Passband Link BER vs. Theory (Correlator Receivers)")
    ax.set_ylim([1e-5, 0.5])
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_passband_ber.png")
    plt.close(fig)


def generate_fig5_advanced_constellations():
    """Figure 5: Complex Signal Space Constellations for 8-PSK, 16-QAM, DBPSK, and BFSK."""
    print("Generating Figure 5: Advanced Constellations...")
    apply_lab_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=150)

    cases = [
        ("8-PSK", 14.0, axes[0, 0]),
        ("16-QAM", 16.0, axes[0, 1]),
        ("DBPSK", 12.0, axes[1, 0]),
        ("Non-coherent BFSK", 14.0, axes[1, 1]),
    ]
    rng = np.random.default_rng(2026)

    for sc, snr, ax in cases:
        bits = rng.integers(0, 2, size=2400, dtype=np.int8)
        _, _, syms, _, _ = simulate_advanced_mod(bits, sc, snr, rng=rng)  # type: ignore
        real = np.real(syms[:600])
        imag = np.imag(syms[:600])

        ax.scatter(real, imag, color="#2563EB", alpha=0.6, s=20)
        ax.axhline(0, color="#9CA3AF", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="#9CA3AF", linestyle="--", linewidth=0.8)
        ax.set_title(f"{sc} Constellation ({snr} dB)")
        ax.set_xlabel("In-Phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_advanced_constellations.png")
    plt.close(fig)


def generate_fig6_phase_offset():
    """Figure 6: Carrier Phase Offset BER Degradation and Constellation Rotation."""
    print("Generating Figure 6: Phase Offset Degradation...")
    apply_lab_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Constellation under 30 deg rotation
    res_rot = simulate_phase_offset_impact("BPSK", eb_n0_db=8.0, phi_deg=30.0, num_bits=1500)
    rx_s = res_rot["rx_symbols"]
    ax1.scatter(np.real(rx_s), np.imag(rx_s), color="#DC2626", alpha=0.5, s=20, label=r"Rotated $\phi = +30^\circ$")
    ax1.axvline(0, color="#1E293B", linestyle="-", linewidth=1.5, label="Decision Boundary")
    ax1.axhline(0, color="#9CA3AF", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("In-Phase (I)")
    ax1.set_ylabel("Quadrature (Q)")
    ax1.set_title(r"BPSK Constellation Rotation ($\phi = 30^\circ$)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    # 2. BER vs Phase Error Phi
    phi_deg_sweep = np.linspace(0.0, 90.0, 19)
    phi_rad_sweep = np.radians(phi_deg_sweep)
    eb_n0_val = 6.0
    theo_curve = theoretical_bpsk_phase_offset_ber(eb_n0_val, phi_rad_sweep)

    sim_curve = []
    for deg in phi_deg_sweep:
        r = simulate_phase_offset_impact("BPSK", eb_n0_db=eb_n0_val, phi_deg=deg, num_bits=40000)
        sim_curve.append(r["ber_sim"])

    ax2.plot(phi_deg_sweep, theo_curve, "b-", linewidth=2.0, label=r"Theory: $Q(\sqrt{2E_b/N_0}\cos\phi)$")
    ax2.plot(phi_deg_sweep, sim_curve, "ro", markersize=6, label="Simulated")
    ax2.set_xlabel(r"Carrier Phase Error $\phi$ (degrees)")
    ax2.set_ylabel(r"Bit Error Rate ($P_b$)")
    ax2.set_title(r"BPSK BER Degradation vs. Phase Error ($E_b/N_0 = 6$ dB)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_phase_offset_impact.png")
    plt.close(fig)


def generate_fig7_gram_schmidt():
    """Figure 7: Gram-Schmidt Orthonormal Bases and Coordinates."""
    print("Generating Figure 7: Gram-Schmidt Analysis...")
    apply_lab_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Preset QPSK
    signals, labels = get_signal_space_preset("QPSK", N=128)
    res = gram_schmidt_orthogonalize(signals)
    t = np.linspace(0, 1, 128)

    # Plot Basis functions
    ax1.plot(t, res.basis_functions[0], "b-", linewidth=2.0, label=r"$\phi_1(t)$ (In-Phase)")
    ax1.plot(t, res.basis_functions[1], "r--", linewidth=2.0, label=r"$\phi_2(t)$ (Quadrature)")
    ax1.set_xlabel("Time (Normalized to Symbol Duration)")
    ax1.set_ylabel("Basis Amplitude")
    ax1.set_title(r"Gram-Schmidt Orthonormal Basis $\{\phi_1(t), \phi_2(t)\}$")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    # Plot signal coordinates in 2D space
    coords = res.coordinates
    ax2.scatter(coords[:, 0], coords[:, 1], color="#2563EB", s=100, zorder=5)
    for i, txt in enumerate(labels):
        ax2.annotate(f"s{i+1}", (coords[i, 0] + 0.05, coords[i, 1] + 0.05), fontsize=11, weight="bold")
    ax2.axhline(0, color="#9CA3AF", linestyle="--")
    ax2.axvline(0, color="#9CA3AF", linestyle="--")
    ax2.set_xlabel(r"Coordinate $s_{i1} = \langle s_i, \phi_1 \rangle$")
    ax2.set_ylabel(r"Coordinate $s_{i2} = \langle s_i, \phi_2 \rangle$")
    ax2.set_title("2D Signal-Space Constellation Coordinates")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_gram_schmidt.png")
    plt.close(fig)


def generate_fig8_all_schemes():
    """Figure 8: Composite Waterfall BER Curve comparing all 8 schemes."""
    print("Generating Figure 8: All Schemes Waterfall...")
    apply_lab_style()
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)

    eb_n0_db_range = np.linspace(0.0, 14.0, 140)
    all_curves = evaluate_all_schemes_ber(eb_n0_db_range)

    colors = {
        "BPSK": "#1D4ED8",
        "QPSK": "#059669",
        "8-PSK": "#D97706",
        "16-QAM": "#7C3AED",
        "BFSK (Coherent)": "#DC2626",
        "BFSK (Non-coherent)": "#EA580C",
        "DBPSK": "#0284C7",
        "OOK": "#9333EA",
    }
    linestyles = {
        "BPSK": "-",
        "QPSK": "--",
        "8-PSK": "-.",
        "16-QAM": "-",
        "BFSK (Coherent)": "-",
        "BFSK (Non-coherent)": ":",
        "DBPSK": "--",
        "OOK": "-.",
    }

    for name, ber in all_curves.items():
        ax.semilogy(eb_n0_db_range, ber, label=name, color=colors.get(name, "#000000"), linestyle=linestyles.get(name, "-"), linewidth=2.0)

    ax.set_xlabel(r"$E_b / N_0$ (dB)")
    ax.set_ylabel(r"Bit Error Rate ($P_b$)")
    ax.set_title("All Modulation Schemes Composite BER Waterfall")
    ax.set_ylim([1e-6, 0.5])
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_all_schemes_waterfall.png")
    plt.close(fig)


def generate_fig9_awgn_vs_rayleigh():
    """Figure 9: AWGN vs Flat Rayleigh Fading Performance Gap."""
    print("Generating Figure 9: AWGN vs Rayleigh Fading Gap...")
    apply_lab_style()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    snr_theo = np.linspace(0.0, 26.0, 150)
    theo_awgn = q_function(np.sqrt(2.0 * (10.0 ** (snr_theo / 10.0))))
    theo_ray = theoretical_rayleigh_bpsk_ber(snr_theo)

    ax.semilogy(snr_theo, theo_awgn, "b-", linewidth=2.2, label="BPSK in AWGN Channel")
    ax.semilogy(snr_theo, theo_ray, "r-", linewidth=2.2, label="BPSK in Flat Rayleigh Fading (Theory)")

    # Simulate Rayleigh points
    snr_sim = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    sim_ray = []
    rng = np.random.default_rng(42)
    for s in snr_sim:
        r = simulate_rayleigh_bpsk(num_bits=80000, eb_n0_db=s, rng=rng)
        sim_ray.append(r["ber_sim"])
    ax.semilogy(snr_sim, sim_ray, "ro", markersize=7, label="Rayleigh Fading (Simulated)")

    # Draw horizontal arrow at BER = 1e-3 showing 17 dB gap
    ax.axhline(1e-3, color="#6B7280", linestyle=":", linewidth=1.2)
    ax.annotate(
        r"17 dB Fading Penalty at $P_b = 10^{-3}$",
        xy=(7.0, 1e-3),
        xytext=(10.0, 3e-3),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.0, headwidth=6),
        fontsize=11,
        weight="semibold",
    )

    ax.set_xlabel(r"Average $E_b / N_0$ (dB)")
    ax.set_ylabel(r"Bit Error Rate ($P_b$)")
    ax.set_title("Performance Gap: AWGN Channel vs. Flat Rayleigh Fading")
    ax.set_ylim([1e-5, 0.5])
    ax.set_xlim([0.0, 26.0])
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig9_awgn_vs_rayleigh.png")
    plt.close(fig)


def generate_fig10_fec_coding_gain():
    """Figure 10: Coded vs Uncoded BPSK Performance showing Coding Gain."""
    print("Generating Figure 10: FEC Coding Gain...")
    apply_lab_style()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    snr_range = np.arange(2.0, 9.5, 1.0)
    uncoded_bers = []
    hamming_bers = []
    viterbi_bers = []

    rng = np.random.default_rng(2026)
    bits = rng.integers(0, 2, size=50000, dtype=np.int8)

    for s in snr_range:
        r_u = simulate_coded_bpsk_link(bits, "None", s, rng=rng)
        r_h = simulate_coded_bpsk_link(bits, "Hamming(7,4)", s, rng=rng)
        r_v = simulate_coded_bpsk_link(bits, "Convolutional(1/2, K=3)", s, rng=rng)
        uncoded_bers.append(r_u["post_decode_ber"])
        hamming_bers.append(r_h["post_decode_ber"])
        viterbi_bers.append(r_v["post_decode_ber"])

    uncoded_bers = np.array(uncoded_bers)
    hamming_bers = np.array(hamming_bers)
    viterbi_bers = np.array(viterbi_bers)

    # Plot continuous theoretical uncoded BPSK
    snr_cont = np.linspace(2.0, 9.5, 100)
    theo_uncoded = q_function(np.sqrt(2.0 * (10.0 ** (snr_cont / 10.0))))
    ax.semilogy(snr_cont, theo_uncoded, "k--", linewidth=1.8, label="Uncoded BPSK (Theory)")

    ax.semilogy(snr_range, uncoded_bers, "ko", markersize=6, label="Uncoded BPSK (Sim)")
    ax.semilogy(snr_range, np.maximum(hamming_bers, 1e-6), "g-s", markersize=6, linewidth=2.0, label="Hamming(7,4) Coded")
    ax.semilogy(snr_range, np.maximum(viterbi_bers, 1e-6), "b-^", markersize=6, linewidth=2.0, label="Convolutional (1/2, K=3) Viterbi")

    ax.set_xlabel(r"$E_b / N_0$ (dB)")
    ax.set_ylabel(r"Information Bit Error Rate ($P_b$)")
    ax.set_title("Forward Error Correction Coding Gain in AWGN Channel")
    ax.set_ylim([1e-5, 0.1])
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig10_fec_coding_gain.png")
    plt.close(fig)


def generate_fig11_transceiver_stages():
    """Figure 11: End-to-end Transceiver Image Transmission Demonstration."""
    print("Generating Figure 11: End-to-End Image Transceiver...")
    apply_lab_style()

    # Create synthetic test pattern image
    sz = 32
    img_arr = np.zeros((sz, sz), dtype=np.uint8)
    img_arr[4:12, 4:12] = 220
    img_arr[16:28, 16:28] = 180
    img_arr[8:24, 8:24] = 120
    test_img = Image.fromarray(img_arr, mode="L")

    # Run transceiver at low SNR (noisy) and high SNR (clean)
    pipe_noisy = TransceiverPipeline(mod_scheme="QPSK", coding_scheme="None", channel_type="AWGN")
    res_noisy = pipe_noisy.run(test_img, "image", eb_n0_db=2.0)

    pipe_fec = TransceiverPipeline(mod_scheme="QPSK", coding_scheme="Convolutional(1/2, K=3)", channel_type="AWGN")
    res_fec = pipe_fec.run(test_img, "image", eb_n0_db=5.0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=150)

    axes[0].imshow(test_img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Transmitted Source Image\n(32x32 Pixels)")
    axes[0].axis("off")

    axes[1].imshow(res_noisy.rx_payload, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Uncoded Rx (2 dB)\nBER={res_noisy.post_decode_ber:.3f}, PSNR={res_noisy.image_psnr_db:.1f} dB")
    axes[1].axis("off")

    axes[2].imshow(res_fec.rx_payload, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"Viterbi Coded Rx (5 dB)\nBER={res_fec.post_decode_ber:.5f}, PSNR={res_fec.image_psnr_db:.1f} dB")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig11_transceiver_stages.png")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Generating all figures for College Project Report and Presentation...")
    print(f"Target directory: {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    generate_fig1_pcm_sqnr()
    generate_fig2_line_codes()
    generate_fig3_rrc_eye_and_isi()
    generate_fig4_passband_ber()
    generate_fig5_advanced_constellations()
    generate_fig6_phase_offset()
    generate_fig7_gram_schmidt()
    generate_fig8_all_schemes()
    generate_fig9_awgn_vs_rayleigh()
    generate_fig10_fec_coding_gain()
    generate_fig11_transceiver_stages()

    print("=" * 60)
    print("All 11 report figures generated and saved successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
