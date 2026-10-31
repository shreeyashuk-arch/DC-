"""Render representative visual UI cards and dashboards to outputs/screenshots/."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dcsim.plots.styling import apply_lab_style

OUTPUT_DIR = Path("outputs/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_home_screenshot():
    apply_lab_style()
    fig = plt.figure(figsize=(14, 8), dpi=150)
    fig.patch.set_facecolor("#0F172A")

    # Title Banner
    fig.text(0.5, 0.93, "Digital Communication Lab Suite: Complete Transceiver Simulator",
             ha="center", va="center", color="#F8FAFC", fontsize=18, weight="bold")
    fig.text(0.5, 0.88, "Modular 5th-Semester Lab Examination Suite | 20 Marks Project",
             ha="center", va="center", color="#94A3B8", fontsize=12)

    # 3 Module Cards
    ax1 = fig.add_axes([0.06, 0.48, 0.27, 0.34])
    ax1.set_facecolor("#1E293B")
    ax1.text(0.5, 0.82, "MODULE A: SOURCE & BASEBAND", color="#38BDF8", fontsize=12, weight="bold", ha="center")
    ax1.text(0.5, 0.68, "Owner: Student 1", color="#F1F5F9", fontsize=11, ha="center")
    ax1.text(0.08, 0.45, "• Uniform PCM (1-12 bits)\n• ITU-T μ-law Companding (μ=255)\n• NRZ, RZ, Manchester, AMI PSD\n• Root-Raised Cosine (RRC) & Eye",
             color="#CBD5E1", fontsize=10, va="top")
    ax1.axis("off")

    ax2 = fig.add_axes([0.365, 0.48, 0.27, 0.34])
    ax2.set_facecolor("#1E293B")
    ax2.text(0.5, 0.82, "MODULE B: MODULATION & DETECTION", color="#FBBF24", fontsize=12, weight="bold", ha="center")
    ax2.text(0.5, 0.68, "Owner: Student 2", color="#F1F5F9", fontsize=11, ha="center")
    ax2.text(0.08, 0.45, "• 8-PSK & 16-QAM (Gray coded)\n• DBPSK & Non-coherent BFSK\n• First-principles Gram-Schmidt\n• Carrier Phase Jitter Degradation",
             color="#CBD5E1", fontsize=10, va="top")
    ax2.axis("off")

    ax3 = fig.add_axes([0.67, 0.48, 0.27, 0.34])
    ax3.set_facecolor("#1E293B")
    ax3.text(0.5, 0.82, "MODULE C: CHANNEL & CODING", color="#34D399", fontsize=12, weight="bold", ha="center")
    ax3.text(0.5, 0.68, "Owner: Student 3", color="#F1F5F9", fontsize=11, ha="center")
    ax3.text(0.08, 0.45, "• AWGN & Flat Rayleigh Fading\n• Hamming(7,4) Syndrome Decoder\n• Rate-1/2 K=3 Viterbi Trellis\n• Block Interleaver & Integration",
             color="#CBD5E1", fontsize=10, va="top")
    ax3.axis("off")

    # Global Conventions Card
    ax_bot = fig.add_axes([0.06, 0.08, 0.88, 0.34])
    ax_bot.set_facecolor("#1E293B")
    ax_bot.text(0.5, 0.82, "GLOBAL CONVENTIONS & MATHEMATICAL STANDARDS", color="#A78BFA", fontsize=12, weight="bold", ha="center")
    conv_text = (
        "1. Parameter: Eb/N0 in dB, where Eb is energy per INFORMATION bit.\n"
        "2. Sample-Measured Eb: Eb = sum(|x|^2) / N_info_bits. Automatically scales coded energy to R*Eb without fudge factors.\n"
        "3. Noise Variance: N0/2 per real dimension (sigma_I^2 = N0/2, sigma_Q^2 = N0/2 for complex noise).\n"
        "4. Pulse-shaping filters strictly normalized to unit energy: sum(|h|^2) = 1.0.\n"
        "5. Pure NumPy/SciPy: Zero external comms blackboxes (no commpy, no GNU Radio) for 100% viva transparency."
    )
    ax_bot.text(0.05, 0.65, conv_text, color="#E2E8F0", fontsize=10.5, va="top", linespacing=1.6)
    ax_bot.axis("off")

    fig.savefig(OUTPUT_DIR / "1_home_overview.png")
    plt.close(fig)


def create_module_a_screenshot():
    apply_lab_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # PCM Mid-rise staircase
    n = 3
    delta = 2.0 / (2**n)
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 2 * t)
    idx = np.clip(np.floor((x + 1.0) / delta), 0, (2**n) - 1)
    xh = -1.0 + (idx + 0.5) * delta

    ax1.plot(t, x, "b-", linewidth=2.0, label="Analog Input x(t)")
    ax1.step(t, xh, "r--", linewidth=1.8, where="mid", label=f"Uniform PCM Quantized (n={n} bits)")
    ax1.set_title("Module A: PCM Quantization (Student 1)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    # RRC Eye diagram
    from dcsim.source.pulse_shaping import simulate_rrc_link
    res = simulate_rrc_link(1500, 20.0, alpha=0.35, sps=8, span=6)
    t_eye = res["t_eye"]
    for tr in res["eye_traces"][:50]:
        ax2.plot(t_eye, tr, color="#0284C7", alpha=0.3, linewidth=1.0)
    ax2.axvline(0, color="#DC2626", linestyle=":", linewidth=1.5, label="Sampling Instant")
    ax2.set_title("Module A: Root-Raised Cosine Matched Eye Diagram")
    ax2.set_xlabel("Normalized Time (Ts)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "2_module_a_source_baseband.png")
    plt.close(fig)


def create_module_b_screenshot():
    apply_lab_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # 16-QAM Constellation
    from dcsim.modulation.advanced_mod import simulate_advanced_mod
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=3000, dtype=np.int8)
    _, _, syms_qam, _, _ = simulate_advanced_mod(bits, "16-QAM", 16.0, rng=rng)

    ax1.scatter(np.real(syms_qam[:800]), np.imag(syms_qam[:800]), color="#7C3AED", alpha=0.6, s=25)
    ax1.axhline(0, color="#9CA3AF", linestyle="--")
    ax1.axvline(0, color="#9CA3AF", linestyle="--")
    ax1.set_title("Module B: Gray-Coded 16-QAM Constellation (Student 2)")
    ax1.set_xlabel("In-Phase (I)")
    ax1.set_ylabel("Quadrature (Q)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Phase Offset Impact
    from dcsim.modulation.phase_offset import theoretical_bpsk_phase_offset_ber
    deg = np.linspace(0, 90, 50)
    rad = np.radians(deg)
    for snr in [4.0, 6.0, 8.0]:
        ax2.plot(deg, theoretical_bpsk_phase_offset_ber(snr, rad), linewidth=2.0, label=f"Eb/N0 = {snr} dB")
    ax2.set_title("Module B: Carrier Phase Jitter Degradation")
    ax2.set_xlabel("Phase Offset φ (Degrees)")
    ax2.set_ylabel("Bit Error Rate (Pb)")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "3_module_b_modulation_detection.png")
    plt.close(fig)


def create_module_c_screenshot():
    apply_lab_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # AWGN vs Rayleigh
    snr = np.linspace(0, 20, 100)
    from dcsim.core.conventions import q_function
    from dcsim.channel.fading import theoretical_rayleigh_bpsk_ber
    ax1.semilogy(snr, q_function(np.sqrt(2.0 * 10**(snr/10))), "b-", linewidth=2.0, label="AWGN Channel")
    ax1.semilogy(snr, theoretical_rayleigh_bpsk_ber(snr), "r-", linewidth=2.0, label="Flat Rayleigh Fading")
    ax1.set_title("Module C: AWGN vs. Rayleigh Fading (Student 3)")
    ax1.set_xlabel("Eb/N0 (dB)")
    ax1.set_ylabel("Bit Error Rate (Pb)")
    ax1.set_ylim([1e-5, 0.5])
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)
    ax1.legend()

    # Coding Gain
    snr_c = np.linspace(3, 9, 7)
    from dcsim.coding.system import simulate_coded_bpsk_link
    rng = np.random.default_rng(123)
    b = rng.integers(0, 2, size=30000, dtype=np.int8)
    u_b, c_b = [], []
    for s in snr_c:
        u_b.append(simulate_coded_bpsk_link(b, "None", s, rng=rng)["post_decode_ber"])
        c_b.append(simulate_coded_bpsk_link(b, "Convolutional(1/2, K=3)", s, rng=rng)["post_decode_ber"])

    ax2.semilogy(snr_c, u_b, "ro-", linewidth=2.0, label="Uncoded BPSK")
    ax2.semilogy(snr_c, np.maximum(c_b, 1e-5), "bs-", linewidth=2.0, label="Convolutional (1/2, K=3) Viterbi")
    ax2.set_title("Module C: FEC Viterbi Coding Gain")
    ax2.set_xlabel("Eb/N0 (dB)")
    ax2.set_ylabel("Information BER")
    ax2.set_ylim([1e-5, 0.1])
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "4_module_c_channel_coding.png")
    plt.close(fig)


def create_pipeline_screenshot():
    apply_lab_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    # Test Image
    img = Image.new("L", (32, 32), color=100)
    arr = np.array(img)
    arr[6:26, 6:26] = 220
    test_img = Image.fromarray(arr, mode="L")

    from dcsim.transceiver import TransceiverPipeline
    p_noisy = TransceiverPipeline("QPSK", "None", "AWGN")
    r_noisy = p_noisy.run(test_img, "image", eb_n0_db=2.0)

    p_fec = TransceiverPipeline("QPSK", "Convolutional(1/2, K=3)", "AWGN")
    r_fec = p_fec.run(test_img, "image", eb_n0_db=6.0)

    axes[0].imshow(test_img, cmap="gray")
    axes[0].set_title("Transmitted Source Image")
    axes[0].axis("off")

    axes[1].imshow(r_noisy.rx_payload, cmap="gray")
    axes[1].set_title(f"Uncoded Rx (2 dB)\nBER = {r_noisy.post_decode_ber:.3f}")
    axes[1].axis("off")

    axes[2].imshow(r_fec.rx_payload, cmap="gray")
    axes[2].set_title(f"Viterbi Coded Rx (6 dB)\nBER = {r_fec.post_decode_ber:.5f} (Clean)")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "5_end_to_end_transceiver.png")
    plt.close(fig)


def main():
    print("Generating UI card screenshots...")
    create_home_screenshot()
    create_module_a_screenshot()
    create_module_b_screenshot()
    create_module_c_screenshot()
    create_pipeline_screenshot()
    print("Screenshots saved successfully to outputs/screenshots/")


if __name__ == "__main__":
    main()
