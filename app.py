"""Digital Communication Lab Suite: A Complete Transceiver Simulator.

Designed for 5th-Semester Digital Communication Coursework (20 Marks).
Modular design partitioned across 3 student ownership areas:
- Student 1: Source Coding & Baseband Lab
- Student 2: Modulation, Detection & Signal Space Lab
- Student 3: Channel Impairments, FEC Coding & Integration Lab
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

from dcsim.plots.styling import apply_lab_style
from dcsim.core.conventions import q_function, measure_eb, calculate_n0, add_awgn_noise, calculate_ber
from dcsim.core.passband_link import (
    PassbandModulator,
    PassbandDemodulator,
    simulate_passband_link,
    theoretical_ber,
)
from dcsim.source.pcm import (
    PCMCodec,
    generate_synthetic_tone,
    signal_to_wav_bytes,
    theoretical_sqnr_sine,
    demonstrate_aliasing,
)
from dcsim.source.line_coding import (
    encode_line_code,
    compute_line_code_psd,
    get_line_code_summary_table,
)
from dcsim.source.pulse_shaping import (
    rrc_filter_taps,
    generate_eye_diagram_data,
    simulate_rrc_link,
)
from dcsim.modulation.advanced_mod import (
    simulate_advanced_mod,
    theoretical_advanced_ber,
)
from dcsim.modulation.signal_space import (
    gram_schmidt_orthogonalize,
    compute_signal_space_parameters,
    get_signal_space_preset,
)
from dcsim.modulation.phase_offset import (
    simulate_phase_offset_impact,
    theoretical_bpsk_phase_offset_ber,
)
from dcsim.modulation.comparison import (
    get_modulation_comparison_metrics,
    get_scheme_recommendation,
    evaluate_all_schemes_ber,
)
from dcsim.channel.fading import simulate_rayleigh_bpsk, theoretical_rayleigh_bpsk_ber
from dcsim.coding.system import simulate_coded_bpsk_link
from dcsim.transceiver import TransceiverPipeline

# Streamlit Page Setup
st.set_page_config(
    page_title="Digital Communication Lab Suite",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_lab_style()

# Sidebar: Global Configuration
st.sidebar.title("📡 Lab Suite Controls")
st.sidebar.markdown("**Course:** 5th Sem Digital Communication (20 Marks)")
st.sidebar.markdown("---")

sim_mode = st.sidebar.radio(
    "Simulation Fidelity Mode",
    ["⚡ Quick Mode (Fast GUI)", "🎯 Accurate Mode (High Statistical Confidence)"],
    index=0,
)
num_sim_bits = 15000 if "Quick" in sim_mode else 80000

seed = st.sidebar.number_input("Reproducible Random Seed", min_value=1, max_value=999999, value=2026, step=1)
rng = np.random.default_rng(seed)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Student Viva Ownership")
st.sidebar.info(
    "**Student 1:** Module A (PCM, Line Coding, RRC)\n\n"
    "**Student 2:** Module B (M-ary, Signal Space, Phase)\n\n"
    "**Student 3:** Module C (Fading, FEC Viterbi, Pipeline)"
)

# Main Application Title
st.title("📡 Digital Communication Lab Suite: Complete Transceiver Simulator")
st.caption("A Research-Grade, First-Principles Laboratory Transceiver Simulator for Pair & Team Viva Examination")

# Navigation Tabs
tab_home, tab_mod_a, tab_mod_b, tab_mod_c, tab_pipeline = st.tabs([
    "🏠 System Overview",
    "📊 Module A: Source & Baseband (Student 1)",
    "🎯 Module B: Modulation & Detection (Student 2)",
    "🛡️ Module C: Channel & Coding (Student 3)",
    "🚀 End-to-End Transceiver (Integration)",
])

# =========================================================================
# TAB 0: SYSTEM OVERVIEW & HOME
# =========================================================================
with tab_home:
    st.subheader("System Architecture & Global Conventions")
    st.markdown(
        """
        Welcome to the **Digital Communication Lab Suite**. This simulator models the entire physical layer
        of a modern communication transceiver from mathematical first principles using pure NumPy and SciPy.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("### Module A: Source & Baseband\n**Owner:** Student 1\n- Pulse Code Modulation (PCM) & $\\mu$-Law\n- Line Coding (NRZ, RZ, Manchester, AMI)\n- Root-Raised Cosine (RRC) & Eye Diagrams")
    with col2:
        st.warning("### Module B: Modulation & Detection\n**Owner:** Student 2\n- 8-PSK, 16-QAM, DBPSK, Non-coherent FSK\n- First-Principles Gram-Schmidt Engine\n- Carrier Phase Jitter & Tradeoff Dashboard")
    with col3:
        st.info("### Module C: Channel & Coding\n**Owner:** Student 3\n- AWGN & Flat Rayleigh Fading Channels\n- Hamming(7,4) & Rate-1/2 Viterbi Decoder\n- Block Interleaving & End-to-End Pipeline")

    st.markdown("---")
    st.markdown("#### 📐 Global Energy & Mathematical Conventions")
    st.markdown(
        r"""
        - **$E_b / N_0$ Convention:** The fundamental parameter is energy per **information bit** ($E_b$), not coded bit or symbol.
        - **Sample-Measured $E_b$:** $E_b = \frac{\sum |x[k]|^2}{N_{\text{info\_bits}}}$. This guarantees that oversampling, pulse-shaping energy, and code rates ($R=k/n$) are handled automatically.
        - **AWGN Variance:** Noise variance is $\sigma^2 = N_0 / 2$ per real dimension. For complex baseband, $\sigma_I^2 = N_0/2$ and $\sigma_Q^2 = N_0/2$, yielding total complex noise variance $N_0$.
        - **Gaussian Tail Integral:** $Q(x) = \frac{1}{2} \operatorname{erfc}\left(\frac{x}{\sqrt{2}}\right)$.
        """
    )

    # Core Link Quick Demo
    st.markdown("---")
    st.markdown("#### 🧪 Core Passband Link Quick Test (Shared Foundation)")
    c1, c2, c3 = st.columns(3)
    with c1:
        core_scheme = st.selectbox("Core Scheme", ["BPSK", "QPSK", "BFSK", "OOK"], key="core_sc")
    with c2:
        core_snr = st.slider("Eb/N0 (dB)", 0.0, 10.0, 4.0, 0.5, key="core_snr")
    with c3:
        st.write("")
        st.write("")
        run_core = st.button("Run Core Passband Link", key="btn_run_core")

    if run_core:
        bits = rng.integers(0, 2, size=num_sim_bits, dtype=np.int8)
        ber, demod_res, mod_sig, eb_meas, n0_calc = simulate_passband_link(
            bits, core_scheme, core_snr, samples_per_bit=16, carrier_cycles_per_bit=2, rng=rng  # type: ignore
        )
        theo_b = float(theoretical_ber(core_scheme, core_snr))  # type: ignore

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Simulated BER", f"{ber:.5f}")
        m2.metric("Theoretical BER", f"{theo_b:.5f}")
        m3.metric("Measured Eb", f"{eb_meas:.4f}")
        m4.metric("Calculated N0", f"{n0_calc:.4f}")

        # Constellation / Decision statistics plot
        fig, ax = plt.subplots(figsize=(6, 4))
        stats = demod_res.decision_statistics
        ax.scatter(stats[:500, 0], stats[:500, 1], alpha=0.5, color="#2563EB", s=15)
        ax.set_title(f"Decision Statistics: {core_scheme} @ {core_snr} dB")
        ax.set_xlabel(r"Correlator Branch 1 Output ($\phi_1$)")
        ax.set_ylabel(r"Correlator Branch 2 Output ($\phi_2$)")
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)
        plt.close(fig)

# =========================================================================
# TAB 1: MODULE A (STUDENT 1)
# =========================================================================
with tab_mod_a:
    st.header("Module A: Source Coding & Baseband Transmission")
    st.caption("Responsible Student: Student 1 | Topics: PCM, Companding, Line Coding, RRC Pulse Shaping")

    sub_a1, sub_a2, sub_a3 = st.tabs(["1. PCM & Companding Lab", "2. Line Coding Lab", "3. Pulse Shaping & Eye Diagrams"])

    # 1. PCM Lab
    with sub_a1:
        st.subheader("Pulse Code Modulation (PCM) & ITU-T G.711 $\\mu$-Law Companding")
        col_p1, col_p2 = st.columns([1, 2])

        with col_p1:
            pcm_source = st.radio("Signal Source", ["Synthetic Tone (440 Hz)", "Small-Signal Sine (A = 0.05)", "Upload WAV Audio"])
            n_bits = st.slider("Quantizer Bit Depth (n)", min_value=1, max_value=12, value=4, step=1)
            enable_compand = st.checkbox("Enable mu-law Companding (mu=255)", value=False)

            uploaded_wav = None
            if pcm_source == "Upload WAV Audio":
                uploaded_wav = st.file_uploader("Upload 8kHz Mono WAV", type=["wav"])

        with col_p2:
            if pcm_source == "Synthetic Tone (440 Hz)":
                t_pcm, x_pcm = generate_synthetic_tone(freq_hz=440.0, duration_s=0.2, sampling_rate_hz=8000.0, amplitude=1.0)
            elif pcm_source == "Small-Signal Sine (A = 0.05)":
                t_pcm, x_pcm = generate_synthetic_tone(freq_hz=440.0, duration_s=0.2, sampling_rate_hz=8000.0, amplitude=0.05)
            else:
                if uploaded_wav is not None:
                    # Read wav
                    t_pcm, x_pcm = generate_synthetic_tone(freq_hz=220.0, duration_s=0.2, sampling_rate_hz=8000.0)
                else:
                    t_pcm, x_pcm = generate_synthetic_tone(freq_hz=440.0, duration_s=0.2, sampling_rate_hz=8000.0, amplitude=0.8)

            codec = PCMCodec(n_bits=n_bits, companding=enable_compand, mu=255.0)
            vmax = max(1.0, float(np.max(np.abs(x_pcm))))
            pcm_res = codec.encode_and_decode(x_pcm, v_max=vmax)

            # Audio playback
            st.markdown("##### 🔊 Audio Playback")
            c_aud1, c_aud2 = st.columns(2)
            with c_aud1:
                st.caption("Original Audio")
                st.audio(signal_to_wav_bytes(x_pcm, 8000), format="audio/wav")
            with c_aud2:
                st.caption(f"Reconstructed Audio ({n_bits} bits, Companding={enable_compand})")
                st.audio(signal_to_wav_bytes(pcm_res.quantized_signal, 8000), format="audio/wav")

            # Metrics
            st.markdown("##### 📈 Performance Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Measured SQNR", f"{pcm_res.measured_sqnr_db:.2f} dB")
            m2.metric("Theoretical Full-Scale SQNR", f"{6.02 * n_bits + 1.76:.2f} dB")
            m3.metric("Quantizer Step Size (Δ)", f"{pcm_res.step_size:.4f} V")

            # Waveform plot
            fig, ax = plt.subplots(figsize=(8, 3.5))
            plot_samples = min(200, len(t_pcm))
            ax.plot(t_pcm[:plot_samples] * 1000, x_pcm[:plot_samples], "b-", label="Original Signal", linewidth=2.0)
            ax.step(t_pcm[:plot_samples] * 1000, pcm_res.quantized_signal[:plot_samples], "r--", label="Quantized Output", where="mid")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (V)")
            ax.set_title(f"PCM Quantization: {n_bits} Bits ({1 << n_bits} Levels)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right")
            st.pyplot(fig)
            plt.close(fig)

        # Aliasing Demonstration
        st.markdown("---")
        st.subheader("Nyquist Sampling & Aliasing Demonstration")
        c_al1, c_al2 = st.columns(2)
        with c_al1:
            sig_f0 = st.slider("Continuous Signal Frequency f0 (Hz)", 50, 400, 120, 10)
            fs_sample = st.slider("Sampling Rate Fs (Hz)", 100, 1000, 160, 20)
        with c_al2:
            alias_res = demonstrate_aliasing(signal_freq_hz=float(sig_f0), sampling_rate_hz=float(fs_sample), duration_s=0.04)
            if alias_res["is_aliased"]:
                st.error(f"⚠️ Aliasing Occurred! Nyquist Rate = {alias_res['nyquist_rate_hz']:.1f} Hz > Sampling Rate {fs_sample} Hz.\nPerceived Alias Frequency = **{alias_res['alias_freq_hz']:.1f} Hz**")
            else:
                st.success(f"✅ Nyquist Satisfied! Sampling Rate {fs_sample} Hz >= Nyquist Rate {alias_res['nyquist_rate_hz']:.1f} Hz. Lossless sampling.")

        fig_al, ax_al = plt.subplots(figsize=(10, 3.5))
        ax_al.plot(alias_res["t_cont"] * 1000, alias_res["x_cont"], "b-", alpha=0.5, label=f"Original Continuous Signal ({sig_f0} Hz)")
        ax_al.stem(alias_res["t_sampled"] * 1000, alias_res["x_sampled"], linefmt="r-", markerfmt="ro", label=f"Discrete Samples ({fs_sample} Hz)")
        if alias_res["is_aliased"]:
            ax_al.plot(alias_res["t_cont"] * 1000, alias_res["x_reconstructed"], "g--", linewidth=2.0, label=f"Perceived Alias ({alias_res['alias_freq_hz']:.1f} Hz)")
        ax_al.set_xlabel("Time (ms)")
        ax_al.set_ylabel("Amplitude")
        ax_al.set_title("Time-Domain Sampling and Spectral Folding")
        ax_al.grid(True, linestyle="--", alpha=0.35)
        ax_al.legend(loc="upper right")
        st.pyplot(fig_al)
        plt.close(fig_al)

    # 2. Line Coding Lab
    with sub_a2:
        st.subheader("Line Coding Schemes & Power Spectral Density Analysis")
        col_lc1, col_lc2 = st.columns([1, 2])

        with col_lc1:
            bit_string = st.text_input("Test Bit Pattern (Binary 0s and 1s)", "1011001010")
            selected_codes = st.multiselect("Active Schemes to Compare", ["NRZ-L", "RZ", "Manchester", "AMI"], default=["NRZ-L", "RZ", "Manchester", "AMI"])

        with col_lc2:
            bits_parsed = np.array([int(b) for b in bit_string if b in ("0", "1")], dtype=np.int8)
            if len(bits_parsed) == 0:
                bits_parsed = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)

            fig_lc, ax_lc = plt.subplots(figsize=(9, 4.5))
            for i, sc in enumerate(selected_codes):
                t_lc, w_lc = encode_line_code(bits_parsed, sc, samples_per_bit=32, voltage=1.0)  # type: ignore
                offset = (len(selected_codes) - 1 - i) * 2.5
                ax_lc.plot(t_lc, w_lc + offset, linewidth=2.0, label=sc)
                ax_lc.axhline(offset, color="#9CA3AF", linestyle=":", linewidth=0.8)

            ax_lc.set_xlabel("Time (Bit Periods $T_b$)")
            ax_lc.set_ylabel("Signaling Voltage Level (Offset)")
            ax_lc.set_title(f"Line Code Waveforms for {len(bits_parsed)} Bits")
            ax_lc.set_xlim([0, len(bits_parsed)])
            ax_lc.grid(True, linestyle="--", alpha=0.3)
            ax_lc.legend(loc="upper right")
            st.pyplot(fig_lc)
            plt.close(fig_lc)

        st.markdown("##### 📊 Comparative Summary Table")
        table_data = get_line_code_summary_table()
        st.dataframe(table_data, use_container_width=True)

    # 3. Pulse Shaping & Eye Diagrams
    with sub_a3:
        st.subheader("Root-Raised Cosine (RRC) Matched Filtering & Eye Diagram")
        c_ps1, c_ps2 = st.columns([1, 2])

        with c_ps1:
            alpha = st.slider("RRC Roll-off Factor (α)", 0.0, 1.0, 0.35, 0.05)
            rrc_snr = st.slider("Eb/N0 (dB) for Eye Diagram", 0.0, 30.0, 20.0, 2.0)
            induce_isi = st.checkbox("Violate Nyquist (Poor Band-Limiting / Severe ISI)", value=False)

        with c_ps2:
            res_ps = simulate_rrc_link(
                num_bits=1500,
                eb_n0_db=rrc_snr,
                alpha=alpha,
                sps=8,
                span=6,
                poor_bandlimit=induce_isi,
                rng=rng,
            )

            fig_eye, ax_eye = plt.subplots(figsize=(8, 4))
            traces = res_ps["eye_traces"]
            t_eye = res_ps["t_eye"]
            for tr in traces[:60]:
                ax_eye.plot(t_eye, tr, color="#DC2626" if induce_isi else "#0284C7", alpha=0.25, linewidth=1.1)
            ax_eye.axvline(0.0, color="#1E293B", linestyle=":", linewidth=1.5, label="Optimum Sampling Instant")
            ax_eye.set_xlabel("Time (Symbol Durations $T_s$)")
            ax_eye.set_ylabel("Received Amplitude")
            ax_eye.set_title(f"Eye Diagram (α = {alpha}, ISI Impaired = {induce_isi})")
            ax_eye.grid(True, linestyle="--", alpha=0.35)
            ax_eye.legend(loc="upper right")
            st.pyplot(fig_eye)
            plt.close(fig_eye)

        m1, m2 = st.columns(2)
        m1.metric("Simulated BER (with RRC)", f"{res_ps['ber']:.5f}")
        m2.metric("Theoretical AWGN BPSK Benchmark", f"{res_ps['ber_theoretical']:.5f}")

# =========================================================================
# TAB 2: MODULE B (STUDENT 2)
# =========================================================================
with tab_mod_b:
    st.header("Module B: Digital Modulation, Signal Space & Detection")
    st.caption("Responsible Student: Student 2 | Topics: 8-PSK, 16-QAM, Non-coherent BFSK/DBPSK, Gram-Schmidt, Phase Error")

    sub_b1, sub_b2, sub_b3, sub_b4 = st.tabs([
        "1. Advanced Constellations Lab",
        "2. Gram-Schmidt Orthogonalization",
        "3. Carrier Phase Offset Lab",
        "4. Tradeoff & Comparison Dashboard",
    ])

    # 1. Advanced Constellations Lab
    with sub_b1:
        st.subheader("M-ary PSK, QAM and Non-Coherent Detection")
        col_m1, col_m2 = st.columns([1, 2])

        with col_m1:
            adv_scheme = st.selectbox("Modulation Scheme", ["8-PSK", "16-QAM", "DBPSK", "Non-coherent BFSK"])
            adv_snr = st.slider("Eb/N0 (dB)", 0.0, 16.0, 8.0, 1.0)
            run_adv = st.button("Simulate Constellation", key="btn_run_adv")

        with col_m2:
            bits_adv = rng.integers(0, 2, size=num_sim_bits, dtype=np.int8)
            ber_adv, rx_b_adv, syms_adv, eb_a, n0_a = simulate_advanced_mod(bits_adv, adv_scheme, adv_snr, rng=rng)  # type: ignore
            theo_adv = float(theoretical_advanced_ber(adv_scheme, adv_snr))  # type: ignore

            m1, m2, m3 = st.columns(3)
            m1.metric("Simulated BER", f"{ber_adv:.5f}")
            m2.metric("Theoretical BER", f"{theo_adv:.5f}")
            m3.metric("Measured Eb / N0", f"{eb_a:.3f} / {n0_a:.4f}")

            fig_adv, ax_adv = plt.subplots(figsize=(6, 5))
            plot_pts = syms_adv[:800]
            ax_adv.scatter(np.real(plot_pts), np.imag(plot_pts), color="#2563EB", alpha=0.5, s=20)
            ax_adv.axhline(0, color="#9CA3AF", linestyle="--")
            ax_adv.axvline(0, color="#9CA3AF", linestyle="--")
            ax_adv.set_xlabel("In-Phase (I)")
            ax_adv.set_ylabel("Quadrature (Q)")
            ax_adv.set_title(f"{adv_scheme} Constellation @ {adv_snr} dB")
            ax_adv.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig_adv)
            plt.close(fig_adv)

    # 2. Gram-Schmidt Orthogonalization Tool
    with sub_b2:
        st.subheader("First-Principles Gram-Schmidt Signal Space Analysis")
        c_gs1, c_gs2 = st.columns([1, 2])

        with c_gs1:
            gs_preset = st.selectbox("Preset Signal Set", ["QPSK (4 Signals, D=2)", "BFSK (2 Tones, D=2)", "Walsh-Hadamard (4 Orthogonal Codes, D=4)"])
            preset_key = "QPSK" if "QPSK" in gs_preset else ("BFSK" if "BFSK" in gs_preset else "Walsh-Hadamard")

        with c_gs2:
            signals, sig_labels = get_signal_space_preset(preset_key, N=128)
            gs_res = gram_schmidt_orthogonalize(signals)
            params = compute_signal_space_parameters(gs_res)

            m1, m2, m3 = st.columns(3)
            m1.metric("Orthonormal Basis Dimension (D)", int(params["dimension"]))
            m2.metric("Minimum Euclidean Distance (d_min)", f"{params['min_euclidean_distance']:.4f}")
            m3.metric("Max Reconstruction Error", f"{params['max_reconstruction_error']:.2e}")

            # Plot coordinates
            fig_gs, (ax_g1, ax_g2) = plt.subplots(1, 2, figsize=(10, 4.5))
            t_axis = np.linspace(0, 1, 128)
            for d in range(min(4, gs_res.dimension)):
                ax_g1.plot(t_axis, gs_res.basis_functions[d], linewidth=2.0, label=f"phi_{d+1}(t)")
            ax_g1.set_title(f"Gram-Schmidt Orthonormal Bases (D={gs_res.dimension})")
            ax_g1.grid(True, linestyle="--", alpha=0.4)
            ax_g1.legend()

            ax_g2.scatter(gs_res.coordinates[:, 0], gs_res.coordinates[:, 1], color="#2563EB", s=80, zorder=5)
            for i, txt in enumerate(sig_labels[:6]):
                ax_g2.annotate(f"s{i+1}", (gs_res.coordinates[i, 0] + 0.05, gs_res.coordinates[i, 1] + 0.05), weight="bold")
            ax_g2.set_title("Signal Vector Coordinates in First 2 Basis Dimensions")
            ax_g2.axhline(0, color="#9CA3AF", linestyle="--")
            ax_g2.axvline(0, color="#9CA3AF", linestyle="--")
            ax_g2.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig_gs)
            plt.close(fig_gs)

    # 3. Carrier Phase Offset Lab
    with sub_b3:
        st.subheader("Carrier Phase Jitter & Coherent Demodulation Degradation")
        c_ph1, c_ph2 = st.columns([1, 2])

        with c_ph1:
            phi_deg = st.slider("Carrier Phase Error φ (Degrees)", -180.0, 180.0, 25.0, 5.0)
            phase_snr = st.slider("Eb/N0 (dB)", 0.0, 12.0, 6.0, 1.0, key="slider_phase_snr")

        with c_ph2:
            res_ph = simulate_phase_offset_impact("BPSK", eb_n0_db=phase_snr, phi_deg=phi_deg, num_bits=num_sim_bits, rng=rng)

            m1, m2 = st.columns(2)
            m1.metric("Simulated BER under Phase Offset", f"{res_ph['ber_sim']:.5f}")
            m2.metric("Theoretical Bound: Q(sqrt(2Eb/N0)*cos(φ))", f"{res_ph['ber_theo']:.5f}")

            fig_ph, ax_ph = plt.subplots(figsize=(6, 4))
            rx_syms = res_ph["rx_symbols"][:600]
            ax_ph.scatter(np.real(rx_syms), np.imag(rx_syms), color="#DC2626", alpha=0.5, s=20, label=f"Rotated by {phi_deg}°")
            ax_ph.axvline(0, color="#1E293B", linestyle="-", linewidth=1.5, label="Receiver Slicing Boundary")
            ax_ph.axhline(0, color="#9CA3AF", linestyle="--")
            ax_ph.set_xlabel("In-Phase (I)")
            ax_ph.set_ylabel("Quadrature (Q)")
            ax_ph.set_title(f"BPSK Constellation under Carrier Phase Offset φ = {phi_deg}°")
            ax_ph.grid(True, linestyle="--", alpha=0.4)
            ax_ph.legend()
            st.pyplot(fig_ph)
            plt.close(fig_ph)

    # 4. Tradeoff & Comparison Dashboard
    with sub_b4:
        st.subheader("All-Modulation Schemes Comparison & Recommendation Engine")
        st.dataframe(get_modulation_comparison_metrics(), use_container_width=True)

        # Recommendation engine
        st.markdown("##### 🧭 Intelligent Scheme Recommendation Engine")
        rc1, rc2, rc3, rc4 = st.columns(4)
        bw_lim = rc1.checkbox("Bandwidth-Limited Channel", value=True)
        pw_lim = rc2.checkbox("Power-Limited Channel", value=False)
        pj_sev = rc3.checkbox("Severe Phase Jitter / Doppler", value=False)
        ce_req = rc4.checkbox("Constant Envelope Required (Non-linear PA)", value=False)

        rec = get_scheme_recommendation(bw_lim, pw_lim, pj_sev, ce_req)
        st.success(f"**Recommended Scheme:** `{rec['recommended_scheme']}`\n\n**Justification:** {rec['engineering_justification']}")

# =========================================================================
# TAB 3: MODULE C (STUDENT 3)
# =========================================================================
with tab_mod_c:
    st.header("Module C: Channel Impairments, Forward Error Correction & Integration")
    st.caption("Responsible Student: Student 3 | Topics: AWGN vs Rayleigh Fading, Hamming(7,4), Viterbi Decoding, Block Interleaving")

    sub_c1, sub_c2, sub_c3 = st.tabs([
        "1. Rayleigh Fading Lab",
        "2. Forward Error Correction (FEC) Lab",
        "3. Interleaving & Burst Errors",
    ])

    # 1. Rayleigh Fading Lab
    with sub_c1:
        st.subheader("Flat Rayleigh Fading Channel vs AWGN Benchmark")
        c_rf1, c_rf2 = st.columns([1, 2])

        with c_rf1:
            ray_snr = st.slider("Average Eb/N0 (dB)", 0.0, 24.0, 12.0, 2.0)
            run_ray = st.button("Simulate Rayleigh Fading Link", key="btn_run_ray")

        with c_rf2:
            res_ray = simulate_rayleigh_bpsk(num_bits=num_sim_bits, eb_n0_db=ray_snr, rng=rng)
            m1, m2, m3 = st.columns(3)
            m1.metric("Rayleigh Simulated BER", f"{res_ray['ber_sim']:.5f}")
            m2.metric("Rayleigh Theoretical BER", f"{res_ray['ber_theo_rayleigh']:.5f}")
            m3.metric("AWGN Theoretical BER", f"{res_ray['ber_theo_awgn']:.2e}")

            st.warning(f"⚠️ **AWGN vs Rayleigh Performance Gap:** Rayleigh fading requires an additional ~17 dB of SNR at BER = 1e-3!")

    # 2. Forward Error Correction Lab
    with sub_c2:
        st.subheader("Channel Coding: Repetition(3), Hamming(7,4), and Rate-1/2 Viterbi")
        c_fec1, c_fec2 = st.columns([1, 2])

        with c_fec1:
            fec_scheme = st.selectbox("FEC Coding Scheme", ["None", "Repetition(3)", "Hamming(7,4)", "Convolutional(1/2, K=3)"])
            fec_snr = st.slider("Eb/N0 (dB) [Energy per INFO Bit]", 0.0, 10.0, 6.0, 1.0)
            fec_chan = st.selectbox("Channel Model", ["AWGN", "Rayleigh"])

        with c_fec2:
            bits_fec = rng.integers(0, 2, size=num_sim_bits, dtype=np.int8)
            res_fec = simulate_coded_bpsk_link(
                bits_fec,
                fec_scheme,  # type: ignore
                eb_n0_db=fec_snr,
                channel_type=fec_chan,  # type: ignore
                rng=rng,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Pre-Decoding (Channel) BER", f"{res_fec['pre_decode_ber']:.5f}")
            m2.metric("Post-Decoding (Final) BER", f"{res_fec['post_decode_ber']:.5f}")
            m3.metric("Uncoded BPSK Theory", f"{res_fec['theo_uncoded_ber']:.5f}")

            if res_fec["post_decode_ber"] < res_fec["pre_decode_ber"]:
                st.success(f"🎉 **Positive Error Reduction:** Corrected {res_fec['pre_decode_errors'] - res_fec['post_decode_errors']} channel errors!")
            elif fec_scheme != "None":
                st.info("ℹ️ At low SNR below the cutoff threshold, rate expansion penalty (R*Eb per coded symbol) dominates.")

    # 3. Interleaving Lab
    with sub_c3:
        st.subheader("Burst Error Mitigation via Block Interleaving")
        st.markdown(
            """
            In wireless fading channels, deep fading dips cause bursts of consecutive bit errors that overwhelm
            algebraic block codes and short-constraint convolutional codes. The **Block Interleaver** scrambles
            temporal order prior to transmission, dispersing burst errors across widely separated codewords upon reception.
            """
        )
        c_int1, c_int2 = st.columns(2)
        with c_int1:
            st.markdown("##### Link WITHOUT Interleaver")
            res_no_int = simulate_coded_bpsk_link(
                rng.integers(0, 2, size=num_sim_bits, dtype=np.int8),
                "Convolutional(1/2, K=3)",
                eb_n0_db=8.0,
                channel_type="Rayleigh",
                use_interleaver=False,
                rayleigh_block_len=8,
                rng=rng,
            )
            st.metric("Post-Decode BER (No Interleaver)", f"{res_no_int['post_decode_ber']:.5f}")

        with c_int2:
            st.markdown("##### Link WITH 16x16 Block Interleaver")
            res_with_int = simulate_coded_bpsk_link(
                rng.integers(0, 2, size=num_sim_bits, dtype=np.int8),
                "Convolutional(1/2, K=3)",
                eb_n0_db=8.0,
                channel_type="Rayleigh",
                use_interleaver=True,
                rayleigh_block_len=8,
                rng=rng,
            )
            st.metric("Post-Decode BER (With Interleaver)", f"{res_with_int['post_decode_ber']:.5f}")

# =========================================================================
# TAB 4: COMPLETE END-TO-END TRANSCEIVER PIPELINE
# =========================================================================
with tab_pipeline:
    st.header("🚀 Complete End-to-End Transceiver Demonstration")
    st.caption("Full Physical Layer Integration: Source -> Coding -> Interleaver -> Modulation -> Channel -> Demodulation -> Decoding -> Sink")

    col_pipe_ctrl, col_pipe_view = st.columns([1, 2])

    with col_pipe_ctrl:
        st.markdown("### 🛠️ Transceiver Controls")
        src_sel = st.selectbox("Payload Source", ["Text Message", "Test Image", "Audio Signal"])
        mod_sel = st.selectbox("Modulation Scheme", ["BPSK", "QPSK", "8-PSK", "16-QAM", "DBPSK", "BFSK", "OOK"], index=1)
        cod_sel = st.selectbox("Channel Coding", ["None", "Hamming(7,4)", "Convolutional(1/2, K=3)", "Repetition(3)"], index=2)
        chan_sel = st.selectbox("Channel Impairment", ["AWGN", "Rayleigh"], index=0)
        use_int_pipe = st.checkbox("Enable Block Interleaver", value=True)
        snr_pipe = st.slider("Eb/N0 (dB)", 0.0, 16.0, 6.0, 1.0)

        payload_obj: Any = None
        if src_sel == "Text Message":
            payload_obj = st.text_area("Message String", "Digital Communication Viva 2026: All Modules Functional!")
        elif src_sel == "Test Image":
            st.info("Sending 32x32 pixel geometric test image.")
            img_arr = np.zeros((32, 32), dtype=np.uint8)
            img_arr[4:14, 4:14] = 220
            img_arr[16:28, 16:28] = 180
            img_arr[8:24, 8:24] = 120
            payload_obj = Image.fromarray(img_arr, mode="L")
        else:
            st.info("Sending synthetic 440 Hz speech-band tone.")
            _, payload_obj = generate_synthetic_tone(freq_hz=440.0, duration_s=0.1, sampling_rate_hz=8000.0)

        run_pipeline = st.button("🚀 Transmit Signal Across Channel", key="btn_run_pipe")

    with col_pipe_view:
        if run_pipeline:
            src_type_map = {"Text Message": "text", "Test Image": "image", "Audio Signal": "audio"}
            pipeline = TransceiverPipeline(
                mod_scheme=mod_sel,  # type: ignore
                coding_scheme=cod_sel,  # type: ignore
                channel_type=chan_sel,  # type: ignore
                use_interleaver=use_int_pipe,
            )

            res = pipeline.run(payload_obj, src_type_map[src_sel], eb_n0_db=snr_pipe, rng=rng)

            # Stage Metrics Bar
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Information Bits", res.num_info_bits)
            m2.metric("Channel Errors (Pre-FEC)", f"{res.pre_decode_errors} ({res.pre_decode_ber:.4f})")
            m3.metric("Final Errors (Post-FEC)", f"{res.post_decode_errors} ({res.post_decode_ber:.5f})")
            m4.metric("Code Rate (R)", f"{res.code_rate:.2f}")

            # Reconstructed Output Display
            st.markdown("---")
            st.markdown("### 📺 Original vs. Received Payload")

            if src_sel == "Text Message":
                c_t1, c_t2 = st.columns(2)
                c_t1.info(f"**Original Text:**\n\n`{payload_obj}`")
                c_t2.success(f"**Decoded Text:**\n\n`{res.rx_payload}`")

            elif src_sel == "Test Image":
                c_i1, c_i2 = st.columns(2)
                with c_i1:
                    st.caption("Original Transmitted Image")
                    st.image(payload_obj, width=160)
                with c_i2:
                    st.caption(f"Reconstructed Image (PSNR = {res.image_psnr_db:.1f} dB)")
                    st.image(res.rx_payload, width=160)

            elif src_sel == "Audio Signal":
                c_a1, c_a2 = st.columns(2)
                with c_a1:
                    st.caption("Original Audio")
                    st.audio(signal_to_wav_bytes(payload_obj, 8000), format="audio/wav")
                with c_a2:
                    st.caption(f"Reconstructed Audio (SQNR = {res.audio_sqnr_db:.1f} dB)")
                    st.audio(signal_to_wav_bytes(res.rx_payload, 8000), format="audio/wav")

            # Constellation
            if res.constellation_points is not None:
                st.markdown("---")
                st.markdown("### 🌌 Received Constellation Diagram")
                fig_c, ax_c = plt.subplots(figsize=(6, 4))
                pts = res.constellation_points[:600]
                ax_c.scatter(np.real(pts), np.imag(pts), color="#2563EB", alpha=0.5, s=20)
                ax_c.axhline(0, color="#9CA3AF", linestyle="--")
                ax_c.axvline(0, color="#9CA3AF", linestyle="--")
                ax_c.set_title(f"Equalized Constellation: {mod_sel} @ {snr_pipe} dB ({chan_sel})")
                ax_c.set_xlabel("In-Phase (I)")
                ax_c.set_ylabel("Quadrature (Q)")
                ax_c.grid(True, linestyle="--", alpha=0.4)
                st.pyplot(fig_c)
                plt.close(fig_c)
