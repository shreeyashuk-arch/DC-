# Digital Communication Lab Suite: Team Ownership & 10-Minute Viva Demo Script

This document details the exact module boundaries, individual student responsibilities, and a timed 10-minute viva presentation script for the 3-student project team.

---

## 1. Student Ownership Mapping

### Student 1: Owner of Module A (Source Coding & Baseband Lab)
- **Primary Source Files:**
  - `dcsim/source/pcm.py` (Uniform quantizer, $\mu$-law compander, SQNR analysis, aliasing)
  - `dcsim/source/line_coding.py` (NRZ-L, RZ, Manchester, AMI waveforms, Welch PSD)
  - `dcsim/source/pulse_shaping.py` (Root-Raised Cosine filter, matched filtering, eye diagrams, ISI)
  - `tests/test_module_a.py`
- **What Student 1 Must Be Able to Explain:**
  1. The mathematical derivation of the PCM SQNR formula: $\text{SQNR} = 6.02n + 1.76\text{ dB}$ for a full-scale sinusoid.
  2. Why $\mu$-law companding ($\mu=255$) preserves SQNR for small signals and its ITU-T G.711 logarithmic equation.
  3. The trade-offs between the four line codes: DC content, bandwidth, clock recovery (Manchester mid-bit transitions), and AMI bipolar violation detection.
  4. Nyquist's First Criterion for zero ISI, why the RRC matched filter pair cascades into a full Raised Cosine, and how an eye diagram visually represents noise margin and timing jitter.
  5. The sampling theorem and how under-sampling below $2 f_0$ folds into alias frequency $f_{\text{alias}} = |f_0 - k F_s|$.

---

### Student 2: Owner of Module B (Modulation, Detection & Signal Space Lab)
- **Primary Source Files:**
  - `dcsim/modulation/advanced_mod.py` (8-PSK, 16-QAM Gray mapping, DBPSK, non-coherent BFSK)
  - `dcsim/modulation/signal_space.py` (First-principles Gram-Schmidt engine, distance matrices, reconstruction)
  - `dcsim/modulation/phase_offset.py` (Carrier phase error simulation, $Q(\sqrt{2E_b/N_0}\cos\phi)$ bound)
  - `dcsim/modulation/comparison.py` (Trade-off table, recommendation engine, composite waterfall)
  - `tests/test_module_b.py`
- **What Student 2 Must Be Able to Explain:**
  1. Constellation geometry and Gray mapping for 8-PSK and 16-QAM, including the energy normalization factor ($1/\sqrt{10}$ for 16-QAM).
  2. Why 16-QAM achieves greater power efficiency than 16-PSK by utilizing a 2D square grid.
  3. The non-coherent detection mechanism of DBPSK (delay-product $r_k r_{k-1}^*$) and non-coherent BFSK (envelope/energy detection across two branches).
  4. The step-by-step vector projections of the Gram-Schmidt orthogonalization algorithm, Parseval energy relation $E_i = \sum s_{ij}^2$, and orthonormality verification $\langle \phi_i, \phi_j \rangle = \delta_{ij}$.
  5. The mathematical derivation of carrier phase error degradation $P_b = Q(\sqrt{2E_b/N_0}\cos\phi)$ and why performance collapses at $\phi = 90^\circ$.
  6. The intelligent modulation recommendation rules (power-limited vs bandwidth-limited vs phase-jittered links).

---

### Student 3: Owner of Module C (Channel, Coding, Integration & Web GUI)
- **Primary Source Files:**
  - `dcsim/channel/fading.py` (Flat Rayleigh fading with perfect CSI, AWGN vs Rayleigh gap)
  - `dcsim/coding/repetition.py` (Repetition(3) majority logic)
  - `dcsim/coding/hamming.py` (Systematic Hamming(7,4) generator $G$, parity check $H$, syndrome table decoding)
  - `dcsim/coding/convolutional.py` (Rate-1/2 $K=3$ $(7,5)_8$ encoder and hard-decision Viterbi trellis decoder)
  - `dcsim/coding/interleaver.py` (Matrix block interleaver and deinterleaver)
  - `dcsim/coding/system.py` (Rate scaling $R \cdot E_b$ convention and coding gain measurement)
  - `dcsim/transceiver.py` (End-to-end multi-stage pipeline)
  - `app.py` (Streamlit web dashboard)
  - `tests/test_module_c.py` and `tests/test_full_transceiver.py`
- **What Student 3 Must Be Able to Explain:**
  1. The statistical derivation of flat Rayleigh fading $P_b = 0.5(1 - \sqrt{g/(1+g)})$ and why there is a massive 17 dB gap relative to AWGN at $\text{BER} = 10^{-3}$.
  2. The syndrome decoding math for Hamming(7,4): why each non-zero syndrome vector $\mathbf{s} = \mathbf{r} H^T$ uniquely identifies the error column.
  3. The Viterbi algorithm trellis operations: branch Hamming metrics, Add-Compare-Select (ACS) recursion, and survivor traceback.
  4. Why channel coding dilutes energy to $R \cdot E_b$ per symbol, why coded BER is worse at low SNR, and how to measure true coding gain.
  5. How a matrix block interleaver combats burst errors in fading channels by converting bursts into isolated single-bit errors.
  6. How the complete transceiver pipeline connects every block from source to sink with PSNR and SQNR diagnostics.

---

## 2. Timed 10-Minute Viva Presentation & Live Demo Script

| Time | Speaker | Action on Screen | Spoken Script / Dialogue |
| :---: | :---: | :---: | :--- |
| **0:00 - 1:00** | **Student 3** | Open Streamlit App on `🏠 System Overview` tab. | *"Good morning, respected examiners. Our project is the Digital Communication Lab Suite: A Complete Transceiver Simulator. We built this from first principles using pure NumPy and SciPy—zero communication blackbox libraries. Notice our global conventions: $E_b/N_0$ is defined per information bit, $E_b$ is sample-measured directly, and noise variance is $N_0/2$ per real dimension. Here is our 3-student partitioning: Student 1 owns Source & Baseband, Student 2 owns Modulation & Signal Space, and I own Channel Impairments, FEC, and the Integrated Pipeline. I now hand over to Student 1."* |
| **1:00 - 3:30** | **Student 1** | Switch to `📊 Module A: Source & Baseband` tab. | *"Thank you. In Module A, I present three baseband subsystems:*<br>*(1) **PCM & Companding:** Here is our uniform quantizer. Notice at $n=4$ bits, the measured SQNR is $25.8\text{ dB}$, matching the theoretical $6.02n + 1.76\text{ dB}$ within 0.3 dB. When I switch to small signals ($A = 0.05$), uniform SQNR drops to $22\text{ dB}$, but enabling $\mu$-law companding ($\mu=255$) restores SQNR to $36.2\text{ dB}$—a $14\text{ dB}$ gain. Let's listen to the reconstructed audio playback.*<br>*(2) **Aliasing:** When sampling frequency $F_s = 160\text{ Hz}$ falls below the Nyquist rate for a $120\text{ Hz}$ tone ($240\text{ Hz}$), the signal folds back to $|120 - 160| = 40\text{ Hz}$.*<br>*(3) **Line Coding & Pulse Shaping:** Here are NRZ, RZ, Manchester, and AMI waveforms with Welch PSDs—notice Manchester has zero DC power. On our RRC filter with roll-off $\alpha=0.35$, the eye diagram is wide open. But if I click 'Violate Nyquist', the eye closes completely due to severe ISI. Over to Student 2."* |
| **3:30 - 6:00** | **Student 2** | Switch to `🎯 Module B: Modulation & Detection` tab. | *"Thank you. In Module B, I cover bandpass modulation, signal space, and phase impairments:*<br>*(1) **Higher-Order Modulation:** Here is our complex baseband Gray-coded 16-QAM constellation at $16\text{ dB}$ SNR. By using a 2D square grid, 16-QAM achieves 4 bps/Hz while preserving power efficiency over 16-PSK. We also support non-coherent DBPSK and BFSK.*<br>*(2) **Gram-Schmidt Tool:** Here is our first-principles Gram-Schmidt engine. For QPSK, it computes 2 orthonormal basis functions $\phi_1(t)$ and $\phi_2(t)$, verifies orthonormality ($\langle\phi_i,\phi_j\rangle = \delta_{ij}$ to $10^{-6}$), and outputs the 2D constellation coordinates.*<br>*(3) **Carrier Phase Jitter:** Adjusting this phase slider to $\phi = 25^\circ$ visibly rotates the BPSK constellation. At $90^\circ$, the in-phase projection disappears, and BER degrades to $0.5$ exactly as predicted by $Q(\sqrt{2E_b/N_0}\cos\phi)$.*<br>*(4) **Comparison Dashboard:** All 8 schemes are benchmarked on our composite waterfall plot with an intelligent recommendation engine. Over to Student 3."* |
| **6:00 - 8:30** | **Student 3** | Switch to `🛡️ Module C: Channel & Coding` tab. | *"Thank you. In Module C, I demonstrate channel impairments and error control:*<br>*(1) **Rayleigh Fading vs AWGN:** In flat Rayleigh fading with perfect CSI, the error rate drops only linearly ($10\text{ dB/decade}$) rather than exponentially. At $\text{BER} = 10^{-3}$, Rayleigh fading requires a massive $17\text{ dB}$ higher SNR than AWGN.*<br>*(2) **Forward Error Correction:** Here we test Hamming(7,4) and Rate-1/2 $K=3$ Convolutional code with Viterbi decoding. Notice that at low SNR, coding increases BER because of rate penalty ($R \cdot E_b$ per coded symbol). But above $5\text{ dB}$, Viterbi achieves over $2.5\text{ dB}$ of true coding gain.*<br>*(3) **Block Interleaving:** Under Rayleigh burst fading, a code alone fails. Enabling our $16\times 16$ block interleaver disperses contiguous burst errors across multiple codewords, restoring error correction capability."* |
| **8:30 - 10:00** | **All Team** | Switch to `🚀 End-to-End Transceiver` tab. | **Student 3:** *"Finally, here is our full end-to-end transceiver. We will transmit an image at $5\text{ dB}$ with QPSK and Viterbi coding. Clicking Transmit... as you can see, the reconstructed image is flawless: $\text{BER} = 0.0$ and $\text{PSNR} = 100\text{ dB}$. If I turn off FEC, the image is severely degraded.*<br>**Student 1:** *All 51 automated unit tests pass in 2.7 seconds.*<br>**Student 2:** *And all report figures are generated from real simulation.*<br>**All:** *We are now ready for your questions. Thank you!"* |
