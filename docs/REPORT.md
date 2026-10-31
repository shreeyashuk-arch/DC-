# Digital Communication Lab Suite: A Complete Transceiver Simulator
## Final Laboratory Course Project Report (5th Semester B.Tech / B.E. — 20 Marks)

**Course:** EC-501: Digital Communication & Systems Laboratory  
**Project Title:** Digital Communication Lab Suite: A Complete Transceiver Simulator  
**Submission Date:** Academic Term 2026  
**Team Members & Ownership:**
- **Student 1 (Roll No. 01):** Owner of Module A — Source Coding, Companding, Line Coding & Baseband Pulse Shaping
- **Student 2 (Roll No. 02):** Owner of Module B — M-ary Modulation, Signal Space, Gram-Schmidt & Carrier Phase Impairments
- **Student 3 (Roll No. 03):** Owner of Module C — Channel Modeling (AWGN & Rayleigh), FEC Coding, Transceiver Integration & Web App

---

## Abstract

This report documents the design, mathematical formulation, software architecture, and empirical verification of the **Digital Communication Lab Suite**, an advanced, modular transceiver simulator developed in Python. Designed for undergraduate engineering students, the suite simulates the entire physical layer of digital communication systems—from information source sampling to sink reconstruction—without reliance on high-level communications blackboxes (such as `commpy` or `GNU Radio`). Every algorithm is implemented from mathematical first principles using pure NumPy and SciPy. The simulator implements strict energy normalization conventions ($E_b/N_0$ defined per information bit, discrete sample-measured energy $E_b$, and noise variance $N_0/2$ per real dimension). The suite is divided into three student-owned modules: Module A (PCM quantization, ITU-T G.711 $\mu$-law companding, line codes with Welch PSD, and Root-Raised Cosine matched filtering), Module B (complex baseband 8-PSK, 16-QAM, non-coherent DBPSK/BFSK, first-principles Gram-Schmidt orthogonalization, and carrier phase jitter analysis), and Module C (AWGN and flat Rayleigh fading channels with perfect CSI, systematic Hamming(7,4) syndrome decoding, rate-1/2 $K=3$ hard-decision Viterbi decoding, matrix block interleaving, and end-to-end payload transmission). All algorithms are verified by an automated test suite of 51 unit and integration tests passing with 100% success. High-resolution simulation figures and an interactive multi-tab Streamlit dashboard allow live viva examination demonstrations.

---

## Chapter 1: Introduction & System Architecture

### 1.1 Motivation and Pedagogical Objectives
Digital Communication coursework at the 5th-semester level requires understanding both continuous-time waveform representations and discrete-time statistical signal processing. Traditional commercial toolkits often hide crucial intermediate steps—such as correlator projections, decision metrics, trellis state transitions, and syndrome lookup tables—behind opaque functions. The objective of this project is to construct a **fully transparent, research-grade, viva-ready communication simulator** where every mathematical formula taught in theory lectures directly corresponds to vectorized Python code.

### 1.2 System Block Diagram
The transceiver follows standard digital communication architecture:
```text
[Information Source] (Text / Image / Audio / Bits)
        │
        ▼
[Source Coding / PCM] (Uniform / μ-Law Quantizer, n = 1..12 bits)
        │
        ▼
[Channel Coding (FEC)] (Repetition(3) / Hamming(7,4) / Conv(1/2, K=3))
        │
        ▼
[Block Interleaver] (Matrix Transpose: Disperses Fading Burst Errors)
        │
        ▼
[Digital Modulator] (BPSK / QPSK / 8-PSK / 16-QAM / DBPSK / BFSK / OOK)
        │
        ▼
[Physical Channel] (AWGN / Flat Rayleigh Fading with CSI)
        │
        ▼
[Demodulator] (Coherent Correlator / Complex Slicing / Envelope Detector)
        │
        ▼
[Deinterleaver] (Matrix Inverse Transpose)
        │
        ▼
[Channel Decoder] (Syndrome Error Corrector / Viterbi Trellis Traceback)
        │
        ▼
[Sink Reconstruction] (Decoded Text / Reconstructed Image / Audio Playback)
```

### 1.3 Global Mathematical Conventions
To ensure absolute integrity across all modules, four global conventions are enforced throughout the codebase:
1. **$E_b / N_0$ per Information Bit:** The signal-to-noise ratio is consistently parameterized as $E_b / N_0$ in decibels (dB), where $E_b$ represents average energy per **raw information bit**, ensuring fair comparisons across different coding rates and modulation alphabet sizes.
2. **Sample-Measured $E_b$:**
   $$E_b = \frac{1}{N_{\text{info\_bits}}} \sum_{k=1}^{N} |x[k]|^2$$
   Measuring energy directly from the synthesized discrete samples automatically handles oversampling factors ($L$), carrier scaling, pulse-shaping energy, and code rates ($R = k/n$).
3. **Noise Variance Scaling:** For noise spectral density $N_0 = E_b / 10^{((E_b/N_0)_{\text{dB}} / 10)}$, the noise variance per real dimension is $\sigma^2 = N_0 / 2$. Complex baseband noise has variance $N_0/2$ on both in-phase (I) and quadrature (Q) components, giving total complex variance $N_0$.
4. **Unit Energy Filters:** Transmit and receive pulse-shaping filters strictly satisfy $\sum |h[k]|^2 = 1.0$.

---

## Chapter 2: Phase 1 — Core Passband Transceiver Link

### 2.1 Passband Signal Model
The core link models real passband continuous waveforms using an integer number of carrier cycles $m$ over each bit duration $T_b$:
$$s_i(t) = A_i \cos(2\pi f_c t + \theta_i), \quad f_c = \frac{m}{T_b}$$
For sampling frequency $F_s$ and $L$ samples per bit, discrete orthonormal carrier bases are constructed:
$$\phi_{\cos}[k] = \frac{\cos(2\pi m k / L)}{\sqrt{\sum \cos^2(2\pi m n / L)}}, \quad \phi_{\sin}[k] = \frac{-\sin(2\pi m k / L)}{\sqrt{\sum \sin^2(2\pi m n / L)}}$$

### 2.2 Correlator Receivers
The receiver correlates the incoming noisy passband signal $r[k] = s[k] + w[k]$ against the local orthonormal basis:
$$y_j = \sum_{k=0}^{L-1} r[k] \phi_j[k] = s_{ij} + n_j$$
Since $\phi_j$ has unit energy and $w[k] \sim \mathcal{N}(0, N_0/2)$, the noise sample at the correlator output is zero-mean Gaussian with variance $\sigma_n^2 = N_0 / 2$.

### 2.3 Empirical Results
Simulated bit error rates over 120,000 bits were benchmarked against theoretical Gaussian tail integrals:
- BPSK & QPSK: $P_b = Q(\sqrt{2 E_b / N_0})$
- Coherent BFSK & OOK: $P_b = Q(\sqrt{E_b / N_0})$

As shown in **Figure 4 (`outputs/figures/fig4_passband_ber.png`)**, simulated BER points match theoretical waterfall curves within 10% relative error across all SNRs, verifying exact numerical calibration of $E_b$, $N_0$, and correlator gain.

---

## Chapter 3: Module A — Source & Baseband Lab (Owner: Student 1)

### 3.1 Pulse Code Modulation (PCM)
Module A implements a fully configurable mid-rise uniform quantizer supporting 1 to 12 bits of resolution. For a full-scale sinusoid $x(t) = \sin(2\pi f t)$ with peak voltage $V_{\max} = 1.0$, the signal power is $P_s = 0.5$. With step size $\Delta = 2 / 2^n$, the quantization noise variance is $\sigma_q^2 = \Delta^2 / 12 = 2^{-2n} / 3$. The theoretical Signal-to-Quantization-Noise Ratio (SQNR) is derived as:
$$\text{SQNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_s}{\sigma_q^2}\right) = 1.7609 + 6.0206 n \text{ dB}$$

**Experimental Verification:** As documented in **Figure 1 (`outputs/figures/fig1_pcm_sqnr.png`)**, measured SQNR for bit depths $n \in \{4, 6, 8, 10\}$ aligns with theory within **0.3 dB** under ergodic sampling, verifying Bennett's quantization model.

### 3.2 ITU-T G.711 $\mu$-Law Companding
To prevent drastic SQNR degradation for low-amplitude speech signals, non-linear logarithmic companding is implemented with standard parameter $\mu = 255$:
$$y = \operatorname{sgn}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}, \quad \hat{x} = \operatorname{sgn}(\hat{y}) \frac{(1 + \mu)^{|\hat{y}|} - 1}{\mu}$$
For small signals ($A = 0.05$, 26 dB below full scale), uniform quantization SQNR collapses to $\approx 22\text{ dB}$, whereas $\mu$-law companding sustains an SQNR of **$36.2\text{ dB}$**, delivering an advantage of **$+14.2\text{ dB}$**.

### 3.3 Line Coding & Spectral Density
Four line codes are evaluated: NRZ-L, RZ (50% duty cycle), Manchester (biphase), and AMI (Alternate Mark Inversion). Power Spectral Densities (PSD) computed via Welch's averaged periodogram are shown in **Figure 2 (`outputs/figures/fig2_line_codes.png`)**:
- **Manchester:** Exhibits strictly zero DC power ($\text{PSD}(0) = -\infty\text{ dB}$) and guaranteed transitions at mid-bit for reliable clock synchronization, at the cost of double bandwidth ($2 R_b$).
- **AMI:** Alternating marks cancel DC bias while maintaining a narrower mainlobe ($R_b$) with built-in error detection via bipolar violations.

### 3.4 Root-Raised Cosine Pulse Shaping & Eye Diagrams
To eliminate Inter-Symbol Interference (ISI) in band-limited channels, an analytical Root-Raised Cosine (RRC) filter impulse response is synthesized with roll-off factor $\alpha \in [0, 1]$ and normalized to unit energy:
$$h(t) = \frac{\sin(\pi \frac{t}{T}(1-\alpha)) + 4\alpha \frac{t}{T} \cos(\pi \frac{t}{T}(1+\alpha))}{\pi \frac{t}{T} (1 - (4\alpha \frac{t}{T})^2)}$$
When identical matched RRC filters are cascaded at transmitter and receiver, their combined frequency response is a full Raised Cosine (RC) filter, yielding exact zero-crossings at integer symbol multiples: $p(k T_s) \approx 0$ for $k \ne 0$.

In **Figure 3 (`outputs/figures/fig3_rrc_eye_and_isi.png`)**, the matched RRC eye diagram exhibits a wide, open eye at $t = 0$ with optimal noise margin. In contrast, violating the Nyquist bandwidth criterion with an unshaped narrow lowpass filter causes severe pulse broadening, completely closing the eye and creating insurmountable ISI.

---

## Chapter 4: Module B — Modulation & Detection Lab (Owner: Student 2)

### 4.1 Advanced M-ary Modulations
Module B implements higher-order and non-coherent schemes in complex baseband:
1. **8-PSK:** Employs Gray coding on the unit circle. Theoretical bit error rate is:
   $$P_b \approx \frac{2}{3} Q\left(\sqrt{6 \frac{E_b}{N_0}} \sin\frac{\pi}{8}\right)$$
2. **16-QAM:** Utilizes an in-phase and quadrature $4\times 4$ rectangular grid with independent Gray slicing, delivering 4 bps/Hz bandwidth efficiency:
   $$P_b \approx 0.75 Q\left(\sqrt{0.8 \frac{E_b}{N_0}}\right)$$
3. **Non-Coherent DBPSK:** Transmits information via phase differences $\Delta\phi \in \{0, \pi\}$. The receiver computes $\operatorname{Re}\{r_k r_{k-1}^*\}$, requiring no carrier recovery. Theoretical BER is $P_b = 0.5 \exp(-E_b/N_0)$.
4. **Non-Coherent BFSK:** Evaluates envelopes on two orthogonal tone branches. Theoretical BER is $P_b = 0.5 \exp(-E_b / (2 N_0))$.

Constellations and decision boundaries are illustrated in **Figure 5 (`outputs/figures/fig5_advanced_constellations.png`)**.

### 4.2 First-Principles Gram-Schmidt Tool
A general Gram-Schmidt orthogonalizer converts any $M$ energy signals into $D$ orthonormal basis functions $\{\phi_j(t)\}$ by projecting out existing components:
$$\psi_k(t) = s_k(t) - \sum_{j=1}^{d} \langle s_k, \phi_j \rangle \phi_j(t), \quad \phi_{d+1}(t) = \frac{\psi_k(t)}{\|\psi_k\|}$$
The tool outputs orthonormal bases, signal coordinates, energy $E_i = \sum s_{ij}^2$, pairwise Euclidean distance matrix $d_{ik}$, and verifies numerical orthonormality $\langle \phi_i, \phi_j \rangle = \delta_{ij}$ and lossless reconstruction ($\max |s_i - \hat{s}_i| < 10^{-6}$), as shown in **Figure 7 (`outputs/figures/fig7_gram_schmidt.png`)**.

### 4.3 Carrier Phase Jitter Impairments
When a coherent receiver experiences carrier phase error $\phi$, the received BPSK symbol is rotated by $e^{j\phi}$. Projecting onto the in-phase decision axis scales signal amplitude by $\cos\phi$:
$$P_b(\phi) = Q\left(\sqrt{\frac{2E_b}{N_0}} \cos\phi\right)$$
As plotted in **Figure 6 (`outputs/figures/fig6_phase_offset_impact.png`)**, as $\phi$ approaches $90^\circ$, the effective signal disappears and BER reaches $0.5$ (complete link failure).

### 4.4 Modulation Comparison & Tradeoff Analysis
All 8 modulation schemes are compared in **Figure 8 (`outputs/figures/fig8_all_schemes_waterfall.png`)**. The dashboard summarizes spectral efficiency (bps/Hz) versus required $E_b/N_0$ for target $\text{BER} = 10^{-4}$:
- **BPSK / QPSK:** $8.4\text{ dB}$ (highest power efficiency).
- **16-QAM:** $12.2\text{ dB}$ (highest spectral efficiency: 4.0 bps/Hz).
- **DBPSK:** $9.3\text{ dB}$ (robust against phase ambiguity with only 0.9 dB penalty).

---

## Chapter 5: Module C — Channel Impairments, FEC & Transceiver Integration (Owner: Student 3)

### 5.1 AWGN vs. Flat Rayleigh Fading Channel
In a multipath propagation channel with non-line-of-sight conditions, the complex channel coefficient $h = h_I + j h_Q$ has envelope $\alpha = |h|$ following a Rayleigh distribution with $\mathbb{E}[\alpha^2] = 1.0$. With perfect Channel State Information (CSI), the receiver equalizes: $\hat{x} = \frac{h^*}{|h|^2} y$. The theoretical average BER for BPSK in flat Rayleigh fading is:
$$P_b = \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}_b}{1 + \bar{\gamma}_b}}\right), \quad \bar{\gamma}_b = \frac{E_b}{N_0}$$
In **Figure 9 (`outputs/figures/fig9_awgn_vs_rayleigh.png`)**, while AWGN error rates drop exponentially, Rayleigh fading drops only inversely with SNR ($10\text{ dB/decade}$). At $\text{BER} = 10^{-3}$, Rayleigh fading suffers an enormous **17 dB penalty** relative to AWGN.

### 5.2 Forward Error Correction (FEC) & Viterbi Decoding
Module C implements three error control codes:
1. **Repetition(3):** Majority-logic decoding corrects up to 1 error per 3 bits ($R = 1/3$).
2. **Hamming(7,4):** Systematic code with $G = [I_4 | P]$ and $H = [P^T | I_3]$. The syndrome vector $\mathbf{s} = \mathbf{r} H^T \pmod 2$ maps directly to the unique column index of the erroneous bit, correcting any single-bit error ($t = 1$).
3. **Rate-1/2 $K=3$ Convolutional Code:** Employs generators $g_1 = 7_8 = (111)_2$ and $g_2 = 5_8 = (101)_2$. An efficient hard-decision Viterbi decoder evaluates Hamming path metrics across a 4-state trellis using Add-Compare-Select (ACS) and traces back survivor paths from terminal state `00`.

**Coding Gain Verification:** Transmitting with code rate $R$ dilutes energy to $R \cdot E_b$ per coded symbol. In **Figure 10 (`outputs/figures/fig10_fec_coding_gain.png`)**, at low SNR (< 4.5 dB), the rate penalty causes coded BER to exceed uncoded BER. However, above 5 dB, the Viterbi decoder exhibits steep performance improvement, achieving **$> 2.5\text{ dB}$ of coding gain** at $\text{BER} = 10^{-4}$.

### 5.3 Matrix Block Interleaver
Under Rayleigh block fading, deep fades cause contiguous bursts of errors. A $16 \times 16$ rectangular block interleaver writes bits by rows and transmits by columns. At the receiver, deinterleaving scatters burst errors across 16 different codewords, enabling the Hamming and Viterbi decoders to correct them as isolated single-bit errors.

### 5.4 End-to-End Transceiver Integration
The complete pipeline integrates source encoding, channel coding, interleaving, modulation, physical channel, demodulation, deinterleaving, decoding, and payload reconstruction. As demonstrated in **Figure 11 (`outputs/figures/fig11_transceiver_stages.png`)**, transmitting a $32 \times 32$ pixel test image across an AWGN channel at 5 dB with Viterbi coding produces a flawless reconstruction with $\text{BER} = 0.0$ and $\text{PSNR} = 100\text{ dB}$, whereas uncoded transmission at 2 dB suffers severe pixel corruption ($\text{BER} = 0.038, \text{PSNR} = 18.4\text{ dB}$).

---

## Chapter 6: Automated Test Verification & Experimental Results

The entire simulation suite is covered by a comprehensive automated test suite implemented in `pytest`. All tests execute deterministically using fixed random seeds.

### 6.1 Test Summary Table

| Test Module | Coverage Scope | Verified Criterion | Result |
| :--- | :--- | :--- | :---: |
| `tests/test_phase1_core.py` | Q-function, Eb measurement, N0 scaling, BPSK, QPSK, BFSK, OOK, Text & Image converters | Noiseless roundtrips zero error; mid-SNR BER within 10% of theory | **12 / 12 PASS** |
| `tests/test_module_a.py` | PCM uniform quantizer, $\mu$-law companding, line codes, RRC unit energy, zero-ISI, aliasing | SQNR within 1 dB of $6.02n+1.76$; $>10$ dB companding gain on small signals | **11 / 11 PASS** |
| `tests/test_module_b.py` | 8-PSK, 16-QAM, DBPSK, Non-coherent BFSK, Gram-Schmidt engine, phase offset, comparison metrics | Orthonormality error $< 10^{-6}$; reconstruction error $< 10^{-6}$; BER matches theory | **11 / 11 PASS** |
| `tests/test_module_c.py` | Rayleigh fading, Hamming(7,4), Viterbi Rate-1/2 decoder, block interleaver | Hamming corrects all single errors; Viterbi exhibits coding gain | **6 / 6 PASS** |
| `tests/test_full_transceiver.py` | Integrated text, image, audio pipeline across AWGN & Rayleigh channels | Zero errors in noiseless link; PSNR = 100 dB; audio SQNR $> 40$ dB | **11 / 11 PASS** |
| **Total Test Suite** | **Complete Transceiver Simulator** | **All 51 automated unit & integration tests** | **51 / 51 PASS (100%)** |

---

## Chapter 7: Conclusion & Future Scope

### 7.1 Conclusion
The **Digital Communication Lab Suite** successfully realizes a complete, modular, and mathematically rigorous transceiver simulator tailored for 5th-semester laboratory coursework. By avoiding third-party communications libraries and implementing all algorithms—from PCM companders to Viterbi trellises and Gram-Schmidt orthogonalizers—using NumPy and SciPy, the project provides complete algorithmic transparency for viva voce examination. The software architecture cleanly delineates responsibilities across three student owners while guaranteeing seamless end-to-end transceiver integration.

### 7.2 Future Scope & Stretch Goals
1. **CRC Packet Framing & ARQ Retransmission:** Implement cyclic redundancy check (CRC-16/CRC-32) error detection paired with a Stop-and-Wait Automatic Repeat reQuest (ARQ) protocol to model packet retransmission counts under varying channel BERs.
2. **Minimum Shift Keying (MSK):** Extend continuous phase frequency shift keying (CPFSK) with modulation index $h = 0.5$ to study constant-envelope spectrally contained signaling.
3. **Soft-Decision Viterbi Decoding:** Implement Euclidean distance branch metrics to demonstrate the theoretical $\approx 2.0\text{ dB}$ performance advantage of soft-decision decoding over hard-decision decoding.
4. **OFDM Multi-Carrier Transmission:** Integrate an IFFT/FFT-based orthogonal frequency division multiplexing engine with cyclic prefix insertion to combat frequency-selective multipath fading.

---

## References
1. John G. Proakis and Masoud Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008.
2. Simon Haykin, *Communication Systems*, 4th ed., John Wiley & Sons, 2001.
3. B. P. Lathi and Zhi Ding, *Modern Digital and Analog Communication Systems*, 4th ed., Oxford University Press, 2010.
4. ITU-T Recommendation G.711, *Pulse Code Modulation (PCM) of Voice Frequencies*, International Telecommunication Union, 1988.
5. Bernard Sklar, *Digital Communications: Fundamentals and Applications*, 2nd ed., Prentice Hall, 2001.
