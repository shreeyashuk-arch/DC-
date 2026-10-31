# Digital Communication Lab Suite: 10-Slide Presentation Outline

A presentation structure designed for a 10- to 15-minute viva presentation.

---

### Slide 1: Title & Team Credentials
- **Slide Title:** Digital Communication Lab Suite: A Complete Transceiver Simulator
- **Course & Exam:** EC-501: Digital Communication Systems Laboratory (20 Marks Project)
- **Team Members & Ownership:**
  - **Student 1:** Module A — Source Coding, Companding, Line Coding & Baseband Pulse Shaping
  - **Student 2:** Module B — M-ary Modulations, Signal Space, Gram-Schmidt & Phase Offset
  - **Student 3:** Module C — Channel Modeling (AWGN & Rayleigh), FEC Coding, Transceiver Integration & Web App
- **Key Takeaway:** A research-grade, first-principles communication simulator built purely in NumPy and SciPy—zero blackboxes (`no commpy`, `no GNU Radio`) for 100% viva transparency.

---

### Slide 2: Mathematical Rigor & Global Conventions
- **Energy Normalization Standard:**
  - Parameterized strictly as $E_b / N_0$ in dB, where $E_b$ is energy per **information** bit.
  - Discrete Sample-Measured $E_b$:
    $$E_b = \frac{\sum_{k=1}^N |x[k]|^2}{N_{\text{info\_bits}}}$$
    Automatically scales coded systems to $R \cdot E_b$ per symbol without manual fudge factors.
  - Noise Variance: $\sigma^2 = N_0 / 2$ per real dimension ($\sigma_I^2 = N_0/2, \sigma_Q^2 = N_0/2$).
  - Gaussian Tail Integral: $Q(x) = \frac{1}{2}\operatorname{erfc}(x / \sqrt{2})$.
  - Unit Energy Pulse Shaping: $\sum |h[k]|^2 = 1.0$.

---

### Slide 3: End-to-End Transceiver Architecture
- **Complete Physical Layer Block Diagram:**
  - Source (Text / Image / Audio / Bits) $\to$ PCM Quantizer / $\mu$-Law Compander $\to$ Channel Encoder (FEC) $\to$ Matrix Interleaver $\to$ Modulator (Passband / Complex Baseband) $\to$ Physical Channel (AWGN / Flat Rayleigh) $\to$ Demodulator $\to$ Deinterleaver $\to$ Channel Decoder (Syndrome / Viterbi) $\to$ Sink Reconstruction.
- **Stage Diagnostics:** Real-time tracking of channel errors (pre-FEC), post-FEC errors, BER, Image PSNR (dB), Audio SQNR (dB), and constellation scatter points.

---

### Slide 4: Module A — PCM, Companding & Aliasing (Student 1)
- **Uniform PCM Quantizer:**
  - Mid-rise uniform quantization over $[-V_{\max}, +V_{\max}]$ with $n = 1 \dots 12$ bits.
  - Empirical SQNR aligns with theoretical $6.02n + 1.76\text{ dB}$ within $0.3\text{ dB}$ under ergodic sampling.
- **ITU-T G.711 $\mu$-Law Companding ($\mu = 255$):**
  - Compresses dynamic range logarithmically before uniform quantization.
  - Boosts small-amplitude signal ($A = 0.05$) SQNR from $22\text{ dB}$ to $36.2\text{ dB}$ (a **$+14.2\text{ dB}$** improvement).
- **Nyquist Sampling & Aliasing:**
  - Visual demonstration of sub-Nyquist sampling ($F_s < 2 f_0$) folding into alias frequency $f_{\text{alias}} = |f_0 - k F_s|$.

---

### Slide 5: Module A — Line Coding & RRC Matched Filtering (Student 1)
- **Line Coding Spectra:**
  - Evaluates NRZ-L, RZ, Manchester, and AMI waveforms.
  - Welch PSD demonstrates Manchester's strictly zero DC power and guaranteed transitions for clock recovery, balanced against AMI's narrower bandwidth ($R_b$) and bipolar error detection.
- **Root-Raised Cosine (RRC) Matched Filtering:**
  - Transmit RRC and Receive matched RRC cascade to form a full Raised Cosine filter.
  - Strictly satisfies Nyquist's First Criterion: $p(k T_s) = 0$ for $k \ne 0$ (zero ISI).
- **Eye Diagram Analysis:**
  - Open eye at $t = 0$ confirms maximal noise margin and low timing jitter sensitivity.
  - Bandwidth violation demonstration proves complete eye closure under severe ISI.

---

### Slide 6: Module B — Advanced M-ary Modulations & Constellations (Student 2)
- **Complex Baseband M-ary Schemes:**
  - **8-PSK:** Gray-coded circular constellation ($k = 3$ bits/sym, constant envelope for non-linear power amplifiers).
  - **16-QAM:** Gray-coded $4 \times 4$ rectangular grid ($k = 4$ bits/sym, $4\text{ bps/Hz}$ spectral efficiency, normalized by $1/\sqrt{10}$).
  - **Non-Coherent DBPSK:** Encodes phase difference $\Delta\phi \in \{0, \pi\}$; delay-product demodulator $r_k r_{k-1}^*$ requires no carrier phase tracking ($P_b = 0.5 e^{-E_b/N_0}$).
  - **Non-Coherent BFSK:** Dual-branch envelope detection ($P_b = 0.5 e^{-E_b/(2N_0)}$).

---

### Slide 7: Module B — Gram-Schmidt Tool & Phase Offset (Student 2)
- **First-Principles Gram-Schmidt Tool:**
  - Projects arbitrary waveforms $\{s_i(t)\}$ onto orthonormal bases $\{\phi_j(t)\}$ from scratch.
  - Computes coordinates, signal energy $E_i = \sum s_{ij}^2$, pairwise distance matrix $d_{ik}$, and verifies numerical orthonormality $\langle\phi_i, \phi_j\rangle = \delta_{ij}$ (error $< 10^{-6}$).
- **Carrier Phase Jitter Degradation:**
  - Receiver phase error $\phi$ attenuates signal amplitude by $\cos\phi$:
    $$P_b(\phi) = Q\left(\sqrt{\frac{2E_b}{N_0}} \cos\phi\right)$$
  - Demonstrates that at $\phi = 90^\circ$, effective signal collapses to zero ($P_b = 0.5$).
- **Intelligent Scheme Recommendation Engine:** Recommends optimal schemes based on bandwidth, power, phase jitter, and PAPR constraints.

---

### Slide 8: Module C — AWGN vs. Flat Rayleigh Fading (Student 3)
- **Flat Rayleigh Fading Channel Model:**
  - Complex Gaussian envelope $h = h_I + j h_Q$ with $|h| \sim \text{Rayleigh}$, $\mathbb{E}[|h|^2] = 1.0$.
  - Maximum ratio equalizer with perfect Channel State Information (CSI): $\hat{x} = \frac{h^*}{|h|^2} y$.
  - Theoretical average BER:
    $$P_b = \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}_b}{1 + \bar{\gamma}_b}}\right) \approx \frac{1}{4 \bar{\gamma}_b}$$
- **The Massive Fading Penalty:**
  - In AWGN, BER drops exponentially. In Rayleigh fading, BER drops only linearly ($10\text{ dB/decade}$).
  - At $\text{BER} = 10^{-3}$, Rayleigh fading requires a **17 dB higher SNR** than AWGN.

---

### Slide 9: Module C — Forward Error Correction & Interleaving (Student 3)
- **Error Control Codes:**
  - **Repetition(3):** Majority logic decoding ($R = 1/3, t = 1$).
  - **Systematic Hamming(7,4):** Syndrome calculation $\mathbf{s} = \mathbf{r} H^T$ maps uniquely to error column index for single-bit flipping ($R = 4/7, t = 1$).
  - **Rate-1/2 $K=3$ Convolutional Code:** Trellis with 4 states, generators $(7, 5)_8$. Hard-decision Viterbi decoder uses Add-Compare-Select (ACS) and traceback.
- **Coding Gain Verification:**
  - Transmitting with rate $R$ dilutes energy to $R \cdot E_b$ per symbol.
  - Above $5\text{ dB}$, Viterbi decoding achieves over **$2.5\text{ dB}$ of true coding gain** over uncoded BPSK.
- **Block Interleaver:**
  - $16 \times 16$ matrix interleaver scrambles contiguous burst errors from Rayleigh fading dips into isolated single-bit errors.

---

### Slide 10: Transceiver Live Demo, Test Suite & Conclusion
- **Live Transceiver Integration:**
  - Flawless transmission of $32 \times 32$ image through AWGN channel with QPSK + Viterbi at $5\text{ dB}$ ($\text{BER} = 0.0, \text{PSNR} = 100\text{ dB}$).
  - Audio transmission with PCM encoding achieving $> 40\text{ dB}$ reconstructed SQNR.
- **Verification Results:**
  - **51 automated pytest unit and integration tests** passing with 100% success in $< 3$ seconds.
  - 11 publication-grade figures generated from real simulations.
- **Conclusion:** A complete, modular, viva-tested digital communication suite ready for laboratory presentation.
