# Digital Communication Lab Suite: Theory and Mathematical Foundations

This document presents the complete mathematical theory, engineering principles, and design rationale behind every block in the **Digital Communication Lab Suite**. Written in clear, accessible language, this guide is designed to empower students to confidently answer examiner questions in the viva voce.

---

## 1. Global Conventions & Energy Normalization

### 1.1 The Fundamental Parameter: $E_b / N_0$
In digital communications, comparing modulation schemes fairly requires normalizing signal energy to the **information bit**:
- $E_b$: Average energy transmitted per raw **information bit** (not channel symbol, nor coded bit).
- $N_0$: One-sided noise power spectral density of Additive White Gaussian Noise (AWGN).
- $E_b / N_0$: Dimensionless SNR per bit, commonly expressed in decibels:
  $$\left(\frac{E_b}{N_0}\right)_{\text{dB}} = 10 \log_{10}\left(\frac{E_b}{N_0}\right)$$

### 1.2 Sample-Measured Energy ($E_b$)
In discrete simulation with sampling rate $F_s$, a continuous signal $x(t)$ is represented by samples $x[k]$. To prevent errors arising from manual scaling factors (e.g., oversampling $L$ samples/symbol, pulse-shaping filter gain, carrier amplitude, or code rate $R = k/n$), $E_b$ is computed directly from the discrete sample sequence:
$$E_b = \frac{1}{N_{\text{info\_bits}}} \sum_{k=1}^{N_{\text{samples}}} |x[k]|^2$$
- **Automatic Code Rate Scaling:** If $N$ information bits are encoded into $N/R$ coded bits (where $R < 1$), the discrete signal contains $N/R$ symbols. The energy per coded symbol automatically becomes $R \cdot E_b$. This correctly reflects that channel coding dilutes energy per transmitted symbol.

### 1.3 AWGN Noise Generation
Given $E_b$ and target $(E_b/N_0)_{\text{dB}}$, the noise spectral density is:
$$N_0 = \frac{E_b}{10^{(E_b/N_0)_{\text{dB}} / 10}}$$
The noise variance per real dimension is:
$$\sigma^2 = \frac{N_0}{2}$$
- **Real Passband Signals:** $w[k] \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = \sqrt{N_0 / 2}$.
- **Complex Baseband Signals:** $w[k] = w_I[k] + j w_Q[k]$, where $w_I, w_Q \sim \mathcal{N}(0, N_0/2)$. The total variance is $\mathbb{E}[|w|^2] = \sigma_I^2 + \sigma_Q^2 = N_0$.

### 1.4 The Gaussian Tail Q-Function
The Gaussian tail probability $Q(x) = P(X > x)$ for $X \sim \mathcal{N}(0, 1)$ is evaluated via the complementary error function $\operatorname{erfc}$:
$$Q(x) = \frac{1}{\sqrt{2\pi}} \int_x^\infty e^{-u^2 / 2} du = \frac{1}{2} \operatorname{erfc}\left(\frac{x}{\sqrt{2}}\right)$$

---

## 2. Phase 1: Core Passband Link

### 2.1 Passband Modulation with Integer Carrier Cycles
In passband transmission, baseband information modulates a sinusoidal carrier $c(t) = \cos(2\pi f_c t)$. To ensure exact orthogonality over a bit duration $T_b$, the carrier frequency is set to an integer number of cycles $m$:
$$f_c = \frac{m}{T_b}$$
For $L$ samples per bit at sampling rate $F_s$, the discrete orthonormal carrier basis is:
$$\phi[k] = \frac{\cos(2\pi m k / L)}{\sqrt{\sum_{n=0}^{L-1} \cos^2(2\pi m n / L)}}, \quad k = 0, \dots, L-1$$
This guarantees $\sum_{k=0}^{L-1} \phi^2[k] = 1.0$.

### 2.2 Core Modulation Schemes
1. **BPSK (Binary Phase Shift Keying):**
   - Bit 1: $s_1(t) = +\sqrt{E_b} \phi(t)$
   - Bit 0: $s_0(t) = -\sqrt{E_b} \phi(t)$
   - Distance between signals: $d = 2\sqrt{E_b}$
   - Decision threshold: 0
   - Theoretical BER: $P_b = Q\left(\frac{d/2}{\sigma}\right) = Q\left(\frac{\sqrt{E_b}}{\sqrt{N_0/2}}\right) = Q\left(\sqrt{\frac{2E_b}{N_0}}\right)$

2. **QPSK (Quadrature Phase Shift Keying):**
   - Transmits 2 bits per symbol $(b_I, b_Q)$ over symbol duration $T_s = 2T_b$.
   - Basis: $\phi_I(t) \propto \cos(2\pi f_c t)$, $\phi_Q(t) \propto -\sin(2\pi f_c t)$.
   - Gray coded mapping:
     - `00` $\to (+1, +1)/\sqrt{2}$
     - `01` $\to (-1, +1)/\sqrt{2}$
     - `11` $\to (-1, -1)/\sqrt{2}$
     - `10` $\to (+1, -1)/\sqrt{2}$
   - Because I and Q carriers are orthogonal, bit errors on I and Q are independent.
   - Theoretical BER: $P_b = Q\left(\sqrt{\frac{2E_b}{N_0}}\right)$ (identical to BPSK, but with twice the spectral efficiency!).

3. **Coherent BFSK (Binary Frequency Shift Keying):**
   - Uses two orthogonal frequencies: $f_1 = m_1 / T_b$, $f_2 = m_2 / T_b$ with $|m_1 - m_2| \ge 1$.
   - Bit 1: $s_1(t) = \sqrt{E_b} \phi_1(t)$
   - Bit 0: $s_0(t) = \sqrt{E_b} \phi_2(t)$
   - Distance in 2D space: $d = \sqrt{(\sqrt{E_b})^2 + (\sqrt{E_b})^2} = \sqrt{2E_b}$.
   - Theoretical BER: $P_b = Q\left(\sqrt{\frac{E_b}{N_0}}\right)$ (3 dB worse than BPSK).

4. **ASK / OOK (On-Off Keying):**
   - Bit 1: $s_1(t) = \sqrt{2E_b} \phi(t)$
   - Bit 0: $s_0(t) = 0$
   - Average energy: $\frac{1}{2}(2E_b + 0) = E_b$.
   - Optimal threshold: $\frac{1}{2}\sqrt{2E_b} = \sqrt{E_b/2}$.
   - Theoretical BER: $P_b = Q\left(\sqrt{\frac{E_b}{N_0}}\right)$.

### 2.3 Correlator Receiver
The receiver projects the received noisy waveform $r(t) = s(t) + w(t)$ onto the orthonormal basis $\phi_j(t)$:
$$y_j = \sum_{k=0}^{L-1} r[k] \phi_j[k]$$
The output vector $\mathbf{y} = [y_1, y_2, \dots]$ represents the decision statistics in signal space, which are compared against decision thresholds to recover the transmitted bits.

---

## 3. Module A: Source & Baseband Lab (Student 1)

### 3.1 Pulse Code Modulation (PCM)
PCM converts an analog continuous signal $x(t)$ into a digital bitstream in three steps: **Sampling $\to$ Quantization $\to$ Encoding**.

#### Quantization Step Size
For a uniform quantizer operating over voltage range $[-V_{\max}, +V_{\max}]$ with $n$ bits ($L = 2^n$ levels):
$$\Delta = \frac{2 V_{\max}}{2^n} = 2^{1-n} V_{\max}$$
In a **mid-rise** quantizer, the reconstruction levels are located at the center of each interval:
$$\hat{x}_k = -V_{\max} + (k + 0.5) \Delta, \quad k \in \{0, 1, \dots, 2^n - 1\}$$

#### Derivation of Theoretical SQNR: $6.02n + 1.76\text{ dB}$
Assume quantization error $e = x - \hat{x}$ is uniformly distributed in $[-\Delta/2, +\Delta/2]$:
$$\sigma_q^2 = \mathbb{E}[e^2] = \frac{1}{\Delta} \int_{-\Delta/2}^{\Delta/2} e^2 de = \frac{\Delta^2}{12} = \frac{(2 V_{\max} / 2^n)^2}{12} = \frac{V_{\max}^2 2^{-2n}}{3}$$
For a full-scale sinusoid $x(t) = V_{\max} \sin(2\pi f t)$, the average signal power is:
$$P_s = \frac{V_{\max}^2}{2}$$
The Signal-to-Quantization-Noise Ratio (SQNR) is:
$$\text{SQNR} = \frac{P_s}{\sigma_q^2} = \frac{V_{\max}^2 / 2}{V_{\max}^2 2^{-2n} / 3} = \frac{3}{2} \cdot 2^{2n} = 1.5 \cdot 2^{2n}$$
Expressing this in decibels:
$$\text{SQNR}_{\text{dB}} = 10 \log_{10}(1.5) + 20 n \log_{10}(2) = 1.7609 + 6.0206 n \text{ dB}$$
> **Viva Takeaway:** Every additional bit of quantization increases the SQNR by approximately **6 dB** (a factor of 4 in noise power reduction).

### 3.2 Non-Uniform Companding: ITU-T G.711 $\mu$-Law
For speech and audio signals, small amplitudes occur much more frequently than large peaks. A uniform quantizer yields poor SQNR for quiet passages because step size $\Delta$ remains large. Companding compresses the dynamic range before uniform quantization, and expands it at the receiver:
- **$\mu$-Law Compression ($\mu = 255$):**
  $$y = \operatorname{sgn}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}, \quad |x| \le 1.0$$
- **$\mu$-Law Expansion (Inverse):**
  $$\hat{x} = \operatorname{sgn}(y) \frac{(1 + \mu)^{|y|} - 1}{\mu}, \quad |y| \le 1.0$$
Companding maintains an approximately constant SQNR over a wide 40 dB dynamic range.

### 3.3 Sampling Theorem & Aliasing
- **Nyquist-Shannon Sampling Theorem:** An analog signal with bandwidth $f_{\max}$ can be uniquely reconstructed if and only if sampling rate $F_s \ge 2 f_{\max}$.
- **Aliasing:** When $F_s < 2 f_0$, higher frequencies fold back into the baseband $[0, F_s/2]$. The perceived alias frequency is:
  $$f_{\text{alias}} = |f_0 - k F_s|, \quad k = \operatorname{round}(f_0 / F_s)$$

### 3.4 Line Coding
Line codes shape digital bit patterns for transmission over electrical baseband channels:
1. **NRZ-L (Non-Return-to-Zero Level):**
   - Bit 1 $\to +V$, Bit 0 $\to -V$. Simple, but possesses high DC content and lacks self-clocking on long strings of identical bits.
2. **RZ (Return-to-Zero):**
   - Bit 1 $\to +V$ for $T_b/2$ then $0$; Bit 0 $\to 0$. Requires double the bandwidth ($2 R_b$).
3. **Manchester (Biphase):**
   - Bit 1 $\to +V$ then $-V$; Bit 0 $\to -V$ then $+V$. Strictly **zero DC content** and guaranteed mid-bit transition for robust clock recovery. Used in 10BASE-T Ethernet.
4. **AMI (Alternate Mark Inversion):**
   - Bit 0 $\to 0$; Bit 1 alternates between $+V$ and $-V$. Eliminates DC bias and provides built-in error detection (bipolar violations). Used in T1/E1 telecommunications.

### 3.5 Pulse Shaping & Root-Raised Cosine (RRC)
Transmitting ideal rectangular pulses requires infinite bandwidth ($\operatorname{sinc}$ spectrum). Passing rectangular pulses through a band-limited channel causes pulse spreading into adjacent time slots, known as **Inter-Symbol Interference (ISI)**.
- **Nyquist's First Criterion:** To achieve zero ISI, the cascaded filter response $p(t) = h_{TX}(t) * h_{RX}(t)$ must have zero-crossings at all non-zero integer symbol multiples: $p(n T_s) = 0$ for $n \ne 0$.
- **Raised Cosine (RC) Spectrum:**
  $$P(f) = \begin{cases} 
  T_s, & |f| \le \frac{1-\alpha}{2T_s} \\
  \frac{T_s}{2} \left[1 + \cos\left(\frac{\pi T_s}{\alpha} \left(|f| - \frac{1-\alpha}{2T_s}\right)\right)\right], & \frac{1-\alpha}{2T_s} < |f| \le \frac{1+\alpha}{2T_s} \\
  0, & |f| > \frac{1+\alpha}{2T_s}
  \end{cases}$$
  where $\alpha \in [0, 1]$ is the roll-off factor. Bandwidth is $B = \frac{1+\alpha}{2 T_s}$.
- **Matched Filter Pair (RRC):** The Raised Cosine response is split equally between transmitter and receiver:
  $$H_{TX}(f) = H_{RX}(f) = \sqrt{P(f)}$$
  The receiver filter $H_{RX}(f)$ is the matched filter, maximizing SNR in AWGN while the cascade $H_{TX} \cdot H_{RX} = P(f)$ satisfies Nyquist's zero-ISI criterion.
- **Eye Diagram:** Superimposing overlapping 2-symbol signal segments reveals:
  - **Eye Opening Height:** Proportional to noise margin.
  - **Zero-Crossing Width:** Sensitivity to timing jitter.
  - **Closure:** Degradation due to ISI.

---

## 4. Module B: Modulation & Detection Lab (Student 2)

### 4.1 Advanced Complex Baseband Modulations
1. **8-PSK (M = 8):**
   - Transmits $k = \log_2(8) = 3$ bits per symbol.
   - Constellation points lie on unit circle: $s_m = e^{j 2\pi m / 8}$, $m \in \{0, \dots, 7\}$.
   - Gray mapping ensures nearest-neighbor phase errors cause only 1 bit error.
   - Theoretical BER:
     $$P_b \approx \frac{2}{\log_2 M} Q\left(\sqrt{2 \log_2 M \frac{E_b}{N_0}} \sin\frac{\pi}{M}\right) = \frac{2}{3} Q\left(\sqrt{6 \frac{E_b}{N_0}} \sin\frac{\pi}{8}\right)$$

2. **16-QAM (M = 16):**
   - Transmits $k = 4$ bits per symbol on a $4 \times 4$ rectangular grid: $I, Q \in \{-3, -1, +1, +3\}$.
   - Average symbol energy for unscaled grid: $E_s = \frac{1}{16}\sum (I^2 + Q^2) = 10.0$.
   - Normalized scaling: $1 / \sqrt{10}$.
   - Independent I and Q slicing allows ultra-fast ML detection.
   - Theoretical BER:
     $$P_b \approx \frac{3}{4} Q\left(\sqrt{\frac{4}{5} \frac{E_b}{N_0}}\right) = 0.75 Q\left(\sqrt{0.8 \frac{E_b}{N_0}}\right)$$

3. **Non-Coherent DBPSK (Differential BPSK):**
   - Encodes data in the phase *difference* between adjacent symbols: $\phi_k = \phi_{k-1} + \pi b_k$.
   - Demodulated by computing the dot product $r_k \cdot r_{k-1}^*$. No phase locked loop (PLL) needed!
   - Exact Theoretical BER:
     $$P_b = \frac{1}{2} e^{-E_b / N_0}$$
     Experiences only a $\approx 0.9\text{ dB}$ power penalty relative to coherent BPSK at high SNR.

4. **Non-Coherent BFSK:**
   - Detects signal envelope on two orthogonal tone branches without carrier phase knowledge:
     $$\hat{b} = 1 \iff |r_1| > |r_2|$$
   - Branch 1 has Rician distribution; branch 2 has Rayleigh distribution.
   - Exact Theoretical BER:
     $$P_b = \frac{1}{2} e^{-E_b / (2 N_0)}$$

### 4.2 First-Principles Gram-Schmidt Orthogonalization
Gram-Schmidt converts an arbitrary set of $M$ energy signals $\{s_1(t), \dots, s_M(t)\}$ into a minimal orthonormal basis $\{\phi_1(t), \dots, \phi_D(t)\}$ where $D \le M$:
1. **First Basis:** $\psi_1(t) = s_1(t)$, $\phi_1(t) = \frac{\psi_1(t)}{\sqrt{\int \psi_1^2(t) dt}}$.
2. **Subsequent Bases ($k = 2, \dots, M$):**
   $$\psi_k(t) = s_k(t) - \sum_{j=1}^{d} c_{kj} \phi_j(t), \quad \text{where } c_{kj} = \int s_k(t) \phi_j(t) dt$$
   If $\int \psi_k^2(t) dt > \epsilon$:
   $$\phi_{d+1}(t) = \frac{\psi_k(t)}{\sqrt{\int \psi_k^2(t) dt}}, \quad d \leftarrow d + 1$$
3. **Signal Space Representation:**
   - Vector coordinates: $\mathbf{s}_i = [s_{i1}, s_{i2}, \dots, s_{iD}]$
   - Signal energy: $E_i = \|\mathbf{s}_i\|^2 = \sum_j s_{ij}^2$
   - Pairwise Euclidean distance: $d_{ik} = \|\mathbf{s}_i - \mathbf{s}_k\|$
   - Exact reconstruction: $\hat{s}_i(t) = \sum_{j=1}^D s_{ij} \phi_j(t) = s_i(t)$.

### 4.3 Carrier Phase Offset Impairment
If receiver carrier recovery has a phase error $\phi$, the received symbol is rotated:
$$r = s e^{j\phi} + w$$
For BPSK, projecting onto the real axis gives decision metric $z = \pm \sqrt{E_b} \cos\phi + w_I$.
The effective signal amplitude is reduced by $\cos\phi$:
$$P_b(\phi) = Q\left(\sqrt{\frac{2E_b}{N_0}} \cos\phi\right)$$
- At $\phi = 0^\circ$: Nominal performance $Q(\sqrt{2E_b/N_0})$.
- At $\phi = 90^\circ$: $\cos(90^\circ) = 0 \implies Q(0) = 0.5$ (complete link failure).
- At $\phi = 180^\circ$: Decision inverted (100% bit errors).

---

## 5. Module C: Channel, Coding & Integration Lab (Student 3)

### 5.1 Flat Rayleigh Fading Channel
In mobile environments without a Line-of-Sight (LoS) path, multi-path reflections arrive with random phases. By the Central Limit Theorem, the in-phase and quadrature components are independent Gaussians:
$$h = h_I + j h_Q, \quad h_I, h_Q \sim \mathcal{N}(0, 0.5)$$
The channel envelope $\alpha = |h| = \sqrt{h_I^2 + h_Q^2}$ follows a **Rayleigh distribution**:
$$f_\alpha(\alpha) = 2\alpha e^{-\alpha^2}, \quad \alpha \ge 0, \quad \mathbb{E}[\alpha^2] = 1.0$$
The instantaneous received SNR is $\gamma = \alpha^2 \frac{E_b}{N_0}$.
Averaging the AWGN conditional error probability over the exponential distribution of $\gamma$:
$$P_b = \int_0^\infty Q(\sqrt{2\gamma}) \frac{1}{\bar{\gamma}_b} e^{-\gamma / \bar{\gamma}_b} d\gamma = \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}_b}{1 + \bar{\gamma}_b}}\right)$$
where $\bar{\gamma}_b = E_b / N_0$.
> **Viva Takeaway:** In AWGN, error probability drops *exponentially* with SNR. In Rayleigh fading, $P_b \approx \frac{1}{4 \bar{\gamma}_b}$ drops only *inversely* ($10\text{ dB per decade}$). At $P_b = 10^{-3}$, Rayleigh fading requires a **17 dB higher SNR** than AWGN!

### 5.2 Forward Error Correction (FEC)

#### 1. Repetition(3) Code ($R = 1/3$)
Repeats each bit 3 times. Majority logic decoder corrects any single bit error per 3-bit codeword:
$$\hat{b} = \operatorname{mode}(r_1, r_2, r_3)$$

#### 2. Systematic Hamming(7,4) Code ($R = 4/7$)
Encodes $k=4$ information bits into $n=7$ coded bits using generator matrix $G = [I_4 | P]$:
$$\mathbf{c} = \mathbf{m} G \pmod 2$$
The receiver computes the 3-bit syndrome vector using parity check matrix $H = [P^T | I_3]$:
$$\mathbf{s} = \mathbf{r} H^T \pmod 2$$
- If $\mathbf{s} = [0, 0, 0]$: No error detected.
- If $\mathbf{s} \ne [0, 0, 0]$: The syndrome matches column $j$ of $H$. The receiver flips bit $r[j]$ to restore the exact message. Corrects $t = 1$ bit error.

#### 3. Rate-1/2 $K=3$ Convolutional Code with Viterbi Decoding
- Constraint length $K=3$ (2-bit shift register, 4 states: `00`, `01`, `10`, `11`).
- Generators: $g_1 = 7_8 = (111)_2$, $g_2 = 5_8 = (101)_2$.
- Encodes with rate $R = 1/2$. Flushed with $K-1=2$ zeros.
- **Viterbi Trellis Algorithm:**
  1. **Branch Metric:** Computes Hamming distance between received 2-bit channel symbol and expected branch outputs.
  2. **Add-Compare-Select (ACS):** For each of the 4 destination states, adds branch metric to previous path metrics, compares the two arriving paths, and selects the survivor path with minimum accumulated metric.
  3. **Traceback:** Starting from state `00` at the end of the trellis, traces backwards through survivor pointers to reconstruct the maximum-likelihood information bit sequence.

#### 4. Coding Gain and Energy Dilution
Transmitting $N$ info bits with code rate $R$ generates $N/R$ channel symbols. Under the global convention $E_b = \sum |x|^2 / N_{\text{info}}$, energy per channel symbol is diluted to $R \cdot E_b$.
- At low SNR, coded BER is *higher* than uncoded BER because the noise power overcomes the code's error-correcting capability while energy per symbol is smaller.
- Above the coding threshold SNR, the slope of the coded BER curve is much steeper, yielding positive **coding gain** (typically $2\text{ to }3.5\text{ dB}$ at $\text{BER} = 10^{-4}$).

### 5.3 Matrix Block Interleaver
In fading channels, deep fades last multiple symbols, creating bursts of errors. A rectangular $N_{\text{rows}} \times N_{\text{cols}}$ matrix interleaver writes coded bits by rows and transmits them by columns. The deinterleaver reverses this operation, scattering contiguous burst errors across widely separated codewords, allowing Hamming and Viterbi decoders to correct them as isolated single-bit errors.

### 5.4 End-to-End Transceiver Quality Metrics
- **Bit Error Rate (BER):** $P_b = \frac{\text{Number of Mismatched Bits}}{\text{Total Transmitted Bits}}$
- **Peak Signal-to-Noise Ratio (PSNR) for Images:**
  $$\text{MSE} = \frac{1}{M N} \sum_{i,j} (I[i,j] - \hat{I}[i,j])^2, \quad \text{PSNR}_{\text{dB}} = 10 \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$$
  Lossless reconstruction yields $\text{MSE} = 0 \implies \text{PSNR} = 100\text{ dB}$.
- **Signal-to-Quantization-Noise Ratio (SQNR) for Audio:**
  $$\text{SQNR}_{\text{dB}} = 10 \log_{10}\left(\frac{\sum |x[k]|^2}{\sum |x[k] - \hat{x}[k]|^2}\right)$$
