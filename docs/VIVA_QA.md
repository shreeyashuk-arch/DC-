# Digital Communication Lab Suite: 40 Viva Voce Questions & Answers

A master examination preparation guide grouped by module and student ownership. Every answer is concise, mathematically accurate, and tailored to impress external examiners.

---

## Section 1: General & Core Link Conventions (Questions 1 to 10)

#### Q1. Why is the fundamental SNR parameter in digital communications $E_b / N_0$ rather than $S / N$?
**Answer:** Signal-to-noise ratio ($S/N$) depends on transmission bandwidth and data rate. $E_b / N_0$ (energy per information bit divided by noise power spectral density) normalizes performance to raw throughput, enabling fair, standardized comparisons across modulations with different alphabet sizes and coding rates.

#### Q2. Why must $E_b$ be measured per *information* bit rather than per *coded* bit or *symbol*?
**Answer:** A channel code dilutes energy across redundant parity bits. Measuring $E_b$ per information bit ensures the code is penalized for rate expansion ($R \cdot E_b$ per coded bit) and only rewarded if its error-correcting capability overcomes this energy reduction (true coding gain).

#### Q3. How is $E_b$ measured directly from transmitted discrete samples?
**Answer:** Using the energy summation formula:
$$E_b = \frac{\sum_{k=1}^N |x[k]|^2}{N_{\text{info\_bits}}}$$
This automatically accounts for oversampling factors, carrier amplitudes, and pulse-shaping filters without empirical fudge factors.

#### Q4. What is the noise variance per real dimension for noise spectral density $N_0$?
**Answer:** The noise variance per real dimension is $\sigma^2 = N_0 / 2$. For a complex baseband signal, noise has variance $N_0 / 2$ on in-phase (I) and $N_0 / 2$ on quadrature (Q), giving total variance $N_0$.

#### Q5. What is the mathematical definition of the Gaussian Q-function?
**Answer:** $Q(x)$ is the tail probability of a standard normal distribution:
$$Q(x) = \frac{1}{\sqrt{2\pi}} \int_x^\infty e^{-u^2 / 2} du = \frac{1}{2} \operatorname{erfc}\left(\frac{x}{\sqrt{2}}\right)$$

#### Q6. What is the theoretical bit error rate (BER) of BPSK in an AWGN channel?
**Answer:** $P_b = Q\left(\sqrt{\frac{2E_b}{N_0}}\right)$.

#### Q7. Why does QPSK achieve the exact same theoretical BER as BPSK?
**Answer:** QPSK transmits on two mutually orthogonal quadrature carriers ($\cos$ and $\sin$). Because $\int \cos(2\pi f_c t) \sin(2\pi f_c t) dt = 0$, the in-phase and quadrature correlators decouple into two independent BPSK links, each receiving energy $E_b$ per bit.

#### Q8. What is the theoretical BER of coherent BFSK, and why is it 3 dB worse than BPSK?
**Answer:** $P_b = Q\left(\sqrt{\frac{E_b}{N_0}}\right)$. The Euclidean distance between orthogonal BFSK signals is $d = \sqrt{2E_b}$, whereas for antipodal BPSK it is $d = 2\sqrt{E_b}$. Because distance is halved in squared terms, BFSK requires 3 dB more power.

#### Q9. How does an optimal correlator receiver decide between binary hypotheses?
**Answer:** It projects the received waveform onto the orthonormal basis functions: $y_j = \int r(t) \phi_j(t) dt$. The decision statistic $y$ is compared against an optimal threshold (0 for antipodal BPSK, halfway between signals for OOK).

#### Q10. What is Gray coding, and why is it critical in M-ary modulation?
**Answer:** Gray coding arranges adjacent constellation points to differ by exactly one binary bit. Consequently, when noise causes a symbol error to an adjacent decision region, it results in only a single bit error, minimizing overall BER: $P_b \approx P_s / \log_2 M$.

---

## Section 2: Module A — Source & Baseband Lab (Questions 11 to 20, Student 1)

#### Q11. What are the three fundamental operations in Pulse Code Modulation (PCM)?
**Answer:**
1. **Sampling:** Discretizes the continuous-time signal at rate $F_s \ge 2 f_{\max}$.
2. **Quantization:** Rounds continuous sample amplitudes into $2^n$ discrete voltage levels.
3. **Encoding:** Maps each quantization level to an $n$-bit binary codeword.

#### Q12. Derive the theoretical SQNR formula for an $n$-bit uniform quantizer on a full-scale sinusoid.
**Answer:** Step size $\Delta = 2 V_{\max} / 2^n$. Noise power $\sigma_q^2 = \Delta^2 / 12 = V_{\max}^2 2^{-2n} / 3$. Sinusoid power $P_s = V_{\max}^2 / 2$.
$$\text{SQNR} = \frac{P_s}{\sigma_q^2} = 1.5 \cdot 2^{2n} \implies \text{SQNR}_{\text{dB}} = 10\log_{10}(1.5) + 20n\log_{10}(2) = 1.76 + 6.02n \text{ dB}$$

#### Q13. What is the "6 dB per bit" rule in PCM?
**Answer:** Increasing quantizer resolution by 1 bit doubles the number of levels ($2^{n+1}$), halving the step size $\Delta$, which reduces quantization noise power by a factor of 4 ($10 \log_{10} 4 \approx 6.02\text{ dB}$).

#### Q14. What is the difference between a mid-tread and a mid-rise quantizer?
**Answer:**
- **Mid-tread:** Has a quantization level exactly at zero (suitable for signals with silence/zero bias).
- **Mid-rise:** Has a decision threshold at zero, with symmetric non-zero levels on either side.

#### Q15. Why is non-uniform companding ($\mu$-law) necessary in speech telecommunications?
**Answer:** Voice signals spend over 80% of their duration at low amplitudes. A uniform quantizer yields poor SQNR for quiet sounds. The logarithmic $\mu$-law compressor ($\mu = 255$) amplifies low amplitudes, keeping SQNR roughly constant across a 40 dB dynamic range.

#### Q16. State the Nyquist Sampling Theorem and explain the consequence of under-sampling.
**Answer:** A band-limited signal with maximum frequency $f_m$ must be sampled at $F_s \ge 2 f_m$. Sampling below $2 f_m$ causes **aliasing**, where high frequencies fold back into baseband at $f_{\text{alias}} = |f_0 - k F_s|$, distorting the signal irreversibly.

#### Q17. Compare Manchester coding with NRZ-L in terms of DC content and bandwidth.
**Answer:**
- **Manchester:** Zero DC content and guaranteed mid-bit transition for clock recovery; requires bandwidth $2 R_b$.
- **NRZ-L:** High DC content and poor clock recovery on long strings of identical bits; requires half the bandwidth ($R_b/2$).

#### Q18. How does Alternate Mark Inversion (AMI) provide built-in error detection?
**Answer:** In AMI, successive binary 1s must alternate in polarity ($+V, -V, +V, \dots$). If noise corrupts a bit causing two consecutive pulses of identical polarity, a **bipolar violation** is flagged immediately without parity bits.

#### Q19. What is Nyquist's First Criterion for zero Inter-Symbol Interference (ISI)?
**Answer:** The overall channel impulse response $p(t)$ must pass through zero at all non-zero integer symbol intervals:
$$p(n T_s) = \begin{cases} 1, & n = 0 \\ 0, & n \ne 0 \end{cases}$$

#### Q20. What features of an eye diagram reveal noise margin and timing jitter sensitivity?
**Answer:**
- **Eye Opening Height:** Directly measures the noise margin before an error occurs.
- **Slope / Zero-Crossing Width:** Measures sensitivity to receiver timing jitter and clock phase errors.

---

## Section 3: Module B — Modulation & Detection Lab (Questions 21 to 30, Student 2)

#### Q21. What is the spectral efficiency of 8-PSK and 16-QAM?
**Answer:**
- 8-PSK: $k = \log_2(8) = 3$ bits/symbol (3.0 bps/Hz nominal).
- 16-QAM: $k = \log_2(16) = 4$ bits/symbol (4.0 bps/Hz nominal).

#### Q22. Why is 16-QAM more power-efficient than 16-PSK?
**Answer:** 16-PSK crams all 16 constellation points onto a 1D circle, dramatically reducing Euclidean distance between neighbors ($d_{\min} \approx 2 \sin(\pi/16)$). 16-QAM distributes points uniformly across a 2D square grid, maintaining a much larger $d_{\min}$ for the same average energy (saving $> 4\text{ dB}$).

#### Q23. Why is 8-PSK preferred over 8-QAM or 16-QAM in satellite communications?
**Answer:** 8-PSK has a **strictly constant envelope** ($\text{PAPR} = 0\text{ dB}$), allowing RF power amplifiers (TWTAs) on satellites to operate in their non-linear saturation region at maximum electrical efficiency without causing intermodulation distortion.

#### Q24. How does non-coherent DBPSK demodulation work without carrier recovery?
**Answer:** The receiver multiplies the current received symbol with the complex conjugate of the previous symbol: $z_k = r_k r_{k-1}^*$. The phase of $z_k$ represents the transmitted phase difference $\Delta\phi \in \{0, \pi\}$. A positive real part indicates bit 0; negative indicates bit 1.

#### Q25. What is the BER expression for non-coherent orthogonal BFSK?
**Answer:** $P_b = \frac{1}{2} e^{-E_b / (2 N_0)}$.

#### Q26. State the fundamental objective of the Gram-Schmidt Orthogonalization Procedure.
**Answer:** To construct an orthonormal basis $\{\phi_1(t), \dots, \phi_D(t)\}$ from $M$ arbitrary energy waveforms such that every signal can be expressed as a coordinate vector $\mathbf{s}_i = [s_{i1}, \dots, s_{iD}]$ with minimal dimensionality $D \le M$.

#### Q27. In Gram-Schmidt, how do we verify if an input signal is linearly dependent?
**Answer:** We subtract its projections onto all existing basis vectors: $\psi_k(t) = s_k(t) - \sum \langle s_k, \phi_j \rangle \phi_j(t)$. If the residual energy $\int \psi_k^2(t) dt < \epsilon \approx 10^{-7}$, the signal is linearly dependent and no new basis function is added.

#### Q28. How is the energy of a signal calculated from its Gram-Schmidt coordinates?
**Answer:** By Parseval's relation in signal space:
$$E_i = \|\mathbf{s}_i\|^2 = \sum_{j=1}^D s_{ij}^2$$

#### Q29. What is the impact of a carrier phase offset $\phi$ on a coherent BPSK receiver?
**Answer:** The decision statistic is attenuated by $\cos\phi$, reducing effective SNR:
$$P_b = Q\left(\sqrt{\frac{2E_b}{N_0}} \cos\phi\right)$$
At $\phi = 90^\circ$, $\cos(90^\circ) = 0 \implies P_b = Q(0) = 0.5$ (complete breakdown).

#### Q30. Which modulation scheme should be recommended for a bandwidth-limited channel with severe phase jitter?
**Answer:** **DBPSK**. It avoids carrier phase tracking failures completely through differential detection while preserving 1.0 bps/Hz spectral efficiency (unlike non-coherent BFSK, which requires twice the bandwidth).

---

## Section 4: Module C — Channel, Coding & Integration Lab (Questions 31 to 40, Student 3)

#### Q31. What is flat Rayleigh fading, and what physical environment does it model?
**Answer:** It models multipath propagation where multiple indirect reflections arrive at the receiver without a dominant Line-of-Sight (LoS) path. The channel envelope $|h|$ is Rayleigh distributed, causing deep constructive and destructive signal fades.

#### Q32. What is the theoretical average BER for BPSK in flat Rayleigh fading?
**Answer:**
$$P_b = \frac{1}{2}\left(1 - \sqrt{\frac{\bar{\gamma}_b}{1 + \bar{\gamma}_b}}\right), \quad \text{where } \bar{\gamma}_b = \frac{E_b}{N_0}$$
At high SNR, $P_b \approx \frac{1}{4 \bar{\gamma}_b}$, which decays only linearly ($10\text{ dB/decade}$) compared to exponential decay in AWGN.

#### Q33. Why is there a 17 dB gap between AWGN and Rayleigh fading at $\text{BER} = 10^{-3}$?
**Answer:** In AWGN, noise is additive and independent of the signal. In Rayleigh fading, random deep fades occasionally reduce the received signal amplitude near zero, where even minor Gaussian noise flips the decision.

#### Q34. How does systematic Hamming(7,4) code achieve single-bit error correction?
**Answer:** It encodes 4 information bits into 7 coded bits ($R = 4/7, d_{\min} = 3$). The receiver computes the 3-bit syndrome $\mathbf{s} = \mathbf{r} H^T \pmod 2$. Each of the 7 non-zero syndrome values uniquely matches one column of parity-check matrix $H$, identifying the exact position of the corrupted bit for single-bit flipping.

#### Q35. What is the code rate and constraint length of the convolutional code implemented?
**Answer:** Code rate $R = 1/2$ (2 coded bits per information bit), constraint length $K = 3$ (shift register length of 2 delay elements, giving $2^{K-1} = 4$ states), with octal generator polynomials $g_1 = 7_8 = (111)_2$ and $g_2 = 5_8 = (101)_2$.

#### Q36. Explain the Add-Compare-Select (ACS) recursion in the Viterbi algorithm.
**Answer:** At each trellis step, for each destination state:
1. **Add:** Adds the branch Hamming metric to the predecessor state's path metric.
2. **Compare:** Compares the candidate metrics from the two entering branches.
3. **Select:** Selects the path with minimum metric as the survivor, storing its predecessor pointer for subsequent traceback.

#### Q37. What is coding gain, and why is coded BER higher than uncoded BER at low SNR?
**Answer:** Coding gain is the reduction in required $E_b/N_0$ achieved by coding at a given target BER. At low SNR, code rate expansion dilutes energy ($R \cdot E_b$ per coded symbol), and channel errors exceed the code's correction threshold ($t$), worsening BER. Above the threshold SNR, error correction dominates and yields positive coding gain.

#### Q38. How does a block interleaver combat burst errors in Rayleigh fading channels?
**Answer:** A deep fade can wipe out a burst of consecutive channel symbols. A matrix block interleaver writes bits by rows and reads them by columns for transmission. The deinterleaver reverses this, spreading contiguous burst errors across multiple separate codewords so the channel decoder only encounters isolated, single-bit errors.

#### Q39. What is PSNR, and what does a value of 100 dB signify in the transceiver simulator?
**Answer:** Peak Signal-to-Noise Ratio (PSNR) measures reconstructed image fidelity:
$$\text{PSNR} = 10 \log_{10}\left(\frac{255^2}{\text{MSE}}\right)$$
When transmission is 100% error-free, $\text{MSE} = 0$, which the simulator clamps to a nominal maximum of $100.0\text{ dB}$ (lossless recovery).

#### Q40. Trace the complete path of an audio signal through the integrated transceiver pipeline.
**Answer:**
$$\text{Audio Waveform} \xrightarrow{\text{PCM}} \text{Quantized Bits} \xrightarrow{\text{FEC}} \text{Coded Bits} \xrightarrow{\text{Interleaver}} \text{Interleaved Bits} \xrightarrow{\text{Modulator}} \text{Passband/Baseband Symbols} \xrightarrow{\text{Channel}} \text{Noisy Faded Signal} \xrightarrow{\text{Demodulator}} \text{Channel Bits} \xrightarrow{\text{Deinterleaver}} \text{Deinterleaved Bits} \xrightarrow{\text{Viterbi Decoder}} \text{Decoded Bits} \xrightarrow{\text{PCM Expansion}} \text{Reconstructed Audio}$$
