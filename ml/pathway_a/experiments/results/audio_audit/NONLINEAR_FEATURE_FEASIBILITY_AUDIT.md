# Stage 6C: Non-Linear Feature Reconstruction Feasibility Audit
**Parkinson's AI Screening | Pathway A (UCI Dataset Only)**

---

## Executive Summary

Stage 6B confirmed that the 16 linear acoustic features extractable via standard software (Praat/Parselmouth) achieve only **0.6667** participant-level balanced accuracy, missing **29.2% of Parkinson's cases** and failing our validation threshold ($\ge 0.70$). This deficit directly stems from the exclusion of the **6 non-linear dynamical features** (`spread1`, `PPE`, `spread2`, `RPDE`, `DFA`, `D2`), which Stage 4 SHAP analysis identified as accounting for **over 60% of champion model attribution**.

This Stage 6C audit investigates the mathematical feasibility, algorithmic implementations, dependencies, and validation constraints of reconstructing these 6 non-linear features directly from raw microphone audio to satisfy the 22-feature champion model contract.

> [!CRITICAL]
> **Key Finding — The Ground-Truth Data Limitation:**
> The public UCI Parkinson's dataset (Little et al., 2007, 2008) contains **only the tabular feature matrix** (`parkinsons.data`) and documentation (`parkinsons.names`). The original raw `.wav` recordings from the National Centre for Voice and Speech (Denver, Colorado) were **never publicly distributed**. Consequently, paired ground-truth validation `(raw_audio, original_feature_values)` is mathematically unavailable within Pathway A. Any new Python implementation cannot be directly verified against the original audio recordings that generated the UCI matrix.

---

## 1. Missing-Feature Inventory

All 6 features were introduced to vocal disorder analysis by Max Little et al. to detect non-linear attractor dynamics and pitch variations that traditional perturbation metrics (jitter/shimmer) miss.

| Feature Name | UCI Column Alias | Category | Clinical / Biophysical Rationale | UCI Range [`min`, `max`] | UCI Mean $\pm$ Std | SHAP Rank | Mean Abs SHAP |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **`spread1`** | `spread1` | Non-linear F0 Variation | Logarithmic tail spread of semitone pitch variations; captures extreme dysphonic frequency excursions. | $[-7.965, -2.434]$ | $-5.684 \pm 1.090$ | **#1** | **0.04394** |
| **`PPE`** | `PPE` | Pitch Period Entropy | Shannon entropy of pitch period perturbations on logarithmic semitone scales; insensitive to normal prosody. | $[0.045, 0.527]$ | $0.207 \pm 0.090$ | **#2** | **0.02910** |
| **`spread2`** | `spread2` | Non-linear F0 Variation | Second spectral dispersion parameter of semitone pitch variations; captures asymmetrical distribution spread. | $[0.006, 0.450]$ | $0.227 \pm 0.083$ | **#4** | **0.02387** |
| **`DFA`** | `DFA` | Fractal Dynamics | Detrended Fluctuation Analysis scaling exponent $\alpha$; quantifies turbulent acoustic airflow self-similarity. | $[0.574, 0.825]$ | $0.718 \pm 0.055$ | **#13** | **0.01180** |
| **`D2`** | `D2` | Chaos / Attractor | Grassberger-Procaccia correlation dimension; estimates fractal dimension of vocal fold chaotic attractor. | $[1.423, 3.671]$ | $2.382 \pm 0.383$ | **#14** | **0.01007** |
| **`RPDE`** | `RPDE` | Recurrence Chaos | Recurrence Period Density Entropy; uncertainty in recurrence times in reconstructed phase space. | $[0.257, 0.685]$ | $0.499 \pm 0.104$ | **#15** | **0.00997** |

---

## 2. Mathematical and Algorithmic Reference

### 2.1 RPDE (Recurrence Period Density Entropy)
- **Reference:** Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM (2007), *BioMedical Engineering OnLine*.
- **Mathematical Formulation:**
  1. Construct time-delay phase-space trajectory:
     $$\mathbf{x}_n = [s(n), s(n+\tau), s(n+2\tau), \dots, s(n+(m-1)\tau)]^T$$
     where $m$ is the embedding dimension and $\tau$ is the time delay.
  2. For each embedded state $\mathbf{x}_n$, identify recurrences when trajectories enter an $\epsilon$-radius hypersphere:
     $$\|\mathbf{x}_n - \mathbf{x}_k\| < \epsilon$$
  3. Compute recurrence times $T_k = t_{k+1} - t_k$ (time between consecutive entries).
  4. Form normalized histogram of recurrence periods $P(T) = N(T) / \sum_t N(t)$.
  5. Compute normalized Shannon entropy:
     $$H_{\text{norm}} = -\frac{1}{\ln(T_{\max})} \sum_{t=1}^{T_{\max}} P(t) \ln P(t)$$
- **Algorithmic Sensitivities:**
  - Highly dependent on $\epsilon$ (recurrence radius). Little et al. used a fixed recurrence rate (e.g. $r_{\text{rate}} = 0.12$).
  - Delay $\tau$ must be set to the first minimum of mutual information or first zero of autocorrelation ($T_0$).
  - Any mismatch in $\epsilon$ or $\tau$ estimation shifts $RPDE$ by $20\text{--}40\%$.

### 2.2 DFA (Detrended Fluctuation Analysis)
- **Reference:** Peng CK et al. (1995), Little et al. (2007).
- **Mathematical Formulation:**
  1. Integrate the zero-mean audio signal:
     $$y(k) = \sum_{i=1}^k [s(i) - \bar{s}]$$
  2. Divide $y(k)$ into $N_n = \lfloor N/n \rfloor$ non-overlapping windows of length $n$.
  3. Fit a least-squares polynomial (typically linear, order 1) $y_n(k)$ to each segment.
  4. Compute the root-mean-square fluctuation:
     $$F(n) = \sqrt{\frac{1}{N} \sum_{k=1}^N [y(k) - y_n(k)]^2}$$
  5. Compute $F(n)$ over a scaling grid $n \in [n_{\min}, n_{\max}]$ and determine scaling exponent $\alpha$:
     $$\log F(n) \approx \alpha \log n + C$$
- **Algorithmic Sensitivities:**
  - $n_{\min}$ must exceed the fundamental pitch period ($\sim 2 \times T_0$) to prevent local pitch harmonic oscillations from corrupting the fractal scaling slope.
  - Scale range $[n_{\min}, n_{\max}]$ selection directly governs the resulting $\alpha$.

### 2.3 D2 (Correlation Dimension)
- **Reference:** Grassberger P, Procaccia I (1983); Little et al. (2007).
- **Mathematical Formulation:**
  1. In reconstructed phase space $\mathbb{R}^m$, compute the correlation integral:
     $$C(r, m) = \frac{2}{M(M-1)} \sum_{i=1}^M \sum_{j=i+W}^M \Theta(r - \|\mathbf{x}_i - \mathbf{x}_j\|)$$
     where $W$ is the Theiler window to exclude temporal autocorrelations.
  2. In the scaling region:
     $$D_2(m) = \lim_{r \to 0} \frac{d \ln C(r, m)}{d \ln r}$$
  3. $D_2$ is estimated as the saturation plateau of $D_2(m)$ as embedding dimension $m$ increases.
- **Algorithmic Sensitivities:**
  - Time complexity $O(M^2)$ pairwise distance comparisons.
  - High susceptibility to background room noise: additive Gaussian noise introduces an artificial infinite-dimensional scaling at small $r$, obliterating the true plateau.

### 2.4 PPE (Pitch Period Entropy)
- **Reference:** Little MA, McSharry PE, Hunter EJ, Ramig LO (2009), *IEEE Transactions on Biomedical Engineering*.
- **Mathematical Formulation:**
  1. Extract fundamental frequency sequence $F_0(i)$ using high-resolution pitch tracking.
  2. Transform $F_0(i)$ to logarithmic semitones relative to a standard pitch reference ($f_{\text{ref}} = 50\text{ Hz}$):
     $$p(i) = 12 \log_2 \left(\frac{F_0(i)}{f_{\text{ref}}}\right)$$
  3. Remove slow prosodic pitch drift using a baseline smoothing filter (e.g., local polynomial or median filter $\tilde{p}(i)$):
     $$r(i) = p(i) - \tilde{p}(i)$$
  4. Discretize residual fluctuations $r(i)$ into bins of width $\delta = 0.05$ semitones:
     $$P(k) = \frac{\text{count}(r(i) \in \text{bin}_k)}{N}$$
  5. Calculate Shannon entropy normalized by maximum theoretical entropy:
     $$PPE = -\sum_{k} P(k) \log_2 P(k)$$
- **Algorithmic Sensitivities:**
  - Insensitive to macroscopic pitch drift, but intensely sensitive to pitch tracker octave doubling/halving errors.

### 2.5 spread1 and spread2
- **Reference:** Little et al. (2009), *IEEE TBME*.
- **Mathematical Formulation:**
  - Built from the semitone pitch residual distribution $P(r)$.
  - `spread1`: Log-transformed dispersion of the tail of the semitone variation distribution:
    $$\text{spread1} = \ln \left( Q_{0.95}(r) - Q_{0.05}(r) \right)$$
    Because $Q_{0.95} - Q_{0.05} < 1.0\text{ semitones}$ in sustained vowels, its natural logarithm is naturally negative (mean $-5.68$ in UCI).
  - `spread2`: Second dispersion/spectral spread parameter capturing the asymmetry and second-order moment of the distribution of variations around median semitone pitch.

---

## 3. Library & Implementation Landscape

| Library / Tool | RPDE Support | DFA Support | D2 Support | PPE Support | spread1 / spread2 | License | Production Viability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Praat / Parselmouth** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None | GPL | Speech acoustics only; no non-linear dynamics algorithms. |
| **`antropy` (v0.2.3, installed)** | ❌ None | ✅ `detrended_fluctuation` | ❌ None | ❌ None | ❌ None | BSD-3 | High-performance, Numba/C-accelerated, verified DFA. |
| **`nolds` (not installed)** | ❌ None | ✅ `dfa` | ✅ `corr_dim` | ❌ None | ❌ None | MIT | Pure Python/NumPy; D2 is very slow on audio ($O(N^2)$). |
| **`rpde` (PyPI package)** | ⚠️ Partial | ❌ None | ❌ None | ❌ None | ❌ None | MIT | Unmaintained wrapper; requires C compilation; non-standard API. |
| **Custom NumPy/SciPy** | ⚠️ Feasible | ⚠️ Feasible | ⚠️ Feasible | ⚠️ Feasible | ⚠️ Feasible | Internal | Requires re-implementing Little's exact MATLAB routines. |

---

## 4. Feature Compatibility & Feasibility Matrix

| Feature | Extractable from Audio? | Best Candidate Tool | Latency (1.5s audio) | Numerical Stability | Known Implementation Differences | Feasibility Verdict |
| :--- | :---: | :--- | :---: | :---: | :--- | :---: |
| **`RPDE`** | Yes | Custom NumPy | $150\text{--}350\text{ ms}$ | Medium | Radius $\epsilon$ selection ($0.12$ vs adaptive); delay $\tau$ calculation. | **Conditionally Feasible** |
| **`DFA`** | Yes | `antropy.detrended_fluctuation` | $30\text{--}80\text{ ms}$ | High | Scale box range $[n_{\min}, n_{\max}]$; sampling frequency. | **Feasible** |
| **`D2`** | Partial | `nolds.corr_dim` / Custom | $1000\text{--}4000\text{ ms}$ | Low | Noise floor destroys plateau; extremely slow without downsampling. | **Unsuitable for Real-Time** |
| **`PPE`** | Yes | Parselmouth pitch + NumPy | $50\text{--}150\text{ ms}$ | High | Semitone bin width $\delta$; trend filter window length. | **Conditionally Feasible** |
| **`spread1`** | Partial | Parselmouth pitch + NumPy | $50\text{--}150\text{ ms}$ | Medium | Exact percentile bounds ($95/5$ vs $90/10$); log base. | **Conditionally Feasible** |
| **`spread2`** | Partial | Parselmouth pitch + NumPy | $50\text{--}150\text{ ms}$ | Medium | Exact higher-order moment / spectral spread formulation. | **Conditionally Feasible** |

---

## 5. Audio Preprocessing Contract for Non-Linear Analysis

Non-linear algorithms require standardized signal conditioning prior to phase-space or semitone analysis:

```text
Raw Browser Microphone Audio (WebM / WAV)
          │
          ▼
1. Container Decoding & Validation
   - Unpack 16-bit PCM uncompressed WAV
   - Enforce duration: 1.0s ≤ duration ≤ 3.5s
          │
          ▼
2. Channel & Sampling Rate Standardization
   - Downmix stereo to mono: s(t) = mean(channels)
   - Resample to exactly 22,050 Hz (scipy.signal.resample_poly)
          │
          ▼
3. Acoustic Signal Conditioning
   - Remove DC offset: s(t) = s(t) - mean(s(t))
   - Bandpass filter: 60 Hz – 4,000 Hz (4th-order Butterworth zero-phase)
   - Peak normalize: s(t) = s(t) / max(|s(t)|)
          │
          ▼
4. Phonation & Voicing Validation
   - Ensure pitch periodicity > 75% of frames
   - Enforce fundamental frequency range: 80 Hz ≤ F0 ≤ 450 Hz
          │
          ▼
5. Non-Linear Dynamical Feature Extraction
   ├── DFA: antropy.detrended_fluctuation(s, nvals=...)
   ├── RPDE: phase-space embedding (m=4, tau=first_zero(ACF), eps=0.12)
   ├── PPE: semitone conversion -> detrend -> Shannon entropy
   └── spread1 / spread2: quantile dispersion of semitone residuals
```

---

## 6. Proposed Validation Protocol (Overcoming the Paired Audio Limitation)

Because paired ground-truth audio does not exist for the UCI dataset, feature equivalence must be demonstrated through a multi-stage validation protocol:

### Step 1: Synthetic Benchmark Calibration
- **DFA:** Benchmark against synthetic Fractional Brownian Motion (fBm) with known theoretical Hurst exponents ($H = 0.50, 0.70, 0.90$). Verify that computed $\alpha = H \pm 0.02$.
- **RPDE:** Benchmark against pure sine waves (theoretical $RPDE \to 0$) and Mackey-Glass / Lorenz chaotic time series (theoretical $RPDE \approx 0.65\text{--}0.85$).
- **PPE / spread1:** Benchmark against synthetic vowel phonations with calibrated semitone pitch tremor ($F_0 = 150\text{ Hz} \pm 2\text{ semitones}$).

### Step 2: Distribution Alignment on Reference Voice Corpora
- Run the 22-feature pipeline on a public sustained phonation voice dataset (e.g. Saarbruecken Voice Database or Italian Parkinson's).
- Compute distribution divergence against the UCI dataset:
  - Verify that computed feature ranges, medians, and standard deviations lie within $1.5$ standard deviations of `parkinsons.data` reference statistics.
  - Evaluate Kolmogorov-Smirnov test statistics between distributions.

### Step 3: Model Invariance & Non-Saturation Test
- Feed reconstructed 22-feature vectors into the frozen champion model (`champion_pipeline.joblib`).
- Verify that predicted probabilities span the legitimate range $[0.05, 0.95]$ without degenerate clustering at $0.0$ or $1.0$.

---

## 7. Risk Register

| Risk ID | Risk Description | Severity | Likelihood | Impact on Project | Mitigation Strategy |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **R1** | **Feature Distribution Shift**<br>Custom Python implementations produce values shifted from Little's MATLAB/C routines. | **CRITICAL** | **HIGH** | Champion model decision trees make arbitrary or invalid predictions. | Calibrate against synthetic signals and enforce strict feature normalization / quantile matching. |
| **R2** | **Audio Compression Distortion**<br>Browser MediaRecorder uses lossy WebM/Opus at $48\text{ kHz}$, discarding chaotic micro-perturbations. | **HIGH** | **HIGH** | Non-linear dynamics are smoothed out by compression artifacts. | Enforce uncompressed 16-bit WAV recording in the frontend where browser support allows. |
| **R3** | **D2 Latency & Complexity**<br>Grassberger-Procaccia correlation dimension requires seconds to compute in Python. | **HIGH** | **HIGH** | Request timeouts and high CPU utilization in FastAPI. | Exclude D2 or approximate using fast $k$-d tree nearest-neighbor downsampling. |
| **R4** | **No Ground-Truth Audio**<br>Cannot directly verify feature accuracy against original UCI subjects. | **CRITICAL** | **CONFIRMED** | Mathematical impossibility of bitwise verification. | Document limitation transparently; evaluate generalization on external voice datasets. |

---

## 8. Final Recommendation: CONDITIONAL GO

> [!IMPORTANT]
> ### Recommendation: **CONDITIONAL GO** (for Stage 6D Algorithmic Formulation & Synthetic Calibration)
> - **Direct Production Deployment:** **NO-GO.** Unverified non-linear implementations must never be directly connected to the frozen champion model.
> - **Calibration & Formulation Feasibility:** **CONDITIONAL GO.** It is technically feasible to implement `DFA` (via `antropy`), `PPE`, `spread1`, and `spread2` (via Praat semitone tracking + NumPy entropy) and benchmark them under strict synthetic controls.
> - **D2 Exclusion:** `D2` should either be approximated via fast algorithm or flagged as computationally unsuitable for web screening.

---

## 9. Proposed Next Stage: Stage 6D Acceptance Criteria

**Stage 6D Title:** *Algorithmic Formulation & Calibration of Core Non-Linear Features (`DFA`, `PPE`, `spread1`, `spread2`)*

### Acceptance Criteria for Stage 6D:
1. **Algorithmic Correctness:** Formulate and verify `DFA` (using `antropy`), `PPE`, `spread1`, and `spread2` in Python.
2. **Synthetic Benchmarking:** Pass unit tests verifying theoretical values on synthetic fractal and chaotic signals within $\pm 5\%$ tolerance.
3. **Range Compatibility:** Verify on synthetic sustained vowel recordings that extracted values fall within the legitimate support bounds of the UCI dataset:
   - `DFA` $\in [0.55, 0.85]$
   - `PPE` $\in [0.04, 0.55]$
   - `spread1` $\in [-8.0, -2.0]$
   - `spread2` $\in [0.00, 0.50]$
4. **Latency Ceiling:** End-to-end extraction latency for a 2-second audio clip must remain under **$500\text{ ms}$**.
5. **No Model Modification:** Champion model artifact (`champion_pipeline.joblib`) remains $100\%$ untouched.

---
*Stage 6C Non-Linear Feature Reconstruction Feasibility Audit completed.*
