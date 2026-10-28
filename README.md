# PerceptionGuard: Detecting Image Corruption Failures and Calibrated Abstention
### A Lightweight Research Prototype for Safety-Critical Vision in Robotics and Medical Imaging

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EUJ_uafoTOQodeAkZlOGt3EfQaVeODIL?usp=sharing)
 or copy paste the following link:
 https://colab.research.google.com/drive/1EUJ_uafoTOQodeAkZlOGt3EfQaVeODIL?usp=sharing

---

## 🎯 Research Overview

### Research Question
> **When does a small image classifier become unreliable under image corruption, and can a lightweight reliability monitor detect likely-wrong predictions and abstain better than confidence-threshold baselines?**

### Motivation
In safety-critical applications such as **autonomous robotics** (sensor lens blur, adverse lighting, weather noise) and **medical screening** (scanner variations, microscope blur, staining inconsistencies), a deep learning vision system must know when its predictions are untrustworthy. Standard classifiers suffer from severe **overconfidence under distribution shift**, making confident errors that could cause robotic collisions or patient misdiagnoses.

---

## 🔬 Core Methodology

1. **Base Classifier**: `SmallVGG` CNN with in-model input normalization (`x - mean / std`), trained purely on **clean** images using mixed precision (`torch.cuda.amp.autocast`) for ~12 epochs.
2. **GPU Synthetic Corruptions (5 Severities Each)**:
   - `gaussian_noise`: Zero-mean Gaussian perturbation.
   - `gaussian_blur`: Depthwise separable 2D Gaussian convolution.
   - `brightness`: Additive illumination shift.
   - `contrast`: Spatial contrast scaling around image mean (**held out as unseen corruption**).
3. **10-D Reliability Features**:
   - *Output Uncertainty*: Max softmax probability, top-2 probability margin, predictive entropy, max logit, energy score.
   - *Internal Representation*: Euclidean distance to nearest clean class centroid in penultimate 64-D embedding space.
   - *Input Image Quality*: Mean brightness, contrast standard deviation, log Laplacian variance (sharpness/blur), mean neighbor-pixel gradient (graininess/noise).
4. **Compared Systems**:
   - `(a) Raw Max-Softmax Baseline`: Confidence thresholding ($\tau = 0.90$).
   - `(b) Temperature Scaling Baseline`: $T^*$ optimized on clean validation NLL.
   - `(c) Output-Only Monitor`: `HistGradientBoostingClassifier` on features 1–5.
   - `(d) PerceptionGuard All-Features Monitor`: `HistGradientBoostingClassifier` on all 10 features.

---

## 📊 Comprehensive Benchmark Suite

The notebook runs an end-to-end experimental evaluation:
- **Accuracy Degradation vs Severity**: Tabular breakdown and curve plots across corruptions and severities 1 to 5.
- **Expected Calibration Error (ECE) & Reliability Diagrams**: Demonstrating calibration collapse on corrupted data.
- **Matched Coverage Comparison**: Strict evaluation at identical coverage levels (e.g. 80% and 90%), reporting Selective Accuracy, Errors Caught (Recall of wrong predictions), AUROC, and AURC across:
  - Clean Test
  - Seen Corruptions Pooled
  - Unseen Contrast Shift
  - Severity 5 Extreme Shift
- **Risk-Coverage Tradeoff Curves**: Area Under Risk-Coverage (AURC).
- **Leave-One-Corruption-Out (LOCO) Generalization**: Transferability to unseen corruption types.
- **Bootstrap 95% Confidence Intervals**: Non-parametric significance testing ($B=1000$) for $\Delta \text{AUROC}$.
- **Threshold-Promise Test**: Testing whether a validation risk promise ($\le 5\%$ error) holds or breaks under severe distribution shift.
- **Cost-Sensitive Operational Analysis**: Comparing total loss under $C_{\text{wrong}}:C_{\text{abstain}}$ ratios of $5:1$, $10:1$, and $20:1$.
- **Multi-Seed Stability**: Seed spread reporting across random seeds.

---

## 💻 Interactive Gradio Application

The notebook terminates with an integrated two-tab Gradio app:
- **Tab 1: Live PerceptionGuard Inspector**: Choose/upload an image, adjust corruption type & severity, view original vs corrupted images, class probabilities, abstention decisions, and the 10 extracted reliability features.
- **Tab 2: Benchmark Reports & Figures**: Interactive gallery of generated charts and a one-click download button for `perception_guard_results.zip`.

---

## 🚀 How to Run in Google Colab (Free Tier GPU)

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select [`PerceptionGuard.ipynb`](file:///C:/Users/Vinay/.gemini/antigravity/scratch/perception_guard/PerceptionGuard.ipynb).
3. Under the **Runtime** menu, select **Change runtime type** -> choose **T4 GPU**.
4. Run all cells (**Runtime** -> **Run all**). The entire notebook executes in approximately 3 to 5 minutes!
5. To toggle between datasets, simply change `DATASET = "cifar10"` to `DATASET = "bloodmnist"` in Section 2.

---

## ⚖️ License & Ethical Disclaimers
- **CIFAR-10**: Academic / MIT license.
- **BloodMNIST**: MedMNIST v2, licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Explicit Non-Clinical Disclaimer**: BloodMNIST and this prototype are designed purely for academic research and benchmarking. **This is NOT clinical software and must NOT be used for real-world patient diagnosis or safety-critical deployment without formal regulatory compliance.**
