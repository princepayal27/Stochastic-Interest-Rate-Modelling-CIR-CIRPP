# Stochastic Interest Rate Modelling — CIR & CIR++

> **Implementing, Calibrating, and Extending the Cox-Ingersoll-Ross Model on Real Yield Curve Data**  
> Finance Club, IIT Roorkee — Open Projects 2026

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Notebook](https://img.shields.io/badge/Notebook-Google%20Colab-orange)](https://colab.research.google.com/)
[![R²](https://img.shields.io/badge/Official%20R%C2%B2%20(3M--2Y)-0.9237-brightgreen)]()
[![Feller](https://img.shields.io/badge/Feller%20Condition-Satisfied-brightgreen)]()

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Mathematical Background](#mathematical-background)
- [Results](#results)
- [Limitations](#limitations)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Technical Skills Demonstrated](#technical-skills-demonstrated)
- [Project Descriptions](#project-descriptions)

---

## Project Overview

This project builds a complete quantitative finance pipeline for **stochastic interest rate modelling** using the **Cox-Ingersoll-Ross (CIR)** framework. The model is calibrated on historical zero-coupon yield data across 9 maturity tenors (3M–30Y), then evaluated on its ability to reconstruct the full yield curve using **only the 3-Month rate as a live input**.

The pipeline implements a **CIR++ extension** (Brigo-Mercurio framework) via a deterministic maturity-specific shift φ(τ), bootstrapped from the final training observation to anchor the model to the observed term structure before projecting forward.

**Key constraint:** At prediction time, only the 3M yield is observable. No re-fitting or re-optimisation is permitted on test data. The official evaluation metric is **out-of-sample R² on the 3M–2Y maturity range**.

---

## Dataset

| Property | Detail |
|---|---|
| Training Period | 2016-05-19 → 2024-04-26 (1,976 daily records) |
| Test Period | 2024-04-29 → 2026-04-29 (495 daily records) |
| Maturities | 3M, 6M, 9M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y |
| Data Format | Zero-coupon yields, decimal form |

Preprocessing: whitespace stripping, date parsing, 'ffill/bfill' for gaps, automatic decimal scale verification.

---

## Mathematical Background

### CIR Stochastic Differential Equation

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$$

| Parameter | Symbol | Interpretation |
|---|---|---|
| Mean reversion speed | $\kappa$ | How quickly $r_t$ returns to $\theta$ |
| Long-run mean | $\theta$ | Equilibrium rate level |
| Volatility | $\sigma$ | Magnitude of stochastic fluctuations |

The **Feller condition** $2\kappa\theta \geq \sigma^2$ ensures rates remain strictly positive.

### Zero-Coupon Yield (Closed Form)

$$y(t, \tau) = \frac{B(\tau)\,r_t - \ln A(\tau)}{\tau}, \qquad \gamma = \sqrt{\kappa^2 + 2\sigma^2}$$

$$B(\tau) = \frac{2(e^{\gamma\tau} - 1)}{2\gamma + (\kappa + \gamma)(e^{\gamma\tau} - 1)}, \qquad A(\tau) = \left(\frac{2\gamma\,e^{(\kappa+\gamma)\tau/2}}{2\gamma + (\kappa + \gamma)(e^{\gamma\tau} - 1)}\right)^{\!\frac{2\kappa\theta}{\sigma^2}}$$

### CIR++ Extension (Brigo-Mercurio)

$$Y_{\text{CIR++}}(r_t, \tau) = Y_{\text{CIR}}(r_t, \tau) + \varphi(\tau)$$

$$\varphi(\tau) = y_{\text{market}}(\tau)\big|_{t=T_{\text{train}}} - y_{\text{CIR}}(r_{T_{\text{train}}}, \tau;\,\kappa,\theta,\sigma)$$

φ(τ) is computed **once** from the final training observation and remains fixed — no test data leakage.

### Calibration Objective

Cross-sectional least squares over all training days × all 9 maturities simultaneously:

$$\min_{\kappa,\,\theta,\,\sigma} \sum_{t=1}^{T} \sum_{i=1}^{9} \left[y_{\text{market}}(t, \tau_i) - y_{\text{CIR}}(r_t, \tau_i;\,\kappa,\theta,\sigma)\right]^2$$

Solved via L-BFGS-B with fully vectorised NumPy broadcasting `(N,1) × (1,9) → (N,9)`.

---

## Results

### Calibrated Parameters

| Parameter | Value |
|---|---|
| $\kappa$ (Mean Reversion Speed) | 0.165979 |
| $\theta$ (Long-Run Mean) | 0.024408 (2.44%) |
| $\sigma$ (Volatility) | 0.000618 |
| Feller: $2\kappa\theta$ vs $\sigma^2$ | 0.008102 ≥ 0.000000 ✅ |

### PCA Factor Analysis

| Factor | Variance Explained |
|---|---|
| PC1 — Level | **96.34%** |
| PC2 — Slope | 3.01% |
| PC3 — Curvature | 0.53% |

PC1 dominance empirically validates the one-factor CIR assumption for the short end.

### Out-of-Sample Performance

| Metric | Base CIR | CIR++ |
|---|---|---|
| Global RMSE | 58.60 bps | 42.27 bps |
| Global R² | 0.0548 | 0.5081 |
| **Official R² (3M–2Y)** | **0.9237 ✅** | **0.9119 ✅** |

Both models exceed the **0.85 threshold** on the official metric. CIR++ improves global RMSE by 27%.

### Per-Maturity Out-of-Sample R²

| Maturity | Base CIR | CIR++ |
|---|---|---|
| 3M | +0.9994 | +0.9975 |
| 6M | +0.9944 | +0.9947 |
| 9M | +0.9675 | +0.9702 |
| 1Y | +0.9102 | +0.9120 |
| 2Y | +0.3897 | +0.2484 |
| 5Y | −3.2845 | −7.0601 |
| 10Y | −14.5262 | −10.1847 |
| 20Y | −17.0844 | −4.4525 |
| 30Y | −13.8756 | −2.2929 |

Negative R² at 5Y–30Y reflects the one-factor structural constraint, not a calibration failure.

---

## Limitations

1. **One-Factor Constraint** — All maturities are driven by a single state variable $r_t$. The long end responds to inflation expectations and term premium that one factor cannot encode. PC2+PC3 (3.54% of variance) are the components the model cannot reach.

2. **Static φ(τ)** — The CIR++ shift is computed once and becomes stale as the market drifts from the training anchor, visibly degrading at 5Y–30Y over time.

3. **Low Diffusion** — Calibrated σ = 0.000618 is very small, making the Feller condition trivially satisfied but underestimating realised short-rate volatility.

4. **Shape Constraints** — CIR cannot reproduce strongly inverted or double-humped curves, both of which appear in the test period.

---

## Repository Structure

```
cir-yield-curve/
│
├── CIR_Yield_Curve_Project.ipynb   # Main notebook (run top-to-bottom)
│
├── data/
│   ├── train_data.csv              # Historical yield data (2016–2024)
│   ├── test_data.csv               # Out-of-sample yield data (2024–2026)
│   └── test_data_3M.csv            # 3M yields only (test period input)
│
├── outputs/
│   ├── yield_movements_over_time.png
│   ├── yield_correlation_matrix.png
│   ├── average_yield_curve_shape.png
│   ├── pca_factor_loadings.png
│   ├── cir_short_rate_theory_analysis.png
│   ├── cir_yield_curve_reconstruction.png
│   └── cir_plus_plus_evaluation.png
│
├── requirements.txt
└── README.md
```

---

## Requirements

```txt
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

> All development was done in **Google Colab**. All packages above are pre-installed in Colab.

---

## Technical Skills Demonstrated

| Category | Skills |
|---|---|
| **Quantitative Finance** | Stochastic calculus, affine term structure models, zero-coupon bond pricing, Feller condition, mean-reversion dynamics |
| **Mathematical Modelling** | Closed-form CIR pricing, PCA factor decomposition, cross-sectional least squares, Brigo-Mercurio φ(τ) bootstrapping |
| **Statistical Analysis** | Time-series preprocessing, out-of-sample backtesting, R²/RMSE/MAE evaluation, distributional analysis |
| **Python / Scientific Computing** | NumPy vectorisation, SciPy L-BFGS-B optimisation, Pandas time-series, Matplotlib/Seaborn, scikit-learn |
| **Critical Thinking** | Structural vs. calibration failure diagnosis, honest limitation analysis, theory-to-empirics connection |

---

## Project Descriptions

### Resume (50 words)
Implemented and calibrated the Cox-Ingersoll-Ross (CIR) stochastic interest rate model on 8 years of zero-coupon yield data across 9 maturities. Extended to CIR++ using Brigo-Mercurio deterministic shift bootstrapping. Achieved out-of-sample R²=0.9237 on the 3M–2Y range using only the 3-Month rate as a live input.

---

### LinkedIn (100 words)
Built a full quantitative finance pipeline implementing the Cox-Ingersoll-Ross (CIR) stochastic interest rate model from scratch — calibrated on 1,976 days of multi-maturity yield data using cross-sectional least squares, then extended to CIR++ via the Brigo-Mercurio deterministic shift framework.

The model reconstructs the full yield curve (6M–30Y) using only the live 3-Month rate as input, achieving an out-of-sample R²=0.9237 on the 3M–2Y range — well above the 0.85 threshold. PCA confirms 96.34% of yield variance is driven by a single level factor, empirically validating the one-factor CIR assumption.

Project completed as part of Finance Club IIT Roorkee Open Projects 2026.

---

### Portfolio (150–200 words)
This project is an end-to-end implementation of the **Cox-Ingersoll-Ross (CIR)** stochastic short-rate model applied to real historical zero-coupon yield data spanning 2016–2026. The model captures interest rate dynamics through a mean-reverting square-root diffusion process and derives closed-form zero-coupon bond prices and yields using the CIR affine term structure framework.

The pipeline begins with robust data engineering (1,976 training days, 9 maturities), proceeds through empirical factor analysis (PCA confirms 96.34% level dominance), calibrates parameters via vectorised cross-sectional least squares, and culminates in strict out-of-sample evaluation where **only the 3-Month yield is observable** at prediction time.

The model is extended to **CIR++** using the Brigo-Mercurio deterministic shift φ(τ), bootstrapped from the final training observation to eliminate systematic level bias. The official evaluation metric — R² on the 3M–2Y maturity range — reaches **0.9237**, exceeding the 0.85 target.

Limitations are analysed rigorously: the one-factor structure explains why long-tenor R² turns negative, and the static φ(τ) degrades as the market drifts from the training anchor — motivating a two-factor extension as the natural next step.
