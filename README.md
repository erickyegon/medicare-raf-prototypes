# Medicare Clinical Performance Analytics Pipeline

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CMS%20HCC-v28%20Model-1B4F72" />
  <img src="https://img.shields.io/badge/Causal%20Inference-DiD%20%7C%20PSM-2E86C1" />
  <img src="https://img.shields.io/badge/ML-XGBoost-orange?logo=xgboost" />
  <img src="https://img.shields.io/badge/Domain-Value--Based%20Care-green" />
  <img src="https://img.shields.io/badge/Cohort-50%2C000%20Beneficiaries-lightgrey" />
</p>

An end-to-end, production-style analytics pipeline for **Medicare Advantage and ACO clinical performance measurement** — covering HCC/RAF risk adjustment, clinical risk stratification, causal attribution of intervention impact, and shared savings projection under MSSP/ACO REACH benchmarking logic.

---

## The Problem This Solves

Every ACO enablement platform and value-based care organisation faces the same core quantitative challenge:

> *"Did our clinical programme reduce total cost of care — and by exactly how much does that translate to shared savings?"*

Answering that question rigorously requires three distinct analytical capabilities working together:

1. **Accurate risk adjustment** — so you're comparing like with like across beneficiary populations
2. **Predictive risk stratification** — so you can identify and prioritise high-risk patients before costs accumulate
3. **Credible causal attribution** — so you can distinguish genuine intervention effects from secular trends, selection bias, and regression to the mean

This pipeline implements all three, end-to-end, in a single reproducible codebase.

---

## Results at a Glance

| Stage | Metric | Result |
|-------|--------|--------|
| **RAF Scoring** | Mean RAF score (N=50,000) | 2.131 |
| **RAF Scoring** | % beneficiaries RAF > 2.0 (high-cost) | 40.4% |
| **Risk Model** | Tier classification accuracy | **91.5%** |
| **Risk Model** | Annual cost prediction MAE | **$2,181** |
| **Risk Model** | Annual cost R² | 0.450 |
| **Causal Attribution (DiD)** | ATT — cost per member | **−$391** (p < 0.0001) |
| **Causal Attribution (PSM)** | ATT — sensitivity check | −$392 *(convergent)* |
| **Parallel trends** | Pre-period balance test | p = 0.582 ✓ |
| **Shared Savings** | Gross savings (25k attributed lives) | **$9.77M** |
| **Shared Savings** | Earned at 50% MSSP sharing rate | **$4.89M** |
| **Pipeline runtime** | End-to-end (50k beneficiaries) | **29 seconds** |

> DiD and PSM estimates converge within $1 — the finding is robust to estimation method.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEDICARE ANALYTICS PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CMS Claims (ICD-10)                                             │
│       │                                                           │
│       ▼                                                           │
│  ┌─────────────────┐                                             │
│  │  HCC Mapping    │  ICD-10 → CMS HCC v28 categories           │
│  │  (hcc_mapper)   │  50+ codes · demographic coefficients       │
│  └────────┬────────┘  disease interactions (CHF×AFib, etc.)     │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────┐                                             │
│  │  RAF Calculator │  RAF = demo + Σ HCC coefficients           │
│  │ (raf_calculator)│       + interaction terms                   │
│  └────────┬────────┘  PMPM cost = RAF × $9,800 baseline         │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────┐                                             │
│  │  Risk           │  XGBoost classifier → risk tier            │
│  │  Stratification │  XGBoost regressor  → annual cost          │
│  └────────┬────────┘  91.5% accuracy · MAE $2,181               │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────┐                                             │
│  │  Causal         │  DiD (TWFE, clustered SE)  → primary ATT   │
│  │  Attribution    │  PSM (1:1 NN, caliper 0.05) → sensitivity  │
│  └────────┬────────┘  Parallel trends validated · p = 0.582     │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────┐                                             │
│  │  Shared Savings │  MSSP benchmarking · MSR threshold         │
│  │  Projection     │  $9.77M gross · $4.89M earned              │
│  └─────────────────┘                                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
medicare-raf-prototypes/
│
├── src/
│   ├── hcc_mapper.py            # ICD-10 → HCC v28 mapping + coefficients
│   ├── raf_calculator.py        # RAF scoring + PMPM cost estimation
│   ├── data_generator.py        # Synthetic 50k CMS-style beneficiary cohort
│   ├── risk_stratification.py   # XGBoost risk tier + cost prediction pipeline
│   └── causal_attribution.py    # DiD, PSM (vectorized), shared savings
│
├── run_pipeline.py              # End-to-end orchestration (all 4 stages, 29s)
│
├── notebooks/
│   ├── 01_hcc_raf_deep_dive.ipynb          # HCC methodology + RAF walkthrough
│   ├── 02_risk_stratification.ipynb        # Model training, SHAP, outreach lists
│   └── 03_shared_savings_attribution.ipynb # DiD/PSM results + HTE by risk tier
│
├── reports/figures/             # Pipeline output visualisations (auto-generated)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/erickyegon/medicare-raf-prototypes.git
cd medicare-raf-prototypes
pip install -r requirements.txt
python run_pipeline.py
```

Full pipeline completes in **~30 seconds**. Figures are written to `reports/figures/`.

To explore interactively:
```bash
jupyter lab notebooks/
```

---

## Methods

### 1 · HCC/RAF Risk Adjustment (CMS v28)

The CMS Hierarchical Condition Category model is the basis of Medicare Advantage risk adjustment. Each beneficiary's RAF score determines expected annual cost relative to the average (RAF = 1.0 equals ~$9,800/year).

```
RAF = demographic_coefficient
    + Σ HCC_coefficients (for all triggered HCC categories)
    + Σ interaction_terms (e.g. CHF×AFib: +0.175)
```

**Implementation details:**
- Age-sex demographic coefficients for community non-dual aged segment (CY2024 v28)
- 50+ ICD-10 code mappings covering highest-prevalence conditions: CHF, T2DM, CKD stages 1–6/ESRD, AFib, COPD, cancer (metastatic through site-specific), neurological, vascular
- Six disease interaction terms: CHF×AFib, CHF×Diabetes, ESRD×CHF, ESRD×Diabetes, COPD×Diabetes, Cancer×CHF
- PMPM cost estimation: `cost = RAF × $9,800` (2023 Medicare FFS baseline)

**Example — complex patient (76F, CHF + AFib + T2DM + CKD Stage 4):**

| Component | Value |
|-----------|-------|
| Demographic (76F) | 0.453 |
| CHF (HCC 85) | +0.323 |
| AFib (HCC 96) | +0.421 |
| T2DM w/ complications (HCC 18) | +0.302 |
| CKD Stage 4 (HCC 137) | +0.138 |
| CHF × AFib interaction | +0.175 |
| **Total RAF** | **1.812** |
| **Estimated annual cost** | **$17,758** |

### 2 · Clinical Risk Stratification (XGBoost)

A two-model ensemble for population health management:

- **Classifier** — 3-tier risk stratification (high / moderate / low) for care management prioritisation
- **Regressor** — continuous annual cost prediction for budget forecasting and benchmark setting

**Feature engineering:**
- 22 HCC binary flags (one-hot encoded per condition category)
- Derived features: HCC count, multi-disease indicators (has_cancer, has_chf, has_ckd, has_copd, has_afib)
- Disease interaction features: CHF×AFib, CHF×Diabetes, CKD×Diabetes
- Demographics: age (scaled + quadratic), sex, dual eligibility

**Top predictors by XGBoost gain:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | hcc_count | 0.434 |
| 2 | has_ckd | 0.082 |
| 3 | chf_afib | 0.054 |
| 4 | hcc_11 (colorectal cancer) | 0.030 |
| 5 | hcc_136 (CKD Stage 5) | 0.025 |

### 3 · Causal Attribution (DiD + PSM)

**Primary estimator — Two-Way Fixed Effects DiD:**

```
Cost_it = α + β₁·Post_t + β₂·Treat_i + β₃·(Post_t × Treat_i) + γ·X_i + ε_it
```

β₃ is the Average Treatment effect on the Treated (ATT) — the causal cost reduction attributable to the intervention.

| Check | Result | Interpretation |
|-------|--------|----------------|
| ATT | −$391.31/member | Intervention reduced cost by $391/member |
| Standard error | $24.10 | Clustered at beneficiary level |
| 95% CI | [−$438.59, −$344.04] | Excludes zero |
| p-value | < 0.0001 | Highly significant |
| Parallel trends (pre-period) | p = 0.582 | ✓ Key DiD assumption holds |

**Sensitivity — Propensity Score Matching (1:1 NN, caliper = 0.05):**

PS estimated from pre-period demographics and utilisation via logistic regression. Vectorized KD-tree matching across 25,000 beneficiary pairs.

| Check | Result | Interpretation |
|-------|--------|----------------|
| ATT (PSM) | −$392.43/member | Convergent with DiD (Δ = $1.12) |
| Matched pairs | 24,979 | Full treatment arm matched |
| Post-match SMD (age) | 0.000 | ✓ Perfect balance (target < 0.10) |

> DiD and PSM estimates converging within $1.12 on a 50,000-person cohort is a strong indicator of a robust, unbiased causal estimate.

**Heterogeneous treatment effects by risk tier:**

The intervention generates larger absolute savings among high-risk beneficiaries — consistent with the clinical logic that complex patients benefit most from care coordination.

### 4 · Shared Savings Projection (MSSP Framework)

```
Gross savings   = (Benchmark PMPM − Actual PMPM) × N attributed lives
Savings rate    = Gross savings / Total benchmark expenditure
Earned savings  = Gross savings × sharing rate  [if savings rate > MSR threshold]
```

| Parameter | Value |
|-----------|-------|
| Benchmark PMPM | $9,800.00 |
| Actual PMPM (post-intervention) | $9,408.69 |
| Attributed lives | 24,979 |
| Gross savings | $9,774,532 |
| Savings rate | 4.0% |
| MSR threshold (2%) | ✓ Exceeded |
| **Shared savings earned (50% rate)** | **$4,887,266** |

---

## Production Considerations

This pipeline is built on synthetic data with a representative subset of CMS mappings. Deploying against real Medicare claims would require:

| Area | Production requirement |
|------|----------------------|
| **HCC completeness** | Full CMS v28 map: ~9,000 ICD-10 codes across 86 HCC categories |
| **Data sources** | Part A (inpatient), Part B (outpatient/physician), Part D (pharmacy) claims |
| **Additional RAF inputs** | ESRD indicators, new enrollee flags, low-income subsidy status, encounter data for MA |
| **Causal methods** | RDD or synthetic control for settings without clean pre/post periods |
| **Infrastructure** | HIPAA-compliant data pipeline, BAA, de-identification, audit logging |
| **Model governance** | Drift monitoring, scheduled retraining, explainability layer for clinical users |
| **Scale** | Distributed computation (Spark / Dask) for multi-million beneficiary populations |

---

## References

- CMS. (2024). *Medicare Advantage Risk Adjustment: CY2024 HCC Model (v28)*. Centers for Medicare & Medicaid Services.
- MedPAC. (2023). *Report to the Congress: Medicare Payment Policy*. Medicare Payment Advisory Commission.
- Angrist, J. & Pischke, J-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Austin, P.C. (2011). An introduction to propensity score methods for reducing confounding. *Multivariate Behavioral Research*, 46(3), 399–424.
- Imai, K. & Ratkovic, M. (2014). Covariate balancing propensity score. *Journal of the Royal Statistical Society*, 76(1), 243–263.

---

## About

**Erick K. Yegon, PhD** — Epidemiologist & Data Science Leader

PhD Epidemiology · 17+ years in quantitative research, causal inference, and clinical performance analytics · 30+ peer-reviewed publications including *The Lancet* · h-index 10

Particular interest in the application of rigorous causal inference methods to healthcare financial performance measurement — bridging the gap between population health science and value-based care economics.

[LinkedIn](https://linkedin.com/in/erickyegon) · [GitHub](https://github.com/erickyegon) · [ORCID](https://orcid.org/0000-0002-7055-4848) · [Google Scholar](https://scholar.google.com)

---

*Data: 100% synthetic. No PHI or real beneficiary data is used anywhere in this repository.*
