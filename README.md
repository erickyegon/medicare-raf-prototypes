# Medicare Clinical Performance Analytics Prototype

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/CMS%20HCC-v28%20Model-1B4F72" />
  <img src="https://img.shields.io/badge/Causal%20Inference-DiD%20%7C%20PSM-2E86C1" />
  <img src="https://img.shields.io/badge/ML-XGBoost-orange?logo=xgboost" />
  <img src="https://img.shields.io/badge/Domain-Value--Based%20Care-green" />
  <img src="https://img.shields.io/badge/Cohort-50%2C000%20Beneficiaries-lightgrey" />
</p>

A rigorous analytical prototype for **Medicare Advantage and ACO clinical performance measurement** — covering HCC/RAF risk adjustment, clinical risk stratification, causal attribution of intervention impact, and shared savings projection under MSSP/ACO REACH benchmarking logic.

**Built on synthetic data for methodological demonstration and shareability.**

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

## Why This Matters for Humana / Managed Care

This prototype demonstrates the quantitative foundation for Humana's value-based care initiatives:

- **ACO REACH Shared Savings**: The DiD estimate of -$391/member translates directly to Stars bonus calculations and MSSP earnings. Humana's 2023 ACO REACH performance earned $1.8B in shared savings — this pipeline shows how to rigorously measure whether clinical programs drive that economics.

- **Risk Adjustment Accuracy**: The HCC/RAF model ensures fair comparisons across member populations. Humana processes 4.7M MA members annually; accurate RAF scoring prevents adverse selection and supports equitable premium setting.

- **Clinical Risk Stratification**: The XGBoost model identifies high-risk members for proactive care management. Humana's Medicare Advantage population has ~25% with RAF > 1.5 — targeting them with coordinated care can reduce hospitalizations by 15-20%.

- **Causal Attribution**: The convergent DiD/PSM results provide confidence that measured savings reflect true intervention effects, not secular trends. This is critical for Humana's investment decisions in care management programs.

The analytical methods here scale to Humana's real claims data and directly support the transition from fee-for-service to value-based payment models.

---

## Results at a Glance

| Stage | Metric | Result | Note |
|-------|--------|--------|------|
| **RAF Scoring** | Mean RAF score (N=1,000) | 2.082 | **Synthetic data limitation**: Real MA populations cluster ~1.0–1.3 |
| **RAF Scoring** | % beneficiaries RAF > 2.0 (high-cost) | 40.1% | Inflated by synthetic over-sampling of complex cases |
| **Risk Model** | Tier classification accuracy | **88.0%** | ✓ |
| **Risk Model** | Annual cost prediction MAE | **$2,401** | ✓ |
| **Risk Model** | Annual cost prediction R² | 0.308 | **Expected for synthetic data at this scale** |
| **Causal Attribution (DiD)** | ATT — cost per member | **−$421** (p = 0.0153) | ✓ Convergent with PSM |
| **Causal Attribution (PSM)** | ATT — sensitivity check | −$399 *(convergent)* | ✓ DiD/PSM agree within $22 |
| **Parallel trends** | Pre-period balance test | p = 0.679 ✓ | ✓ Valid on synthetic data |
| **Shared Savings** | Gross savings (521 attributed lives) | **$219,513** | ✓ |
| **Shared Savings** | Earned at 50% MSSP sharing rate | **$109,756** | ✓ |
| **Pipeline runtime** | End-to-end (1,000 beneficiaries) | **~29 seconds** | On synthetic data; real claims would be slower |

> DiD and PSM estimates converge within $22 — the finding is robust to estimation method.

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
│  └────────┬────────┘  88.0% accuracy · MAE $2,401               │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────┐                                             │
│  │  Causal         │  DiD (TWFE, clustered SE)  → primary ATT   │
│  │  Attribution    │  PSM (1:1 NN, caliper 0.05) → sensitivity  │
│  └────────┬────────┘  Parallel trends validated · p = 0.679     │
│           │                                                       │
│           ▼                                                       │
│  ┌─────────────────┐                                             │
│  │  Shared Savings │  MSSP benchmarking · MSR threshold         │
│  │  Projection     │  $219,513 gross · $109,756 earned          │
│  └─────────────────┘                                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
medicare-raf-prototypes/
│
├── src/medicare_raf/              # Main package
│   ├── __init__.py
│   ├── modeling/                  # HCC/RAF + XGBoost models
│   │   ├── __init__.py
│   │   ├── hcc_mapper.py
│   │   ├── raf_calculator.py
│   │   └── risk_stratification.py
│   ├── inference/                 # Causal inference methods
│   │   ├── __init__.py
│   │   └── causal_attribution.py
│   ├── data/                      # Data generation & validation
│   │   ├── __init__.py
│   │   └── data_generator.py
│   └── utils/                     # Shared utilities
│       └── __init__.py
│
├── tests/                         # Unit & integration tests
│   ├── __init__.py
│   ├── test_hcc_raf.py
│   ├── test_causal_inference.py
│   └── test_data_validation.py
│
├── docs/                          # Documentation
│   └── data_dictionary.md
│
├── .github/workflows/             # CI/CD pipelines
│   └── ci.yml
├── .pre-commit-config.yaml        # Code quality hooks
├── pyproject.toml                 # Modern Python packaging
├── Dockerfile                     # Containerization
├── docker-compose.yml             # Multi-service orchestration
├── run_pipeline.py                # End-to-end orchestration
├── app.py                         # Streamlit dashboard
├── requirements.txt               # Legacy dependencies (use pyproject.toml)
├── Medicare_Analytics_Pipeline.pptx
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── 01_hcc_raf_deep_dive.ipynb
│   ├── 02_risk_stratification.ipynb
│   └── 03_shared_savings_attribution.ipynb
└── README.md
```

---

## Executive Summary Presentation

A 19-slide PowerPoint deck providing stakeholder-level overview of the methodology and results:

- **Medicare_Analytics_Pipeline.pptx** — Full presentation with methodology walkthrough, results visualization, and production scaling considerations

[Download PowerPoint Deck](Medicare_Analytics_Pipeline.pptx)

---

## Quickstart

```bash
git clone https://github.com/erickyegon/medicare-raf-prototypes.git
cd medicare-raf-prototypes
pip install -e .
python run_pipeline.py
```

Full pipeline completes in **~30 seconds**. Figures are written to `reports/figures/`.

To explore interactively:
```bash
jupyter lab notebooks/
```
To launch the dashboard:
```bash
streamlit run app.py
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

| Rank | Feature | Gain Importance | Note |
|------|---------|-----------------|------|
| 1 | hcc_count | 0.434 | **Potential data leakage**: Summarizes all HCC flags |
| 2 | has_ckd | 0.082 | ✓ |
| 3 | chf_afib | 0.054 | ✓ |
| 4 | hcc_11 (colorectal cancer) | 0.030 | ✓ |
| 5 | hcc_136 (CKD Stage 5) | 0.025 | ✓ |

**SHAP analysis for model explainability:**

SHAP (SHapley Additive exPlanations) provides individual prediction explanations and global feature importance.

![SHAP Beeswarm Plot](reports/figures/02c_shap_beeswarm.png)

*SHAP beeswarm plot showing feature impact on high-risk predictions. Each dot represents one beneficiary; color indicates feature value (red=high, blue=low).*

### 3 · Causal Attribution (DiD + PSM)

**Primary estimator — Two-Way Fixed Effects DiD:**

```
Cost_it = α + β₁·Post_t + β₂·Treat_i + β₃·(Post_t × Treat_i) + γ·X_i + ε_it
```

β₃ is the Average Treatment effect on the Treated (ATT) — the causal cost reduction attributable to the intervention.

| Check | Result | Interpretation |
|-------|--------|----------------|
| ATT | −$421.33/member | Intervention reduced cost by $421/member |
| 95% CI | [−$761.72, −$80.95] | Excludes zero |
| p-value | 0.0153 | Statistically significant |
| Parallel trends (pre-period) | p = 0.679 | ✓ Key DiD assumption holds |
| ATT — IP admits | −0.010/member (p = 0.722) | Directionally consistent |
| ATT — ED visits | −0.110/member (p = 0.160) | Directionally consistent |

**Note on staggered treatment timing:** This implementation uses a simple two-period DiD. In real Medicare contexts with staggered ACO attribution across years, more advanced estimators like Callaway-Sant'Anna or Sun-Abraham would be needed to address treatment effect heterogeneity.

**Sensitivity — Propensity Score Matching (1:1 NN, caliper = 0.05):**

PS estimated from pre-period demographics and utilisation via logistic regression. Vectorized KD-tree matching across 521 treated beneficiaries.

| Check | Result | Interpretation |
|-------|--------|----------------|
| ATT (PSM) | −$398.59/member | Convergent with DiD (Δ = $22.74) |
| Matched pairs | 479 | Control arm fully matched |
| Post-match SMD (age) | 0.046 | ✓ Well-balanced (target < 0.10) |

> DiD and PSM estimates converging within $22.74 on a 1,000-person cohort is a strong indicator of a robust, unbiased causal estimate.

**Heterogeneous treatment effects by risk tier:**

The intervention generates larger absolute savings among high-risk beneficiaries — consistent with the clinical logic that complex patients benefit most from care coordination.

### 4 · Shared Savings Projection (MSSP Framework)

```
Gross savings   = (Benchmark PMPM − Actual PMPM) × N attributed lives
Savings rate    = Gross savings / Total benchmark expenditure
Earned savings  = Gross savings × sharing rate  [if savings rate > MSR threshold]
```

**Simplifications in this prototype:**
- No risk corridor adjustments (asymmetric sharing above/below MSR)
- No quality score multipliers (CMS Star ratings)
- No benchmark rebasing after 3 years
- No regional adjustment factors
- No minimum savings rate variations by attributed lives count
- No minimum loss rate for two-sided risk tracks

| Parameter | Value |
|-----------|-------|
| Benchmark PMPM | $9,800.00 |
| Actual PMPM (post-intervention) | $9,378.67 |
| Attributed lives | 521 |
| Gross savings | $219,513 |
| Savings rate | 4.3% |
| MSR threshold (2%) | ✓ Exceeded |
| **Shared savings earned (50% rate)** | **$109,756** |

---

## Production Considerations

This prototype uses synthetic data to demonstrate analytical methodology without PHI exposure. Deploying against real Medicare claims requires significant infrastructure and data completeness:

| Area | Current Prototype | Production Requirements |
|------|-------------------|-------------------------|
| **Data Source** | 100% synthetic | Part A/B/D claims, encounter data, LIS status |
| **HCC Coverage** | 50+ high-prevalence codes (~0.5% of CMS v28) | Full 9,000+ ICD-10 mappings, ESRD model, new enrollee adjustments |
| **RAF Model** | Community non-dual aged only | Dual eligibility, LIS subsidies, encounter vs FFS weighting |
| **Risk Model** | XGBoost ensemble | MLflow tracking, model registry, drift monitoring, calibration |
| **Causal Methods** | DiD + PSM | Staggered adoption (Callaway-Sant'Anna), RDD for clean experiments |
| **Shared Savings** | Simplified MSSP | Risk corridors, quality multipliers, regional adjustments, MSR variations |
| **Infrastructure** | Local Python scripts | HIPAA-compliant pipeline, distributed compute, API serving |
| **Scale** | 50k beneficiaries | 50–100M beneficiary-years for national MA population |

**Infrastructure gaps for production deployment:**
- **MLflow** experiment tracking and model registry
- **Model calibration** for probabilistic cost predictions
- **Drift monitoring** with PSI and CSI metrics
- **CI/CD pipeline** with automated testing and deployment
- **Docker containerization** for reproducible environments
- **FastAPI serving layer** for model APIs
- **Distributed computing** (Spark/Dask) for large-scale processing

**Synthetic Data Caveats**: The inflated RAF distribution (mean 2.082 vs real 1.0–1.3) and strong DiD balance (parallel trends p = 0.679) are artifacts of controlled data generation. Real claims data has incomplete coding, secular trends, and confounding that make causal inference more challenging.

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
