# Medicare RAF + Clinical Performance Analytics Pipeline

**End-to-end prototype for CMS HCC/RAF risk adjustment, clinical risk stratification, and causal attribution of shared savings in value-based care settings.**

Built by [Erick K. Yegon, PhD](https://linkedin.com/in/erickyegon) · Epidemiologist & Data Science Leader

---

## Overview

This project demonstrates a production-style analytics pipeline for Medicare Advantage and ACO populations, modelling the core quantitative challenges in value-based care:

1. **HCC/RAF Scoring** — ICD-10 → CMS HCC v28 mapping → RAF score calculation for 50,000 synthetic beneficiaries
2. **Risk Stratification** — XGBoost classifier/regressor predicting risk tier and annual cost from HCC flags and demographics
3. **Causal Attribution (DiD + PSM)** — Estimating intervention impact on total cost of care, IP admissions, and ED utilisation
4. **Shared Savings Projection** — MSSP/ACO REACH benchmarking logic to translate causal estimates into projected earned savings

### Key Results (50,000-beneficiary synthetic cohort)

| Metric | Value |
|--------|-------|
| Mean RAF score | 2.131 |
| Risk model tier accuracy | 91.5% |
| Cost prediction MAE | $2,181 |
| DiD ATT (cost/member) | **−$391** (p < 0.0001) |
| PSM ATT (sensitivity check) | −$392 (convergent) |
| Projected gross savings | **$9.77M** |
| Shared savings earned (50% rate) | **$4.89M** |
| Parallel trends p-value | 0.582 (assumption holds) |

---

## Clinical & Business Context

Pearl Health and similar ACO enablement platforms answer one fundamental question:  
**"Did our clinical intervention reduce total cost of care, and by how much?"**

This pipeline operationalises that question end-to-end:

```
CMS Claims (ICD-10) 
    → HCC Mapping (v28 coefficients)
    → RAF Score (demographic + HCC + interaction terms)
    → Risk Stratification (XGBoost)
    → Causal Attribution (DiD / PSM)
    → Shared Savings (MSSP benchmarking)
    → Executive narrative
```

The methods here — DiD for value attribution, PSM for sensitivity analysis, HCC-based risk adjustment — are directly analogous to what ACO analytics teams use in production against real Medicare claims (Part A, B, D) and EHR data.

---

## Repository Structure

```
medicare-raf-prototypes/
│
├── src/
│   ├── hcc_mapper.py           # ICD-10 → HCC mapping (CMS v28 coefficients)
│   ├── raf_calculator.py       # RAF score calculation + PMPM cost estimation
│   ├── data_generator.py       # Synthetic 50k-beneficiary CMS-style cohort
│   ├── risk_stratification.py  # XGBoost risk tier + cost prediction
│   └── causal_attribution.py   # DiD, PSM, shared savings projection
│
├── run_pipeline.py             # End-to-end orchestration (all 4 stages)
│
├── notebooks/
│   ├── 01_hcc_raf_deep_dive.ipynb       # RAF methodology walkthrough
│   ├── 02_risk_stratification.ipynb     # Model training + SHAP analysis
│   └── 03_shared_savings_attribution.ipynb  # Causal inference results
│
├── data/
│   ├── processed/              # Generated data (gitignored)
│   └── README_data.md          # Data sources and CMS SynPUF instructions
│
├── reports/
│   └── figures/                # Pipeline output visualisations
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/erickyegon/medicare-raf-prototypes.git
cd medicare-raf-prototypes
pip install -r requirements.txt

# 2. Run full pipeline (generates data, trains model, runs causal analysis)
python run_pipeline.py

# 3. Explore in notebooks
jupyter lab notebooks/
```

Runtime: ~3 minutes on a standard laptop (50k beneficiaries).

---

## Methods Detail

### HCC/RAF Model (CMS v28)

RAF score = `demographic_coefficient` + `Σ HCC_coefficients` + `interaction_terms`

- Demographic coefficients: age-sex bands (65–69 through 90+), community non-dual aged segment
- 50+ ICD-10 → HCC mappings covering high-prevalence conditions (CHF, diabetes, CKD, cancer, COPD, AFib, neurological)
- Key interactions: CHF×AFib (+0.175), CHF×Diabetes (+0.121), ESRD×CHF (+0.312)
- PMPM cost = RAF × $9,800 (2023 average Medicare FFS spend)

### Risk Stratification (XGBoost)

Features: HCC flag columns (22 key HCCs), RAF components, demographic features, disease interaction indicators

| Metric | Value |
|--------|-------|
| Tier classification accuracy | 91.5% |
| Annual cost MAE | $2,181 |
| Annual cost R² | 0.450 |

Top predictors: HCC count, CKD status, CHF×AFib interaction, age scaling

### Causal Attribution

**Primary: Difference-in-Differences (TWFE)**
```
Y_it = α + β₁·Post_t + β₂·Treat_i + β₃·(Post_t × Treat_i) + γ·X_i + ε_it
```
β₃ = ATT (Average Treatment effect on the Treated)

- Clustered standard errors at beneficiary level
- Parallel trends check: p = 0.582 (assumption holds)
- Covariates: age, dual eligibility status

**Sensitivity: Propensity Score Matching (1:1 NN, caliper = 0.05)**
- PS estimated via logistic regression on demographics + utilisation history
- Post-match SMD (age): 0.001 (well below 0.10 threshold)
- DiD and PSM estimates converge ($−391 vs $−392) — robust finding

**Shared Savings (MSSP logic)**
- Gross savings = (Benchmark − Actual) × N attributed lives
- MSR threshold = 2% of benchmark (standard MSSP basic track)
- Shared savings = gross savings × 50% sharing rate

---

## Limitations & Production Considerations

This prototype uses synthetic data and a representative subset of ICD-10/HCC mappings. In a production context:

- **Full CMS v28 HCC map** contains ~9,000 ICD-10 codes across 86 HCC categories
- **Real claims data** (Part A, B, D) would include additional risk adjustment inputs: ESRD indicators, new enrollee flags, low-income subsidy status
- **Encounter data** would enable more precise risk adjustment for MA plans
- **Regression discontinuity (RDD)** and **synthetic control** methods should be considered for settings without clean pre-post periods
- **HIPAA compliance** infrastructure required for PHI-protected claims (de-identification, BAA, audit logging)

---

## Related Work

This pipeline draws on methods from:

- CMS Medicare Advantage Risk Adjustment documentation (CY2024 v28 model)
- Austin (2011): *An Introduction to Propensity Score Methods*
- Angrist & Pischke (2009): *Mostly Harmless Econometrics* (DiD chapter)
- Imai & Ratkovic (2014): Covariate balancing propensity score
- MedPAC (2023): Report to Congress on Medicare Advantage risk adjustment

---

## Author

**Erick K. Yegon, PhD** — Epidemiologist & Data Science Leader  
17+ years in global health analytics, causal inference, and clinical performance measurement  
[LinkedIn](https://linkedin.com/in/erickyegon) · [GitHub](https://github.com/erickyegon) · [ORCID](https://orcid.org/0000-0002-7055-4848)
