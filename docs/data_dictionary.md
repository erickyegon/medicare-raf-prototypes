# Medicare RAF Analytics Data Dictionary

This document describes the synthetic data schema used in the Medicare RAF analytics pipeline.

## Beneficiary Cohort (`beneficiary_cohort.parquet`)

| Field | Type | Description | Example | Validation |
|-------|------|-------------|---------|------------|
| `bene_id` | string | Unique beneficiary identifier | "BENE_000001" | 7-15 characters |
| `age` | int | Age in years | 75 | 65-120 |
| `sex` | string | Biological sex | "F" | "M" or "F" |
| `race_ethnicity` | string | Racial/ethnic category | "Non-Hispanic White" | CMS MA categories |
| `dual_eligible` | int | Medicare-Medicaid dual eligibility | 0 | 0 or 1 |
| `risk_tier` | string | Clinical risk tier (synthetic) | "high" | "low", "moderate", "high" |
| `icd10_codes` | list[string] | ICD-10 diagnosis codes | ["I500", "E119"] | Valid ICD-10 format |
| `intervention` | int | Treatment group assignment | 1 | 0 or 1 |
| `county_fips` | string | County FIPS code | "21097" | 5-digit string |
| `plan_type` | string | Medicare Advantage plan type | "HMO" | "HMO", "PPO", "PFFS", "SNP" |

## Utilization Panel (`utilization_panel.parquet`)

| Field | Type | Description | Example | Validation |
|-------|------|-------------|---------|------------|
| `bene_id` | string | Beneficiary identifier | "BENE_000001" | Matches cohort |
| `year` | int | Calendar year | 0 | 0 (pre) or 1 (post) |
| `period` | string | Time period label | "pre" | "pre" or "post" |
| `intervention` | int | Treatment assignment | 1 | 0 or 1 |
| `risk_tier` | string | Risk tier | "high" | "low", "moderate", "high" |
| `age` | int | Age at time of utilization | 75 | 65-120 |
| `sex` | string | Biological sex | "F" | "M" or "F" |
| `dual_eligible` | int | Dual eligibility status | 0 | 0 or 1 |
| `county_fips` | string | County FIPS code | "21097" | 5-digit string |
| `total_cost` | float | Total medical costs (PMPM) | 8500.50 | ≥ 0 |
| `ip_admits` | int | Inpatient admissions | 1 | ≥ 0 |
| `ed_visits` | int | Emergency department visits | 2 | ≥ 0 |

## RAF Calculation Features

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `raf_score` | float | Risk Adjustment Factor | HCC coefficients + demographics |
| `demographic_raf` | float | Demographic component | Age-sex coefficients |
| `hcc_raf` | float | Hierarchical condition component | Sum of HCC coefficients |
| `hcc_count` | int | Number of triggered HCCs | Count of non-zero HCC flags |
| `has_cancer` | int | Cancer diagnosis flag | HCC 8,9,10,11,12 |
| `has_chf` | int | Congestive heart failure flag | HCC 85 |
| `has_diabetes` | int | Diabetes diagnosis flag | HCC 17,18,19 |
| `has_ckd` | int | Chronic kidney disease flag | HCC 134-138 |
| `has_copd` | int | COPD diagnosis flag | HCC 111 |
| `has_afib` | int | Atrial fibrillation flag | HCC 96 |

## Risk Stratification Model Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `age_scaled` | float | Age normalized to mean | (-3, +3) |
| `age_sq` | float | Age squared | (4225, 14400) |
| `is_female` | int | Female indicator | 0 or 1 |
| `dual_eligible` | int | Dual eligibility | 0 or 1 |
| `chf_afib` | int | CHF + AFib comorbidity | 0 or 1 |
| `chf_diabetes` | int | CHF + Diabetes comorbidity | 0 or 1 |
| `ckd_diabetes` | int | CKD + Diabetes comorbidity | 0 or 1 |
| `cancer_age` | float | Cancer × age interaction | 0 or age_scaled |

## Data Generation Notes

- **Synthetic Data**: All data is artificially generated for methodological demonstration
- **Realism**: Distributions approximate CMS Medicare Advantage populations
- **PHI Compliance**: No real beneficiary data or protected health information
- **Scale**: Default cohort size is 50,000 beneficiaries × 2 years = 100,000 records
- **Intervention Effect**: Treatment group receives -$420 PMPM cost reduction signal