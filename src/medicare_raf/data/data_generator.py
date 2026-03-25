"""
data_generator.py
-----------------
Generates a realistic synthetic Medicare beneficiary claims dataset
mimicking CMS SynPUF (Synthetic Public Use Files) structure.

Used for prototyping HCC/RAF modeling, risk stratification,
and shared savings attribution in the absence of PHI-protected claims.

N = 50,000 beneficiaries across two calendar years (baseline + intervention)
covering a realistic ACO-like population distribution.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator, Field
from typing import List, Optional
from ..modeling.hcc_mapper import ICD10_TO_HCC

SEED = 42


class BeneficiaryRecord(BaseModel):
    """Pydantic model for beneficiary data validation."""
    bene_id: str = Field(..., min_length=1, max_length=20)
    age: int = Field(..., ge=65, le=120)
    sex: str = Field(..., pattern="^[MF]$")
    race_ethnicity: str
    dual_eligible: int = Field(..., ge=0, le=1)
    risk_tier: str = Field(..., pattern="^(low|moderate|high)$")
    icd10_codes: List[str]
    intervention: int = Field(..., ge=0, le=1)
    county_fips: str = Field(..., min_length=5, max_length=5)
    plan_type: str

    @validator('icd10_codes')
    def validate_icd10_codes(cls, v):
        """Validate ICD-10 codes are properly formatted."""
        for code in v:
            if not isinstance(code, str) or len(code) < 3:
                raise ValueError(f"Invalid ICD-10 code: {code}")
        return v


class UtilizationRecord(BaseModel):
    """Pydantic model for utilization data validation."""
    bene_id: str
    year: int = Field(..., ge=0, le=1)
    period: str = Field(..., pattern="^(pre|post)$")
    intervention: int = Field(..., ge=0, le=1)
    risk_tier: str = Field(..., pattern="^(low|moderate|high)$")
    age: int = Field(..., ge=65, le=120)
    sex: str = Field(..., pattern="^[MF]$")
    dual_eligible: int = Field(..., ge=0, le=1)
    county_fips: str
    total_cost: float = Field(..., ge=0)
    ip_admits: int = Field(..., ge=0)
    ed_visits: int = Field(..., ge=0)


def validate_beneficiary_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Validate beneficiary cohort data using Pydantic."""
    validated_records = []
    for _, row in cohort.iterrows():
        try:
            record = BeneficiaryRecord(**row.to_dict())
            validated_records.append(record.dict())
        except Exception as e:
            raise ValueError(f"Validation failed for bene_id {row.get('bene_id', 'unknown')}: {e}")

    return pd.DataFrame(validated_records)


def validate_utilization_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate utilization panel data using Pydantic."""
    validated_records = []
    for _, row in panel.iterrows():
        try:
            record = UtilizationRecord(**row.to_dict())
            validated_records.append(record.dict())
        except Exception as e:
            raise ValueError(f"Validation failed for bene_id {row.get('bene_id', 'unknown')}: {e}")

    return pd.DataFrame(validated_records)


def _sample_icd10_codes(n_codes: int, risk_tier: str, rng) -> list:
    """Sample ICD-10 codes based on beneficiary risk tier."""
    all_codes = list(ICD10_TO_HCC.keys())

    # High-prevalence codes by tier
    high_risk_codes = [
        "I500", "I501", "I5020", "I5022",           # CHF
        "E1140", "E1141",                            # T2DM CKD
        "N184", "N185", "N186",                      # CKD Stage 4-6
        "I480", "I481", "I4811",                     # AFib
        "C3410", "C7800", "C800",                    # Cancer
        "G20", "G3500",                              # Parkinson's / MS
        "J4420", "J441",                             # COPD
    ]
    moderate_risk_codes = [
        "E119", "E1165",                             # T2DM w/o complication
        "I702", "I7020", "I739",                     # Vascular
        "I2101", "I219",                             # AMI
        "F320", "F321", "F3289",                     # Depression
        "N181", "N182", "N183",                      # CKD Stage 1-3
        "E6601",                                     # Morbid obesity
        "M0500", "M0510",                            # RA
    ]

    if risk_tier == "high":
        pool = high_risk_codes + moderate_risk_codes
    elif risk_tier == "moderate":
        pool = moderate_risk_codes + rng.choice(all_codes, 20, replace=False).tolist()
    else:
        pool = rng.choice(all_codes, 30, replace=False).tolist()

    n_sample = min(n_codes, len(pool))
    return rng.choice(pool, size=n_sample, replace=False).tolist()


def generate_beneficiary_cohort(
    n: int = 50_000,
    seed: int = SEED,
    intervention_prevalence: float = 0.50,
) -> pd.DataFrame:
    """
    Generate a synthetic Medicare beneficiary cohort.

    Parameters
    ----------
    n                        : int, number of beneficiaries
    seed                     : int, random seed
    intervention_prevalence  : float, share assigned to intervention arm

    Returns
    -------
    pd.DataFrame with columns:
        bene_id, age, sex, race_ethnicity, dual_eligible,
        risk_tier, icd10_codes, intervention,
        county_fips, plan_type
    """
    rng = np.random.default_rng(seed)

    bene_ids = [f"BENE_{i:07d}" for i in range(1, n + 1)]

    # Age distribution mimicking Medicare Advantage (65+)
    age_probs = [0.28, 0.24, 0.20, 0.14, 0.08, 0.06]  # 65-69..90+
    age_bands = ["65-69", "70-74", "75-79", "80-84", "85-89", "90+"]
    age_band_map = {"65-69": (65, 70), "70-74": (70, 75), "75-79": (75, 80),
                    "80-84": (80, 85), "85-89": (85, 90), "90+": (90, 97)}
    age_band_choices = rng.choice(age_bands, size=n, p=age_probs)
    ages = np.array([rng.integers(*age_band_map[ab]) for ab in age_band_choices])

    # Sex
    sexes = rng.choice(["M", "F"], size=n, p=[0.44, 0.56])

    # Race/ethnicity (CMS MA distribution)
    race_probs = [0.76, 0.10, 0.07, 0.04, 0.03]
    races = rng.choice(
        ["Non-Hispanic White", "Black", "Hispanic", "Asian", "Other"],
        size=n, p=race_probs
    )

    # Dual eligibility (Medicare + Medicaid) — ~20% in MA
    dual = rng.choice([0, 1], size=n, p=[0.80, 0.20])

    # Risk tier (drives disease complexity)
    # High risk more common among older, dual-eligible
    risk_base = np.where(ages >= 80, 0.30, np.where(ages >= 75, 0.20, 0.12))
    risk_base += dual * 0.10
    risk_tier_probs = np.column_stack([
        risk_base,
        np.full(n, 0.35),
        1 - risk_base - 0.35,
    ])
    risk_tier_probs = np.clip(risk_tier_probs, 0.01, None)
    risk_tier_probs /= risk_tier_probs.sum(axis=1, keepdims=True)

    risk_tiers = np.array([
        rng.choice(["high", "moderate", "low"], p=risk_tier_probs[i])
        for i in range(n)
    ])

    # ICD-10 codes (number varies by risk tier)
    n_codes_map = {"high": (6, 14), "moderate": (2, 7), "low": (0, 3)}
    icd10_codes = [
        _sample_icd10_codes(
            rng.integers(*n_codes_map[rt]),
            rt,
            rng,
        )
        for rt in risk_tiers
    ]

    # Intervention assignment (roughly 50/50 for DiD/PSM analysis)
    intervention = rng.choice([0, 1], size=n, p=[1 - intervention_prevalence,
                                                    intervention_prevalence])

    # County FIPS (simulate a multi-county ACO)
    county_fips = rng.choice(
        ["21097", "21151", "21067", "21179", "21017",
         "21209", "21227", "21239", "21047", "21113"],
        size=n,
    )

    # Plan type
    plan_types = rng.choice(
        ["HMO", "PPO", "PFFS", "SNP"],
        size=n,
        p=[0.55, 0.30, 0.08, 0.07],
    )

    cohort = pd.DataFrame({
        "bene_id": bene_ids,
        "age": ages,
        "sex": sexes,
        "race_ethnicity": races,
        "dual_eligible": dual,
        "risk_tier": risk_tiers,
        "icd10_codes": icd10_codes,
        "intervention": intervention,
        "county_fips": county_fips,
        "plan_type": plan_types,
    })

    # Validate the generated data
    validated_cohort = validate_beneficiary_cohort(cohort)
    return validated_cohort


def generate_utilization_panel(
    cohort: pd.DataFrame,
    seed: int = SEED,
    intervention_effect_pmpm: float = -420.0,
) -> pd.DataFrame:
    """
    Generate two-year panel (pre/post) utilization data.

    Intervention arm receives a realistic cost reduction signal
    (default -$420 PMPM ≈ a plausible ACO shared savings effect).

    Parameters
    ----------
    cohort                    : pd.DataFrame from generate_beneficiary_cohort()
    seed                      : int
    intervention_effect_pmpm  : float, true causal effect (negative = cost reduction)

    Returns
    -------
    pd.DataFrame with one row per bene-year (2 rows per bene)
    """
    rng = np.random.default_rng(seed + 1)

    cost_base = {
        "high":     12_800,
        "moderate":  9_200,
        "low":       6_400,
    }

    records = []
    for _, row in cohort.iterrows():
        base = cost_base[row["risk_tier"]]

        # Add age-related cost scaling
        age_scale = 1.0 + (row["age"] - 72) * 0.008

        # Baseline year (year=0, pre-intervention)
        noise_pre = rng.normal(0, base * 0.30)
        cost_pre = max(0, base * age_scale + noise_pre)

        # Post-intervention year (year=1)
        noise_post = rng.normal(0, base * 0.30)
        secular_trend = rng.normal(250, 80)   # secular cost increase ~$250/yr
        treatment_effect = intervention_effect_pmpm if row["intervention"] == 1 else 0.0

        # High-risk patients get a larger absolute effect
        if row["risk_tier"] == "high" and row["intervention"] == 1:
            treatment_effect *= 1.4

        cost_post = max(0, cost_pre + secular_trend + treatment_effect + noise_post)

        # IP admissions
        ip_base = {"high": 0.35, "moderate": 0.18, "low": 0.06}
        ip_admits_pre  = rng.poisson(ip_base[row["risk_tier"]])
        ip_admits_post = rng.poisson(
            ip_base[row["risk_tier"]] * (0.85 if row["intervention"] == 1 else 1.0)
        )

        # ED visits
        ed_base = {"high": 1.8, "moderate": 0.9, "low": 0.3}
        ed_visits_pre  = rng.poisson(ed_base[row["risk_tier"]])
        ed_visits_post = rng.poisson(
            ed_base[row["risk_tier"]] * (0.88 if row["intervention"] == 1 else 1.0)
        )

        for year, cost, ip, ed in [
            (0, cost_pre, ip_admits_pre, ed_visits_pre),
            (1, cost_post, ip_admits_post, ed_visits_post),
        ]:
            records.append({
                "bene_id": row["bene_id"],
                "year": year,
                "period": "pre" if year == 0 else "post",
                "intervention": row["intervention"],
                "risk_tier": row["risk_tier"],
                "age": row["age"],
                "sex": row["sex"],
                "dual_eligible": row["dual_eligible"],
                "county_fips": row["county_fips"],
                "total_cost": round(cost, 2),
                "ip_admits": ip,
                "ed_visits": ed,
            })

    panel = pd.DataFrame(records)

    # Validate the generated data
    validated_panel = validate_utilization_panel(panel)
    return validated_panel


if __name__ == "__main__":
    print("Generating 50,000-beneficiary cohort...")
    cohort = generate_beneficiary_cohort(n=50_000)
    print(f"  Cohort shape: {cohort.shape}")
    print(f"  Risk tier distribution:\n{cohort['risk_tier'].value_counts()}")

    print("\nGenerating utilization panel...")
    panel = generate_utilization_panel(cohort)
    print(f"  Panel shape: {panel.shape}")

    cohort.to_parquet("data/processed/beneficiary_cohort.parquet", index=False)
    panel.to_parquet("data/processed/utilization_panel.parquet", index=False)
    print("\nSaved to data/processed/")
