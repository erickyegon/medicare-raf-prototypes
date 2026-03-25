"""
raf_calculator.py
-----------------
Calculates Risk Adjustment Factor (RAF) scores for Medicare beneficiaries
using the CMS-HCC v28 model structure.

RAF score = demographic score + sum of HCC coefficients (with interaction terms)
A score of 1.0 represents average expected cost.

Reference: CMS Medicare Advantage Risk Adjustment Model (v28, CY2024)
"""

import pandas as pd

from medicare_raf.modeling.hcc_mapper import (
    get_hcc_coefficient,
    get_hcc_label,
    map_icd10_to_hcc,
)

# ── Demographic base coefficients (community non-dual aged) ──────────────────
# Age/sex coefficients from CMS v28 model
DEMOGRAPHIC_COEFFICIENTS = {
    # (age_band, sex) → coefficient
    ("65-69", "M"): 0.379,
    ("65-69", "F"): 0.316,
    ("70-74", "M"): 0.435,
    ("70-74", "F"): 0.371,
    ("75-79", "M"): 0.516,
    ("75-79", "F"): 0.453,
    ("80-84", "M"): 0.601,
    ("80-84", "F"): 0.528,
    ("85-89", "M"): 0.643,
    ("85-89", "F"): 0.565,
    ("90+", "M"): 0.712,
    ("90+", "F"): 0.634,
}

# ── Disease interaction coefficients (selected key interactions) ─────────────
INTERACTION_COEFFICIENTS = {
    frozenset({85, 96}): 0.175,  # CHF + AFib
    frozenset({85, 17}): 0.121,  # CHF + Diabetes
    frozenset({85, 18}): 0.121,  # CHF + Diabetes w/ chronic complications
    frozenset({111, 17}): 0.099,  # COPD + Diabetes
    frozenset({8, 85}): 0.224,  # Metastatic Cancer + CHF
    frozenset({134, 85}): 0.312,  # ESRD + CHF
    frozenset({134, 17}): 0.208,  # ESRD + Diabetes
    frozenset({17, 19}): 0.0,  # Use higher HCC only (17 > 19)
}


def get_age_band(age: int) -> str:
    if age < 70:
        return "65-69"
    elif age < 75:
        return "70-74"
    elif age < 80:
        return "75-79"
    elif age < 85:
        return "80-84"
    elif age < 90:
        return "85-89"
    else:
        return "90+"


def calculate_raf(
    age: int,
    sex: str,
    icd10_codes: list,
    new_enrollee: bool = False,
) -> dict:
    """
    Calculate RAF score for a single beneficiary.

    Parameters
    ----------
    age          : int, beneficiary age
    sex          : str, 'M' or 'F'
    icd10_codes  : list of str, ICD-10-CM diagnosis codes from claims
    new_enrollee : bool, use new enrollee model (no diagnosis data)

    Returns
    -------
    dict with keys:
        raf_score        : float, total risk score
        demographic_raf  : float
        hcc_raf          : float
        interaction_raf  : float
        hccs             : list of int
        hcc_labels       : list of str
        hcc_details      : dict {hcc: coefficient}
    """
    age_band = get_age_band(age)
    sex = sex.upper()

    # Demographic score
    demo_coeff = DEMOGRAPHIC_COEFFICIENTS.get((age_band, sex), 0.40)

    if new_enrollee:
        return {
            "raf_score": demo_coeff,
            "demographic_raf": demo_coeff,
            "hcc_raf": 0.0,
            "interaction_raf": 0.0,
            "hccs": [],
            "hcc_labels": [],
            "hcc_details": {},
        }

    # Map ICD-10 → HCC
    hccs = map_icd10_to_hcc(icd10_codes)

    # HCC score (sum of coefficients)
    hcc_details = {}
    hcc_raf = 0.0
    for hcc in sorted(hccs):
        coeff = get_hcc_coefficient(hcc)
        if coeff > 0:
            hcc_details[hcc] = coeff
            hcc_raf += coeff

    # Interaction terms
    interaction_raf = 0.0
    hcc_set = frozenset(hccs)
    for interaction_hccs, coeff in INTERACTION_COEFFICIENTS.items():
        if interaction_hccs.issubset(hcc_set):
            interaction_raf += coeff

    total_raf = demo_coeff + hcc_raf + interaction_raf

    return {
        "raf_score": round(total_raf, 4),
        "demographic_raf": round(demo_coeff, 4),
        "hcc_raf": round(hcc_raf, 4),
        "interaction_raf": round(interaction_raf, 4),
        "hccs": sorted(hccs),
        "hcc_labels": [get_hcc_label(h) for h in sorted(hccs)],
        "hcc_details": hcc_details,
    }


def calculate_raf_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised RAF calculation for a DataFrame.

    Expected columns: bene_id, age, sex, icd10_codes (list)
    Returns original DataFrame with added RAF columns.
    """
    results = df.apply(
        lambda row: calculate_raf(
            age=row["age"],
            sex=row["sex"],
            icd10_codes=row.get("icd10_codes", []),
        ),
        axis=1,
        result_type="expand",
    )
    return pd.concat([df, results], axis=1)


def estimate_pmpm_cost(raf_score: float, base_pmpm: float = 9_800.0) -> float:
    """
    Estimate annual per-member cost from RAF score.

    Base PMPM of $9,800/year ≈ average 2023 Medicare FFS annual spend.
    PMPM cost = RAF × base_pmpm

    Parameters
    ----------
    raf_score  : float
    base_pmpm  : float, average annual cost at RAF=1.0 (default $9,800)

    Returns
    -------
    float : estimated annual cost
    """
    return round(raf_score * base_pmpm, 2)


def summarise_cohort_raf(df: pd.DataFrame) -> dict:
    """
    Summarise RAF distribution for a cohort.

    Parameters
    ----------
    df : DataFrame with 'raf_score' column

    Returns
    -------
    dict of summary statistics
    """
    return {
        "n": len(df),
        "mean_raf": round(df["raf_score"].mean(), 3),
        "median_raf": round(df["raf_score"].median(), 3),
        "p25_raf": round(df["raf_score"].quantile(0.25), 3),
        "p75_raf": round(df["raf_score"].quantile(0.75), 3),
        "p90_raf": round(df["raf_score"].quantile(0.90), 3),
        "pct_raf_above_2": round((df["raf_score"] > 2.0).mean() * 100, 1),
        "estimated_total_cost": round(df["raf_score"].sum() * 9_800, 0),
    }
