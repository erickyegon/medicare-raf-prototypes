"""
hcc_mapper.py
-------------
Maps ICD-10-CM diagnosis codes to CMS HCC categories (v28 model).
Uses a representative subset of high-prevalence HCC mappings
derived from CMS Medicare Advantage Risk Adjustment documentation.

CMS HCC v28 Reference:
https://www.cms.gov/medicare/health-plans/medicareadvtgspecratestats/risk-adjustors
"""

# ── CMS HCC v28 coefficient table (community non-dual aged segment) ──────────
# Source: CY2024 CMS-HCC Model Coefficients (community, non-dual, aged)
HCC_COEFFICIENTS_V28 = {
    # Diabetes
    17:  0.302,   # Diabetes with Acute Complications
    18:  0.302,   # Diabetes with Chronic Complications
    19:  0.118,   # Diabetes without Complication

    # Heart
    85:  0.323,   # Congestive Heart Failure
    86:  0.218,   # Acute Myocardial Infarction
    88:  0.174,   # Angina Pectoris
    96:  0.421,   # Specified Heart Arrhythmias

    # COPD / Respiratory
    110: 0.335,   # Cystic Fibrosis
    111: 0.245,   # COPD
    112: 0.167,   # Fibrosis of Lung

    # Cancer
    8:   2.488,   # Metastatic Cancer
    9:   0.899,   # Lung and Other Severe Cancers
    10:  0.639,   # Lymphoma and Other Cancers
    11:  0.439,   # Colorectal, Bladder, and Other Cancers
    12:  0.150,   # Breast, Prostate, and Other Cancers

    # Renal
    134: 0.289,   # Dialysis Status
    135: 0.289,   # Renal Failure
    136: 0.143,   # Chronic Kidney Disease, Stage 5
    137: 0.138,   # Chronic Kidney Disease, Severe (Stage 4)
    138: 0.071,   # Chronic Kidney Disease, Moderate (Stage 3)

    # Neurological
    52:  1.149,   # Drug/Alcohol Psychosis
    54:  0.358,   # Drug Dependence
    55:  0.234,   # Alcohol Dependence
    57:  0.625,   # Schizophrenia
    58:  0.421,   # Major Depressive, Bipolar, and Paranoid Disorders
    70:  1.203,   # Quadriplegia
    71:  0.665,   # Paraplegia
    72:  0.415,   # Spinal Cord Disorders/Injuries
    75:  0.589,   # Myasthenia Gravis/Myoneural Disorders
    77:  0.597,   # Multiple Sclerosis
    78:  0.406,   # Parkinson's and Huntington's Diseases
    79:  0.448,   # Seizure Disorders and Convulsions

    # Musculoskeletal
    40:  0.455,   # Rheumatoid Arthritis and Inflammatory Connective Tissue Disease
    41:  0.174,   # Osteoarthritis of Hip or Knee

    # Vascular
    107: 0.299,   # Vascular Disease with Complications
    108: 0.178,   # Vascular Disease

    # Other chronic
    22:  0.178,   # Morbid Obesity
    23:  0.029,   # Other Significant Endocrine and Metabolic Disorders
}

# ── Simplified ICD-10 → HCC mapping (representative subset) ─────────────────
ICD10_TO_HCC = {
    # Diabetes
    "E1010": 17, "E1011": 17, "E1065": 17,   # T1DM with acute complication
    "E1100": 17, "E1101": 17,                  # T2DM with acute complication
    "E1140": 18, "E1141": 18, "E1142": 18,    # T2DM with diabetic CKD
    "E1151": 18, "E1152": 18,                  # T2DM with diabetic peripheral angiopathy
    "E119":  19, "E1165": 19, "E118":  19,    # T2DM without complication

    # Congestive Heart Failure
    "I500":  85, "I501":  85, "I5020": 85,
    "I5021": 85, "I5022": 85, "I5023": 85,
    "I5030": 85, "I5031": 85, "I5032": 85,
    "I5033": 85, "I5040": 85, "I5041": 85,
    "I5042": 85, "I5043": 85, "I509":  85,

    # AMI
    "I2101": 86, "I2102": 86, "I2109": 86,
    "I214":  86, "I219":  86,

    # COPD
    "J4420": 111, "J4421": 111, "J4422": 111,
    "J441":  111, "J449":  111, "J961":  111,

    # Lung cancer
    "C3410": 9,  "C3411": 9,  "C3412": 9,
    "C3490": 9,  "C3491": 9,  "C3492": 9,

    # Breast cancer
    "C5011": 12, "C5012": 12, "C5091": 12, "C509": 12,

    # Colorectal cancer
    "C180":  11, "C182":  11, "C183":  11, "C184": 11,
    "C185":  11, "C186":  11, "C187":  11, "C188": 11,
    "C189":  11, "C20":   11,

    # Metastatic cancer
    "C7800": 8,  "C7801": 8,  "C7802": 8,
    "C781":  8,  "C782":  8,  "C786":  8,
    "C787":  8,  "C7889": 8,  "C790":  8,
    "C791":  8,  "C792":  8,  "C7931": 8,
    "C7932": 8,  "C7951": 8,  "C7952": 8,
    "C7981": 8,  "C7989": 8,  "C799":  8,
    "C800":  8,  "C801":  8,

    # CKD
    "N181":  138, "N182": 138, "N183": 138,  # CKD Stage 1-3
    "N184":  137,                              # CKD Stage 4
    "N185":  136,                              # CKD Stage 5
    "N186":  134,                              # CKD Stage 6 (ESRD)
    "Z9931": 134,                              # Dependence on renal dialysis

    # Atrial fibrillation / arrhythmia
    "I480":  96, "I481": 96, "I4811": 96,
    "I4819": 96, "I482": 96, "I4820": 96,
    "I4821": 96, "I483": 96, "I484":  96,
    "I489":  96, "I491": 96, "I492":  96,

    # Vascular disease
    "I702":  108, "I7020": 108, "I7021": 108,
    "I7090": 108, "I739":  108,
    "I7001": 107, "I7011": 107, "I7012": 107,

    # Morbid obesity
    "E6601": 22, "E6609": 22,

    # Depression / bipolar
    "F3110": 58, "F3111": 58, "F3112": 58,
    "F320":  58, "F321":  58, "F322":  58,
    "F3289": 58, "F329":  58,

    # Rheumatoid arthritis
    "M0500": 40, "M0510": 40, "M0520": 40,
    "M0600": 40, "M0610": 40, "M0620": 40,
    "M0630": 40, "M0690": 40,

    # Seizure disorders
    "G40001": 79, "G40009": 79, "G40011": 79,
    "G40019": 79, "G40101": 79, "G40109": 79,

    # Parkinson's
    "G20":   78, "G210":  78, "G211": 78,

    # Multiple sclerosis
    "G3500": 77, "G3510": 77, "G3520": 77, "G353": 77,
}


def map_icd10_to_hcc(icd10_codes: list) -> set:
    """
    Map a list of ICD-10-CM codes to a set of HCC categories.

    Parameters
    ----------
    icd10_codes : list of str

    Returns
    -------
    set of int : unique HCC category numbers triggered
    """
    hccs = set()
    for code in icd10_codes:
        code_clean = code.replace(".", "").upper().strip()
        if code_clean in ICD10_TO_HCC:
            hccs.add(ICD10_TO_HCC[code_clean])
    return hccs


def get_hcc_coefficient(hcc: int) -> float:
    """Return CMS v28 RAF coefficient for an HCC."""
    return HCC_COEFFICIENTS_V28.get(hcc, 0.0)


def get_hcc_label(hcc: int) -> str:
    """Return human-readable label for an HCC number."""
    labels = {
        8: "Metastatic Cancer", 9: "Lung & Severe Cancers",
        10: "Lymphoma", 11: "Colorectal/Bladder Cancer",
        12: "Breast/Prostate Cancer", 17: "Diabetes w/ Acute Complications",
        18: "Diabetes w/ Chronic Complications", 19: "Diabetes w/o Complication",
        22: "Morbid Obesity", 40: "Rheumatoid Arthritis",
        52: "Drug/Alcohol Psychosis", 54: "Drug Dependence",
        55: "Alcohol Dependence", 57: "Schizophrenia",
        58: "Major Depression/Bipolar", 70: "Quadriplegia",
        71: "Paraplegia", 72: "Spinal Cord Disorders",
        75: "Myasthenia Gravis", 77: "Multiple Sclerosis",
        78: "Parkinson's Disease", 79: "Seizure Disorders",
        85: "Congestive Heart Failure", 86: "Acute MI",
        88: "Angina Pectoris", 96: "Atrial Fibrillation",
        107: "Vascular Disease w/ Complications", 108: "Vascular Disease",
        110: "Cystic Fibrosis", 111: "COPD",
        134: "Dialysis / ESRD", 135: "Renal Failure",
        136: "CKD Stage 5", 137: "CKD Stage 4",
        138: "CKD Stage 3",
    }
    return labels.get(hcc, f"HCC {hcc}")
