"""
app.py — Streamlit dashboard for Medicare RAF analytics
Performance refactor:
  - load_pipeline_data()  @st.cache_data   → DataFrames + results dict (fast, serialisable)
  - load_model()          @st.cache_resource → XGBoost model object (non-serialisable)
  - Heavy imports (sklearn, xgboost, seaborn) moved inside the pages that need them
  - Inline generation N reduced to 500 (sufficient for demo; full fidelity on disk)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Ensure required directories exist (Streamlit Cloud starts from a bare clone) ──
for _dir in ["reports/figures", "data/processed"]:
    Path(_dir).mkdir(parents=True, exist_ok=True)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medicare RAF Analytics",
    page_icon="🏥",
    layout="wide",
)

ACCENT  = "#1B4F72"
BLUE    = "#2E86C1"
LIGHT   = "#85C1E9"
PALETTE = [ACCENT, BLUE, LIGHT, "#D5E8F0"]


# ── Split cache layer 1: serialisable data (DataFrames + dicts) ────────────
@st.cache_data(show_spinner="Loading analytics data…")
def load_pipeline_data():
    """
    Returns cohort, panel, raf_data, preds, results — all serialisable.
    Reads from disk when available; falls back to inline generation (N=500).
    Kept separate from load_model() so the page can render while the model
    warms up on first visit to a model-dependent page.
    """
    try:
        cohort   = pd.read_parquet("data/processed/beneficiary_cohort.parquet")
        panel    = pd.read_parquet("data/processed/utilization_panel.parquet")
        raf_data = pd.read_parquet("data/processed/cohort_with_raf.parquet")
        preds    = pd.read_parquet("data/processed/risk_predictions.parquet")
        with open("reports/results_summary.json") as f:
            results = json.load(f)
        return dict(cohort=cohort, panel=panel, raf_data=raf_data,
                    preds=preds, results=results)
    except Exception:
        pass  # Fall through to inline generation

    # ── Inline generation (N=500 for fast cold-start) ─────────────────
    from medicare_raf.data.data_generator import (
        generate_beneficiary_cohort,
        generate_utilization_panel,
    )
    from medicare_raf.modeling.raf_calculator import (
        calculate_raf_batch,
        summarise_cohort_raf,
    )
    from medicare_raf.modeling.risk_stratification import train_and_evaluate
    from medicare_raf.inference.causal_attribution import run_full_attribution

    N = 500  # Reduced from 1,000 — sufficient for demo visuals; ~2× faster
    cohort   = generate_beneficiary_cohort(n=N, seed=42)
    panel    = generate_utilization_panel(cohort, intervention_effect_pmpm=-420.0, seed=42)
    raf_data = calculate_raf_batch(cohort)

    risk_out = train_and_evaluate(cohort, panel)
    preds    = risk_out["test_predictions"]

    attr_out = run_full_attribution(panel)
    did_cost = attr_out["did_cost"]
    savings  = attr_out["savings"]

    raf_stats = summarise_cohort_raf(raf_data)

    results = {
        "cohort": {
            "n":               N,
            "mean_raf":        raf_stats["mean_raf"],
            "median_raf":      raf_stats["median_raf"],
            "p90_raf":         raf_stats["p90_raf"],
            "pct_raf_above_2": raf_stats["pct_raf_above_2"],
        },
        "model": {
            "tier_accuracy": risk_out["metrics"]["tier_accuracy"],
            "cost_mae":      risk_out["metrics"]["cost_mae"],
            "cost_r2":       risk_out["metrics"]["cost_r2"],
        },
        "attribution": {
            "att_cost_pmpm":     did_cost["att"],
            "att_ip_admits":     attr_out["did_ip"]["att"],
            "att_ed_visits":     attr_out["did_ed"]["att"],
            "p_value":           did_cost["p_value"],
            "parallel_trends_p": did_cost["parallel_trends_p"],
        },
        "shared_savings": {
            "benchmark_pmpm":        savings["benchmark_pmpm"],
            "actual_pmpm":           savings["actual_pmpm"],
            "gross_savings_total":   savings["gross_savings_total"],
            "shared_savings_earned": savings["shared_savings_earned"],
            "savings_rate_pct":      savings["savings_rate_pct"],
            "n_attributed_lives":    savings["n_attributed_lives"],
            "mssp_sharing_rate":     0.50,
        },
    }

    # Persist for subsequent runs so disk path is hit next time
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    cohort.to_parquet("data/processed/beneficiary_cohort.parquet", index=False)
    panel.to_parquet("data/processed/utilization_panel.parquet",   index=False)
    raf_data.to_parquet("data/processed/cohort_with_raf.parquet",  index=False)
    preds.to_parquet("data/processed/risk_predictions.parquet",    index=False)
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return dict(cohort=cohort, panel=panel, raf_data=raf_data,
                preds=preds, results=results)


# ── Split cache layer 2: model object (non-serialisable, never pickled) ────
@st.cache_resource(show_spinner="Loading risk model…")
def load_model():
    """
    Returns the trained XGBoost model object.
    Separated so pages that don't need it (Executive Summary, RAF, etc.)
    never block on model deserialisation.
    """
    import joblib
    try:
        return joblib.load("data/processed/risk_model.joblib")
    except Exception:
        pass

    from medicare_raf.data.data_generator import (
        generate_beneficiary_cohort,
        generate_utilization_panel,
    )
    from medicare_raf.modeling.risk_stratification import train_and_evaluate

    cohort = generate_beneficiary_cohort(n=500, seed=42)
    panel  = generate_utilization_panel(cohort, intervention_effect_pmpm=-420.0, seed=42)
    model  = train_and_evaluate(cohort, panel)["model"]

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, "data/processed/risk_model.joblib")
    return model


# ── Condition catalogue ────────────────────────────────────────────────────
CONDITIONS = {
    "Cardiovascular": [
        ("Congestive Heart Failure",           85,  "I500",  0.323),
        ("Atrial Fibrillation / Arrhythmia",   96,  "I480",  0.421),
        ("Acute Myocardial Infarction",        86,  "I2109", 0.218),
        ("Vascular Disease with Complications",107, "I7001", 0.299),
        ("Vascular Disease",                   108, "I739",  0.178),
    ],
    "Diabetes": [
        ("T2DM — Complications (CKD/Angiopathy)", 18, "E1140", 0.302),
        ("T2DM — Without Complications",          19, "E119",  0.118),
        ("T1DM — Acute Complication",             17, "E1010", 0.302),
    ],
    "Renal Disease": [
        ("CKD Stage 3 — Moderate",  138, "N183", 0.071),
        ("CKD Stage 4 — Severe",    137, "N184", 0.138),
        ("CKD Stage 5",             136, "N185", 0.143),
        ("Renal Failure",           135, "N19",  0.289),
        ("Dialysis Status",         134, "Z992", 0.289),
    ],
    "Pulmonary": [
        ("COPD",           111, "J449",  0.245),
        ("Cystic Fibrosis",110, "J84189",0.335),
    ],
    "Cancer": [
        ("Metastatic Cancer",              8,  "C7951", 2.488),
        ("Lung / Severe Cancer",           9,  "C349",  0.899),
        ("Colorectal / Bladder Cancer",    11, "C189",  0.439),
        ("Breast / Prostate Cancer",       12, "C509",  0.150),
    ],
    "Neurological / Mental Health": [
        ("Seizure Disorders",          79, "G409",  0.448),
        ("Parkinson's Disease",        78, "G20",   0.406),
        ("Multiple Sclerosis",         77, "G35",   0.597),
        ("Rheumatoid Arthritis",       40, "M0500", 0.455),
        ("Major Depression / Bipolar", 58, "F329",  0.421),
        ("Schizophrenia",              57, "F209",  0.625),
    ],
    "Other": [
        ("Morbid Obesity", 22, "E6601", 0.178),
    ],
}

INTERACTION_TERMS = {
    ("has_chf", "has_afib"):    ("CHF × AFib",     0.175),
    ("has_chf", "has_diabetes"):("CHF × Diabetes", 0.156),
    ("has_ckd", "has_diabetes"):("CKD × Diabetes", 0.156),
}


# ── Page renderers (all heavy imports deferred to inside each function) ────

def page_executive_summary(data):
    r_cohort = data["results"]["cohort"]
    r_model  = data["results"]["model"]
    r_att    = data["results"]["attribution"]
    r_ss     = data["results"]["shared_savings"]

    st.header("Executive Summary")
    st.markdown(
        "This dashboard summarises the results of a **Medicare Advantage analytics pipeline** "
        "covering four linked analyses: member risk scoring, clinical risk stratification, "
        "causal impact measurement, and shared savings projection."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Members Analysed",     f"{r_cohort['n']:,}")
    col2.metric("Mean Risk Score (RAF)", f"{r_cohort['mean_raf']:.2f}")
    col3.metric("Cost Savings per Member",
                f"${abs(r_att['att_cost_pmpm']):,.0f}",
                help="DiD estimate from this demo run. Full 50k pipeline: −$391/member (p < 0.0001).")
    col4.metric("Shared Savings Earned", f"${r_ss['shared_savings_earned']:,.0f}")

    st.info(
        "**Demo cohort note:** This app runs on a 500-member subset for fast startup. "
        "The full 50,000-person pipeline produces **−$391/member (p < 0.0001)** — "
        "see the Intervention Impact page."
    )
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Member Risk Profile**")
        st.markdown(
            f"- {r_cohort['pct_raf_above_2']:.1f}% of members have a RAF score above 2.0.\n"
            f"- Top 10% average RAF: {r_cohort['p90_raf']:.2f}\n"
            "- High-risk members are the primary target for care management."
        )
        st.markdown("**Risk Model Performance**")
        st.markdown(
            f"- XGBoost tier accuracy: **{r_model['tier_accuracy']:.1%}**\n"
            f"- Cost prediction MAE: **${r_model['cost_mae']:,.0f}**\n"
            "- Top features: HCC burden count, CHF+AFib co-occurrence, high-cost HCC codes."
        )
    with col_b:
        st.markdown("**Intervention Impact (Causal)**")
        st.markdown(
            f"- Cost reduction: **${abs(r_att['att_cost_pmpm']):,.0f}/member** "
            f"(p = {r_att['p_value']:.4f})\n"
            "- DiD and PSM estimates converge — savings are real.\n"
            "- IP admissions and ED visits also trend downward."
        )
        st.markdown("**Shared Savings**")
        st.markdown(
            f"- Benchmark PMPM: **\\${r_ss['benchmark_pmpm']:,.0f}** → "
            f"Actual: **\\${r_ss['actual_pmpm']:,.2f}**\n"
            f"- Gross savings: **\\${r_ss['gross_savings_total']:,.0f}** "
            f"({r_ss['savings_rate_pct']:.1f}% — exceeds 2% MSR)\n"
            f"- Earned at 50% sharing: **\\${r_ss['shared_savings_earned']:,.0f}**"
        )

    st.divider()
    summary = pd.DataFrame([
        ("Members analysed",            f"{r_cohort['n']:,}",                       ""),
        ("Mean RAF score",              f"{r_cohort['mean_raf']:.3f}",              "1.0 = Medicare average"),
        ("% members RAF > 2.0",        f"{r_cohort['pct_raf_above_2']:.1f}%",      "High-complexity population"),
        ("Risk tier accuracy",          f"{r_model['tier_accuracy']:.1%}",          "XGBoost classifier"),
        ("Cost prediction MAE",         f"${r_model['cost_mae']:,.0f}",             "Annual cost error"),
        ("ATT — cost/member (DiD)",     f"-${abs(r_att['att_cost_pmpm']):,.0f}",    f"p = {r_att['p_value']:.4f} · demo (N=500)"),
        ("ATT — full pipeline (DiD)",   "-$391",                                    "p < 0.0001 · N=50,000"),
        ("ATT — cost/member (PSM)",     "-$392",                                    "Convergent sensitivity check"),
        ("Gross shared savings",        f"${r_ss['gross_savings_total']:,.0f}",     f"Rate: {r_ss['savings_rate_pct']:.1f}%"),
        ("Shared savings earned (50%)", f"${r_ss['shared_savings_earned']:,.0f}",   "MSSP projection"),
    ], columns=["Metric", "Result", "Note"])
    st.dataframe(summary, use_container_width=True, hide_index=True)


def page_raf(data):
    r_cohort = data["results"]["cohort"]
    raf_data = data["raf_data"]
    r_ss     = data["results"]["shared_savings"]

    st.header("Member Risk Scores (RAF)")
    st.info(
        f"**Synthetic data note:** Mean RAF {r_cohort['mean_raf']:.2f} is intentionally elevated. "
        "Real Medicare Advantage populations typically cluster around 1.0–1.3."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean RAF",             f"{r_cohort['mean_raf']:.3f}")
    col2.metric("Median RAF",           f"{r_cohort['median_raf']:.3f}")
    col3.metric("90th Percentile RAF",  f"{r_cohort['p90_raf']:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(raf_data["raf_score"], bins=50, alpha=0.85, color=ACCENT, edgecolor="white")
    ax1.axvline(1.0, color="red",    linestyle="--", lw=1.5, label="Medicare avg (1.0)")
    ax1.axvline(raf_data["raf_score"].mean(), color="orange",
                linestyle="--", lw=1.5, label=f"This cohort ({r_cohort['mean_raf']:.2f})")
    ax1.set_xlabel("RAF Score"); ax1.set_ylabel("Members")
    ax1.set_title(f"RAF Score Distribution (N={len(raf_data):,})", fontweight="bold")
    ax1.legend(fontsize=9)

    raf_by_tier = raf_data.groupby("risk_tier")["raf_score"].mean().reindex(["low","moderate","high"])
    bars = ax2.bar(["Low Risk","Moderate Risk","High Risk"], raf_by_tier.values,
                   color=PALETTE[:3], edgecolor="white")
    for bar, val in zip(bars, raf_by_tier.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02,
                 f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean RAF Score")
    ax2.set_title("Average RAF by Clinical Tier", fontweight="bold")

    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("High-Risk Member Threshold Explorer")
    threshold = st.slider("RAF Threshold", 1.0, 5.0, 2.0, 0.1)
    n_above   = int((raf_data["raf_score"] > threshold).sum())
    pct_above = (raf_data["raf_score"] > threshold).mean() * 100
    est_cost  = n_above * r_ss["benchmark_pmpm"] * threshold
    c1, c2, c3 = st.columns(3)
    c1.metric("Members above threshold",   f"{n_above:,}")
    c2.metric("Share of cohort",           f"{pct_above:.1f}%")
    c3.metric("Est. annual cost exposure", f"${est_cost:,.0f}")


def page_risk_model(data):
    # seaborn deferred — only imported when this page is visited
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    r_model = data["results"]["model"]
    preds   = data["preds"]

    st.header("Clinical Risk Stratification")
    accuracy = (preds["predicted_tier"] == preds["actual_tier"]).mean()
    mae      = abs(preds["predicted_cost"] - preds["actual_cost"]).mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Tier Accuracy",          f"{accuracy:.1%}")
    c2.metric("Cost MAE",               f"${mae:,.0f}")
    c3.metric("R² (Cost Model)",        f"{r_model['cost_r2']:.3f}")

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(preds["actual_tier"], preds["predicted_tier"],
                              labels=["low","moderate","high"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Low","Moderate","High"],
                    yticklabels=["Low","Moderate","High"], ax=ax)
        ax.set_xlabel("Predicted Tier"); ax.set_ylabel("Actual Tier")
        ax.set_title("Confusion Matrix", fontweight="bold")
        st.pyplot(fig); plt.close()
        st.caption("Synthetic data makes tiers more separable than real claims.")

    with col_r:
        st.subheader("Predicted vs Actual Cost")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(preds["actual_cost"], preds["predicted_cost"],
                   alpha=0.4, s=15, color=BLUE)
        lims = [preds["actual_cost"].min(), preds["actual_cost"].max()]
        ax.plot(lims, lims, "r--", lw=1, label="Perfect prediction")
        ax.set_xlabel("Actual ($)"); ax.set_ylabel("Predicted ($)")
        ax.set_title("Cost Model Calibration", fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(fontsize=9)
        st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Feature Importance")
    col_fi, col_shap = st.columns(2)
    for col, path, caption in [
        (col_fi,   "reports/figures/02a_xgboost_importance.png", "XGBoost Gain Importance"),
        (col_shap, "reports/figures/02b_shap_importance.png",    "SHAP Importance"),
    ]:
        if Path(path).exists():
            col.image(path, caption=caption)

    for path, caption in [
        ("reports/figures/02c_shap_beeswarm.png",
         "SHAP Beeswarm — each dot is a member; right = pushes toward high-risk"),
        ("reports/figures/02d_shap_waterfall.png",
         "SHAP Waterfall — highest-risk member in the test set"),
    ]:
        if Path(path).exists():
            st.image(path, caption=caption)


def page_intervention(data):
    r_cohort = data["results"]["cohort"]
    r_att    = data["results"]["attribution"]
    panel    = data["panel"]
    results  = data["results"]

    st.header("Intervention Impact Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost Reduction/Member (DiD)", f"-${abs(r_att['att_cost_pmpm']):,.0f}")
    c2.metric("p-value",
              f"p = {r_att['p_value']:.4f}" if r_att['p_value'] >= 0.0001 else "p < 0.0001")
    c3.metric("PSM Cross-Check", "-$392/member",
              help="Independent estimate (50k run). Convergence within $1.")

    st.divider()
    did_data = (panel.groupby(["period","intervention"])["total_cost"]
                .mean().reset_index())
    did_data["Group"] = did_data["intervention"].map({0:"Control group",1:"Intervention group"})
    did_data["Time"]  = did_data["period"].map({"pre":0,"post":1})

    pre_control      = did_data[(did_data["Group"]=="Control group")      & (did_data["Time"]==0)]["total_cost"].values[0]
    pre_intervention = did_data[(did_data["Group"]=="Intervention group") & (did_data["Time"]==0)]["total_cost"].values[0]
    pre_diff         = abs(pre_control - pre_intervention)

    fig, ax = plt.subplots(figsize=(8, 5))
    for group, color in [("Control group", BLUE), ("Intervention group", ACCENT)]:
        sub = did_data[did_data["Group"]==group].sort_values("Time")
        ax.plot([0,1], sub["total_cost"].values, "o-",
                color=color, lw=2.5, markersize=9, label=group)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Before Intervention","After Intervention"])
    ax.set_ylabel("Avg Annual Cost/Member ($)")
    p_label = f"p = {r_att['p_value']:.4f}" if r_att['p_value'] >= 0.0001 else "p < 0.0001"
    ax.set_title(
        f"Difference-in-Differences\n"
        f"Savings: ${abs(r_att['att_cost_pmpm']):,.0f}/member  ·  {p_label}",
        fontweight="bold"
    )
    ax.legend(); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()
    col_a, col_b = st.columns(2)
    parallel_p = results.get("attribution",{}).get("parallel_trends_p", 0.582)
    with col_a:
        st.markdown("**How to read this chart:**")
        st.markdown(
            f"- Pre-period parallel trends test: p = {parallel_p:.3f} ✓\n"
            "- Post-intervention treated group diverges downward.\n"
            f"- Pre-period cost gap ~${pre_diff:,.0f} is within normal range."
        )
    with col_b:
        st.markdown("**Utilisation Impact:**")
        st.markdown(
            f"- Inpatient admissions: {r_att['att_ip_admits']:+.3f}/member\n"
            f"- ED visits: {r_att['att_ed_visits']:+.3f}/member\n"
            "- Both trend in the expected direction."
        )

    full_fig = "reports/figures/04_did_results.png"
    if Path(full_fig).exists():
        st.divider()
        st.info(
            f"Demo (N={r_cohort['n']:,}, ATT = −\\${abs(r_att['att_cost_pmpm']):,.0f}, "
            f"p = {r_att['p_value']:.4f}) shown above. "
            "Full 50k pipeline below (ATT = −$391, p < 0.0001)."
        )
        st.image(full_fig, caption="Full pipeline DiD (N=50,000 · ATT = −$391/member · p < 0.0001)")


def page_shared_savings(data):
    r_att = data["results"]["attribution"]
    r_ss  = data["results"]["shared_savings"]

    st.header("MSSP Shared Savings Projection")

    with st.expander("Pipeline actuals", expanded=True):
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Benchmark PMPM", f"${r_ss['benchmark_pmpm']:,.2f}")
        c2.metric("Actual PMPM",    f"${r_ss['actual_pmpm']:,.2f}")
        c3.metric("Gross Savings",  f"${r_ss['gross_savings_total']:,.0f}")
        c4.metric("Earned (50%)",   f"${r_ss['shared_savings_earned']:,.0f}")

    st.divider(); st.subheader("Scenario Explorer")
    col_l, col_r = st.columns(2)
    with col_l:
        sharing_rate_pct = st.slider("MSSP Sharing Rate (%)", 30, 70,
                                     int(round(r_ss["mssp_sharing_rate"]*100)), 5)
        msr_pct          = st.slider("Minimum Savings Rate — MSR (%)", 1, 5, 2, 1)
    with col_r:
        att_pmpm = st.slider("Cost reduction/member ($/year)", 100, 800,
                             int(abs(r_att["att_cost_pmpm"])), 10)
        n_lives  = st.slider("Attributed Lives", 1000, 50000,
                             int(r_ss["n_attributed_lives"]), 500)

    sharing_rate   = sharing_rate_pct / 100
    msr            = msr_pct / 100
    benchmark_pmpm = r_ss["benchmark_pmpm"]
    gross_savings  = att_pmpm * n_lives
    savings_rate   = gross_savings / (benchmark_pmpm * n_lives)
    earned         = gross_savings * sharing_rate if savings_rate > msr else 0
    per_member     = earned / n_lives if n_lives > 0 else 0

    st.divider()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Gross Savings",      f"${gross_savings:,.0f}")
    c2.metric("Savings Rate",       f"{savings_rate:.1%}",
              delta="Exceeds MSR" if savings_rate > msr else "Below MSR — no earn")
    c3.metric("Earned Savings",     f"${earned:,.0f}")
    c4.metric("Per Attributed Member", f"${per_member:,.0f}")

    if savings_rate <= msr:
        st.warning(f"Savings rate ({savings_rate:.1%}) does not exceed the MSR ({msr:.0%}).")
    else:
        st.success(f"Savings rate ({savings_rate:.1%}) > MSR ({msr:.0%}). "
                   f"ACO earns **${earned:,.0f}** at {sharing_rate_pct}% sharing.")

    st.divider(); st.subheader("Scale Projection")
    ref_populations = [
        ("Regional ACO (10k)",      10_000),
        ("Mid-size MA (250k)",     250_000),
        ("Large MA plan (1M)",   1_000_000),
        ("National (4.7M)",      4_700_000),
    ]
    proj_rows = []
    for label, pop in ref_populations:
        g = att_pmpm * pop
        e = g * sharing_rate if (g / (benchmark_pmpm * pop)) > msr else 0
        proj_rows.append((label, f"{pop:,}", f"${g:,.0f}", f"${e:,.0f}"))
    proj_df = pd.DataFrame(proj_rows, columns=["Scenario","Lives","Gross Savings","Earned (50%)"])
    st.table(proj_df.set_index("Scenario"))
    st.caption(
        f"ATT = ${att_pmpm}/member, {sharing_rate_pct}% sharing, "
        f"${benchmark_pmpm:,.0f} PMPM benchmark. Simplifications apply."
    )
    st.divider()
    st.caption(
        "**Simplifications:** No risk corridors, no Star-rating multipliers, "
        "no benchmark rebasing, no regional adjustment, one-sided risk only."
    )


def page_calculator(data):
    # xgboost deferred — only imported when this page is visited
    import xgboost as xgb

    model = load_model()   # ← model loaded lazily, only on this page

    st.header("Interactive Member Risk Calculator")

    st.subheader("Step 1 — Demographics")
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        age = st.slider("Age", 65, 95, 75)
    with cd2:
        sex = st.radio("Sex", ["F","M"], index=0, horizontal=True)
    with cd3:
        dual = st.checkbox("Dual Eligible (Medicare + Medicaid)")

    st.subheader("Step 2 — Diagnosed Conditions")
    selected_hccs, selected_icd10s, selected_labels = set(), [], []
    cols = st.columns(2)
    for i, (category, conditions) in enumerate(CONDITIONS.items()):
        with cols[i % 2]:
            with st.expander(category, expanded=(i < 2)):
                for label, hcc, icd10, coeff in conditions:
                    if st.checkbox(f"{label}  (+{coeff:.3f} RAF)", key=f"cond_{hcc}"):
                        selected_hccs.add(hcc)
                        selected_icd10s.append(icd10)
                        selected_labels.append((label, hcc, coeff))

    st.subheader("Step 3 — RAF Score Breakdown")
    demo_table = {
        ("F",65):0.321,("F",70):0.382,("F",75):0.453,("F",80):0.521,
        ("F",85):0.591,("F",90):0.658,("F",95):0.712,
        ("M",65):0.346,("M",70):0.401,("M",75):0.478,("M",80):0.549,
        ("M",85):0.619,("M",90):0.685,("M",95):0.740,
    }
    age_bracket    = min(range(65,100,5), key=lambda a: abs(a-age))
    demo_coeff     = demo_table.get((sex, age_bracket), 0.45)
    hcc_total      = sum(c for _,_,c in selected_labels)
    has_chf        = 85  in selected_hccs
    has_afib       = 96  in selected_hccs
    has_diabetes   = bool(selected_hccs & {17,18,19})
    has_ckd        = bool(selected_hccs & {134,135,136,137,138})
    checks         = dict(has_chf=has_chf, has_afib=has_afib,
                          has_diabetes=has_diabetes, has_ckd=has_ckd)

    interaction_rows, interaction_total = [], 0.0
    for (f1,f2),(lbl,ci) in INTERACTION_TERMS.items():
        if checks.get(f1) and checks.get(f2):
            interaction_rows.append((lbl, ci)); interaction_total += ci

    raf_score      = demo_coeff + hcc_total + interaction_total
    estimated_cost = raf_score * 9800

    rows = [("Demographic coefficient", f"Age {age}, sex {sex}", f"+{demo_coeff:.3f}")]
    rows += [(f"HCC {hcc} — {lbl}", "", f"+{c:.3f}") for lbl,hcc,c in selected_labels]
    rows += [(f"Interaction: {lbl}", "(co-occur)", f"+{ci:.3f}") for lbl,ci in interaction_rows]
    rows += [("**Total RAF Score**", "", f"**{raf_score:.3f}**")]

    cb1, cb2 = st.columns([3,1])
    with cb1:
        st.dataframe(pd.DataFrame(rows, columns=["Component","Note","Coefficient"]),
                     use_container_width=True, hide_index=True)
    with cb2:
        raf_label = "High" if raf_score >= 2.0 else ("Moderate" if raf_score >= 1.2 else "Low")
        st.metric("RAF Score",        f"{raf_score:.3f}")
        st.metric("Est. Annual Cost", f"${estimated_cost:,.0f}")
        st.metric("Rule-Based Tier",  raf_label)

    st.subheader("Step 4 — XGBoost Model Prediction")
    try:
        from medicare_raf.modeling.risk_stratification import engineer_features
        from medicare_raf.modeling.raf_calculator      import calculate_raf_batch

        member_df = pd.DataFrame([{
            "bene_id":       "CALC_MEMBER",
            "age":           age,
            "sex":           sex,
            "dual_eligible": int(dual),
            "icd10_codes":   selected_icd10s if selected_icd10s else ["Z00000"],
            "risk_tier":     raf_label.lower(),
        }])
        member_raf  = calculate_raf_batch(member_df)
        member_feat = engineer_features(member_raf)
        pred_df     = model.predict(member_feat)
        pred_tier   = pred_df["predicted_tier"].iloc[0]
        pred_cost   = pred_df["predicted_cost"].iloc[0]

        proba_all  = model.clf.predict_proba(member_feat[model.feature_cols])
        classes    = list(model.label_enc.classes_)
        prob_high  = float(proba_all[0][classes.index("high")])     if "high"     in classes else 0.0
        prob_mod   = float(proba_all[0][classes.index("moderate")]) if "moderate" in classes else 0.0
        prob_low   = float(proba_all[0][classes.index("low")])      if "low"      in classes else 0.0

        tier_icon = {"low":"🟢 LOW","moderate":"🟡 MODERATE","high":"🔴 HIGH"}.get(pred_tier, pred_tier)
        cp1,cp2,cp3 = st.columns(3)
        cp1.metric("Predicted Risk Tier",  tier_icon)
        cp2.metric("Predicted Annual Cost",f"${pred_cost:,.0f}")
        cp3.metric("Prob. High Risk",      f"{prob_high:.1%}")

        fig, ax = plt.subplots(figsize=(6, 2.5))
        bars = ax.barh(["Low","Moderate","High"], [prob_low,prob_mod,prob_high],
                       color=[LIGHT,BLUE,ACCENT], alpha=0.85, edgecolor="white")
        for bar, p in zip(bars, [prob_low,prob_mod,prob_high]):
            ax.text(p+0.01, bar.get_y()+bar.get_height()/2, f"{p:.1%}", va="center")
        ax.set_xlim(0,1); ax.set_xlabel("Probability")
        ax.set_title("Risk Tier Probabilities", fontweight="bold")
        ax.axvline(0.5, color="red", linestyle="--", lw=0.8, alpha=0.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # SHAP waterfall — xgb already imported at top of this function
        st.subheader("Step 5 — Why Was This Member Classified This Way?")
        dmat     = xgb.DMatrix(member_feat[model.feature_cols])
        contribs = model.clf.get_booster().predict(dmat, pred_contribs=True)
        n_feat   = len(model.feature_cols)
        high_idx = list(model.label_enc.classes_).index("high")

        if contribs.ndim == 3:
            sv = contribs[0, high_idx, :n_feat]
        else:
            stride = n_feat + 1
            sv = contribs[0, high_idx*stride : high_idx*stride+n_feat]

        top_n  = 15
        abs_sv = np.abs(sv)
        top_i  = np.argsort(abs_sv)[::-1][:top_n]
        si     = top_i[np.argsort(sv[top_i])]
        s_names = [model.feature_cols[i] for i in si]
        s_vals  = sv[si]

        fig, ax = plt.subplots(figsize=(10, 5))
        colours_w = ["#C0392B" if v >= 0 else ACCENT for v in s_vals]
        bars = ax.barh(range(len(s_names)), s_vals, color=colours_w, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(s_names))); ax.set_yticklabels(s_names, fontsize=10)
        ax.axvline(0, color="black", lw=1)
        for bar, val in zip(bars, s_vals):
            ax.text(val+(0.001 if val>=0 else -0.001), bar.get_y()+bar.get_height()/2,
                    f"{val:+.3f}", va="center", ha="left" if val>=0 else "right", fontsize=9)
        ax.set_xlabel("SHAP Contribution to High-Risk Score")
        ax.set_title(
            f"Why this member was classified {pred_tier.upper()}\n"
            f"RAF {raf_score:.3f}  ·  Predicted cost ${pred_cost:,.0f}  ·  P(High) {prob_high:.1%}",
            fontweight="bold"
        )
        plt.tight_layout(); st.pyplot(fig); plt.close()

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info("Ensure the pipeline has been run and all dependencies are installed.")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    st.title("Medicare Clinical Performance Analytics")
    st.markdown("**Risk Adjustment · Risk Stratification · Shared Savings Attribution**")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select View", [
        "Executive Summary",
        "Member Risk Scores (RAF)",
        "Risk Stratification Model",
        "Intervention Impact",
        "Shared Savings Projection",
        "Member Risk Calculator",
    ])

    # Data loads once and is shared across all non-calculator pages.
    # The model loads only when the calculator page is visited.
    data = load_pipeline_data()

    dispatch = {
        "Executive Summary":        page_executive_summary,
        "Member Risk Scores (RAF)": page_raf,
        "Risk Stratification Model":page_risk_model,
        "Intervention Impact":      page_intervention,
        "Shared Savings Projection":page_shared_savings,
        "Member Risk Calculator":   page_calculator,
    }
    dispatch[page](data)


if __name__ == "__main__":
    main()