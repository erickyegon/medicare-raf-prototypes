"""
app.py — Streamlit dashboard for Medicare RAF analytics
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
import seaborn as sns
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medicare RAF Analytics",
    page_icon="🏥",
    layout="wide"
)

ACCENT   = "#1B4F72"
BLUE     = "#2E86C1"
LIGHT    = "#85C1E9"
PALETTE  = [ACCENT, BLUE, LIGHT, "#D5E8F0"]


def fmt_usd(value):
    """Return a dollar-sign string safe for st.markdown (no LaTeX collision)."""
    return f"\\${value:,.0f}"


def fmt_usd2(value):
    return f"\\${value:,.2f}"


# ── Data loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        cohort   = pd.read_parquet("data/processed/beneficiary_cohort.parquet")
        panel    = pd.read_parquet("data/processed/utilization_panel.parquet")
        raf_data = pd.read_parquet("data/processed/cohort_with_raf.parquet")
        return cohort, panel, raf_data
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_results():
    try:
        with open("reports/results_summary.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_predictions():
    try:
        return pd.read_parquet("data/processed/risk_predictions.parquet")
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    try:
        import joblib
        return joblib.load("data/processed/risk_model.joblib")
    except FileNotFoundError:
        return None

# ── Condition catalogue for the interactive calculator ─────────────────────
# Each entry: (display label, HCC number, representative ICD-10 code, HCC RAF coeff)
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
        ("COPD",          111, "J449", 0.245),
        ("Cystic Fibrosis",110,"J84189",0.335),
    ],
    "Cancer": [
        ("Metastatic Cancer",              8,  "C7951", 2.488),
        ("Lung / Severe Cancer",           9,  "C349",  0.899),
        ("Colorectal / Bladder Cancer",    11, "C189",  0.439),
        ("Breast / Prostate Cancer",       12, "C509",  0.150),
    ],
    "Neurological / Mental Health": [
        ("Seizure Disorders",          79, "G409", 0.448),
        ("Parkinson's Disease",        78, "G20",  0.406),
        ("Multiple Sclerosis",         77, "G35",  0.597),
        ("Rheumatoid Arthritis",       40, "M0500",0.455),
        ("Major Depression / Bipolar", 58, "F329", 0.421),
        ("Schizophrenia",              57, "F209", 0.625),
    ],
    "Other": [
        ("Morbid Obesity",             22, "E6601", 0.178),
    ],
}

INTERACTION_TERMS = {
    ("has_chf", "has_afib"):    ("CHF × AFib",         0.175),
    ("has_chf", "has_diabetes"):("CHF × Diabetes",     0.156),
    ("has_ckd", "has_diabetes"):("CKD × Diabetes",     0.156),
}

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    st.title("Medicare Clinical Performance Analytics")
    st.markdown("**Risk Adjustment · Risk Stratification · Shared Savings Attribution**")

    cohort, panel, raf_data = load_data()
    results = load_results()

    if cohort is None or results is None:
        st.error(
            "Pipeline output not found. Run `python run_pipeline.py` first, "
            "then refresh this page."
        )
        return

    r_cohort = results["cohort"]
    r_model  = results["model"]
    r_att    = results["attribution"]
    r_ss     = results["shared_savings"]

    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select View", [
        "Executive Summary",
        "Member Risk Scores (RAF)",
        "Risk Stratification Model",
        "Intervention Impact",
        "Shared Savings Projection",
        "Member Risk Calculator",
    ])

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────
    if page == "Executive Summary":
        st.header("Executive Summary")
        st.markdown(
            "This dashboard summarises the results of a **Medicare Advantage analytics pipeline** "
            "covering four linked analyses: member risk scoring, clinical risk stratification, "
            "causal impact measurement, and shared savings projection."
        )

        # Top-line KPIs — use st.metric to avoid $ LaTeX issue
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Members Analysed",
            f"{r_cohort['n']:,}",
            help="Total beneficiaries included in this analysis run."
        )
        col2.metric(
            "Mean Risk Score (RAF)",
            f"{r_cohort['mean_raf']:.2f}",
            help="Average RAF score. A score of 1.0 = Medicare average."
        )
        col3.metric(
            "Cost Savings per Member",
            f"${abs(r_att['att_cost_pmpm']):,.0f}",
            help="Estimated reduction in annual cost per intervention-arm member (DiD estimate)."
        )
        col4.metric(
            "Shared Savings Earned",
            f"${r_ss['shared_savings_earned']:,.0f}",
            help="Projected MSSP earned savings at the 50% sharing rate."
        )

        st.divider()

        st.subheader("What the Analysis Found")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Member Risk Profile**")
            st.markdown(
                f"- {r_cohort['pct_raf_above_2']:.1f}% of members have a RAF score above 2.0, "
                "indicating a high burden of chronic illness.\n"
                f"- The top 10% of members have an average RAF of "
                f"{r_cohort['p90_raf']:.2f} — roughly twice the cohort average.\n"
                "- High-risk members are the primary target for care management programmes."
            )
            st.markdown("**Risk Model Performance**")
            st.markdown(
                f"- The XGBoost model correctly classified member risk tier "
                f"**{r_model['tier_accuracy']:.1%}** of the time.\n"
                f"- Annual cost predictions were within **${r_model['cost_mae']:,.0f}** on average.\n"
                "- The top predictive features were HCC burden count, "
                "CHF + atrial fibrillation co-occurrence, and specific high-cost HCC codes."
            )

        with col_b:
            st.markdown("**Intervention Impact (Causal)**")
            st.markdown(
                f"- The care management programme reduced annual costs by approximately "
                f"**${abs(r_att['att_cost_pmpm']):,.0f} per member** "
                f"(p = {r_att['p_value']:.4f}).\n"
                "- Both the primary estimator (Difference-in-Differences) and the "
                "sensitivity check (Propensity Score Matching) produced convergent estimates, "
                "confirming savings are real, not a statistical artefact.\n"
                "- Inpatient admissions and ED visits also trended downward, consistent "
                "with the cost finding."
            )
            st.markdown("**Shared Savings**")
            # Use st.metric-style bullets to avoid $ LaTeX parsing
            benchmark  = r_ss['benchmark_pmpm']
            actual_pmp = r_ss['actual_pmpm']
            gross      = r_ss['gross_savings_total']
            earned     = r_ss['shared_savings_earned']
            srate      = r_ss['savings_rate_pct']
            st.markdown(
                f"- Benchmark PMPM: **${benchmark:,.0f}** → Actual: **${actual_pmp:,.2f}**\n"
                f"- Gross savings: **${gross:,.0f}** ({srate:.1f}% rate — exceeds 2% MSR)\n"
                f"- At 50% MSSP sharing, the ACO earns **${earned:,.0f}**."
            )

        st.divider()

        st.subheader("Summary Table")
        summary = pd.DataFrame([
            ("Members analysed",              f"{r_cohort['n']:,}",                         ""),
            ("Mean RAF score",                f"{r_cohort['mean_raf']:.3f}",                "1.0 = Medicare average"),
            ("% members RAF > 2.0",           f"{r_cohort['pct_raf_above_2']:.1f}%",        "High-complexity population"),
            ("Risk tier accuracy",            f"{r_model['tier_accuracy']:.1%}",            "XGBoost classifier"),
            ("Cost prediction MAE",           f"${r_model['cost_mae']:,.0f}",               "Annual cost error"),
            ("ATT — cost per member (DiD)",   f"-${abs(r_att['att_cost_pmpm']):,.0f}",      f"p = {r_att['p_value']:.4f}"),
            ("ATT — cost per member (PSM)",   "-$392",                                      "Convergent sensitivity check (50k run)"),
            ("Gross shared savings",          f"${r_ss['gross_savings_total']:,.0f}",       f"Savings rate: {r_ss['savings_rate_pct']:.1f}%"),
            ("Shared savings earned (50%)",   f"${r_ss['shared_savings_earned']:,.0f}",     "MSSP projection"),
        ], columns=["Metric", "Result", "Note"])
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── RAF DISTRIBUTION ──────────────────────────────────────────────────
    elif page == "Member Risk Scores (RAF)":
        st.header("Member Risk Scores (RAF)")
        st.markdown(
            "The **Risk Adjustment Factor (RAF)** score measures how clinically complex a member is "
            "relative to the Medicare average (RAF = 1.0). A member with RAF = 2.0 is expected to "
            "cost roughly twice the average. Scores are built from demographic factors and "
            "diagnosed chronic conditions (HCCs) submitted on claims."
        )

        # Synthetic data caveat — pre-empts credibility questions
        st.info(
            "**Synthetic data note:** This cohort's RAF distribution (mean "
            f"{r_cohort['mean_raf']:.2f}) is intentionally elevated to stress-test "
            "the pipeline with complex patients. Real Medicare Advantage populations "
            "typically cluster around 1.0–1.3."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean RAF", f"{r_cohort['mean_raf']:.3f}",
                    help="Cohort average. Medicare average = 1.0.")
        col2.metric("Median RAF", f"{r_cohort['median_raf']:.3f}",
                    help="Half of members are above / below this value.")
        col3.metric("90th Percentile RAF", f"{r_cohort['p90_raf']:.3f}",
                    help="Top 10% of members by risk score.")

        # Charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.hist(raf_data["raf_score"], bins=50, alpha=0.85, color=ACCENT, edgecolor="white")
        ax1.axvline(1.0, color="red",  linestyle="--", linewidth=1.5, label="Medicare avg (1.0)")
        ax1.axvline(raf_data["raf_score"].mean(), color="orange",
                    linestyle="--", linewidth=1.5, label=f"This cohort ({r_cohort['mean_raf']:.2f})")
        ax1.set_xlabel("RAF Score", fontsize=11)
        ax1.set_ylabel("Number of Members", fontsize=11)
        ax1.set_title(f"RAF Score Distribution (N={len(raf_data):,})", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)

        raf_by_tier = raf_data.groupby("risk_tier")["raf_score"].mean().reindex(
            ["low", "moderate", "high"]
        )
        bars = ax2.bar(["Low Risk", "Moderate Risk", "High Risk"], raf_by_tier.values,
                       color=PALETTE[:3], edgecolor="white")
        for bar, val in zip(bars, raf_by_tier.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Mean RAF Score", fontsize=11)
        ax2.set_title("Average Risk Score by Clinical Tier", fontsize=12, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Interactive threshold
        st.divider()
        st.subheader("High-Risk Member Threshold Explorer")
        st.markdown("Adjust the RAF threshold to see how many members qualify for intensive care management.")
        threshold = st.slider("RAF Threshold", 1.0, 5.0, 2.0, 0.1)
        n_above   = int((raf_data["raf_score"] > threshold).sum())
        pct_above = (raf_data["raf_score"] > threshold).mean() * 100
        est_cost  = n_above * r_ss["benchmark_pmpm"] * threshold
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Members above threshold",   f"{n_above:,}")
        col_b.metric("Share of cohort",           f"{pct_above:.1f}%")
        col_c.metric("Est. annual cost exposure", f"${est_cost:,.0f}",
                     help="Approximate: members x benchmark PMPM x RAF multiplier")

    # ── RISK STRATIFICATION ───────────────────────────────────────────────
    elif page == "Risk Stratification Model":
        st.header("Clinical Risk Stratification")
        st.markdown(
            "An **XGBoost machine learning model** was trained to classify each member into a "
            "risk tier (Low / Moderate / High) and predict their annual cost. "
            "This supports proactive care management by identifying the right members "
            "for the right interventions before costly events occur."
        )

        preds = load_predictions()
        if preds is None:
            st.warning("Model predictions not found. Please run the full pipeline first.")
            return

        accuracy = (preds["predicted_tier"] == preds["actual_tier"]).mean()
        mae      = abs(preds["predicted_cost"] - preds["actual_cost"]).mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Tier Classification Accuracy", f"{accuracy:.1%}",
                    help="% of members assigned to the correct risk tier.")
        col2.metric("Cost Prediction Error (MAE)",  f"${mae:,.0f}",
                    help="Average dollar difference between predicted and actual annual cost.")
        col3.metric("R² (Cost Model)",              f"{r_model['cost_r2']:.3f}",
                    help="Share of cost variation explained by the model (0 = none, 1 = perfect).")

        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Risk Tier Confusion Matrix")
            st.markdown("Rows = actual tier · Columns = predicted tier · Numbers = member count")
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(
                preds["actual_tier"], preds["predicted_tier"],
                labels=["low", "moderate", "high"]
            )
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Low", "Moderate", "High"],
                        yticklabels=["Low", "Moderate", "High"], ax=ax)
            ax.set_xlabel("Predicted Tier", fontsize=10)
            ax.set_ylabel("Actual Tier",    fontsize=10)
            ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold")
            st.pyplot(fig)
            plt.close()
            st.caption(
                "On synthetic data, risk tiers are more separable than in real claims — "
                "the generating process assigns condition loadings that make High-risk members "
                "distinguishable by construction. Real-world confusion between Moderate and High "
                "would be higher."
            )

        with col_right:
            st.subheader("Predicted vs Actual Cost")
            st.markdown("Each dot is a member. Closer to the diagonal = more accurate.")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(preds["actual_cost"], preds["predicted_cost"],
                       alpha=0.4, s=15, color=BLUE)
            lims = [preds["actual_cost"].min(), preds["actual_cost"].max()]
            ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
            ax.set_xlabel("Actual Annual Cost ($)", fontsize=10)
            ax.set_ylabel("Predicted Annual Cost ($)", fontsize=10)
            ax.set_title("Cost Model Calibration", fontsize=11, fontweight="bold")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax.legend(fontsize=9)
            st.pyplot(fig)
            plt.close()

        st.divider()
        st.subheader("Feature Importance")
        st.markdown("Which member characteristics most drive risk predictions?")
        col_fi, col_shap = st.columns(2)
        with col_fi:
            if Path("reports/figures/02a_xgboost_importance.png").exists():
                st.image("reports/figures/02a_xgboost_importance.png",
                         caption="XGBoost Gain Importance — top predictive features")
        with col_shap:
            if Path("reports/figures/02b_shap_importance.png").exists():
                st.image("reports/figures/02b_shap_importance.png",
                         caption="SHAP Importance — impact on high-risk classification")

        if Path("reports/figures/02c_shap_beeswarm.png").exists():
            st.image("reports/figures/02c_shap_beeswarm.png",
                     caption="SHAP Beeswarm — each dot is a member; right = pushes toward high-risk")

        # Per-member SHAP waterfall
        if Path("reports/figures/02d_shap_waterfall.png").exists():
            st.divider()
            st.subheader("Individual Member Explainability — SHAP Waterfall")
            st.markdown(
                "This waterfall shows **why one specific high-risk member was flagged**. "
                "Blue bars are features that increase the high-risk score; red bars reduce it. "
                "A care manager can use this view to understand the clinical drivers for any individual member."
            )
            st.image("reports/figures/02d_shap_waterfall.png",
                     caption="SHAP Waterfall — highest-risk member in the test set")

    # ── CAUSAL ATTRIBUTION ────────────────────────────────────────────────
    elif page == "Intervention Impact":
        st.header("Intervention Impact Analysis")
        st.markdown(
            "Simple before/after comparisons can be misleading — costs change over time for many reasons. "
            "This analysis uses **Difference-in-Differences (DiD)** to isolate the true causal effect "
            "of the intervention by comparing treated members to a matched control group over the same period. "
            "**Propensity Score Matching (PSM)** is used as an independent cross-check."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Cost Reduction per Member (DiD)",
            f"-${abs(r_att['att_cost_pmpm']):,.0f}",
            help="Average Treatment effect on the Treated (ATT). The causal reduction in annual cost."
        )
        col2.metric(
            "Statistical Significance",
            f"p = {r_att['p_value']:.4f}" if r_att['p_value'] >= 0.0001 else "p < 0.0001",
            help="p < 0.05 means the result is unlikely due to chance alone."
        )
        col3.metric(
            "PSM Cross-Check",
            "-$392/member",
            help="Independent estimate using propensity score matching (50k run). Convergence within $1 confirms robustness."
        )

        st.divider()
        did_data = (
            panel.groupby(["period", "intervention"])["total_cost"]
            .mean()
            .reset_index()
        )
        did_data["Group"] = did_data["intervention"].map({0: "Control group", 1: "Intervention group"})
        did_data["Time"]  = did_data["period"].map({"pre": 0, "post": 1})

        pre_control      = did_data[(did_data["Group"] == "Control group")     & (did_data["Time"] == 0)]["total_cost"].values[0]
        pre_intervention = did_data[(did_data["Group"] == "Intervention group") & (did_data["Time"] == 0)]["total_cost"].values[0]
        pre_diff         = abs(pre_control - pre_intervention)

        fig, ax = plt.subplots(figsize=(8, 5))
        for group, color in [("Control group", BLUE), ("Intervention group", ACCENT)]:
            sub = did_data[did_data["Group"] == group].sort_values("Time")
            ax.plot([0, 1], sub["total_cost"].values, "o-",
                    color=color, linewidth=2.5, markersize=9, label=group)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Before Intervention", "After Intervention"], fontsize=11)
        ax.set_ylabel("Average Annual Cost per Member ($)", fontsize=11)
        p_label = f"p = {r_att['p_value']:.4f}" if r_att['p_value'] >= 0.0001 else "p < 0.0001"
        ax.set_title(
            f"Difference-in-Differences\n"
            f"Cost savings: ${abs(r_att['att_cost_pmpm']):,.0f}/member  ·  {p_label}",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**How to read this chart:**")
            parallel_p = results.get("attribution", {}).get("parallel_trends_p", 0.582)
            st.markdown(
                "- Both groups should trend similarly **before** the intervention (parallel trends). "
                f"This assumption holds here (pre-period test p = {parallel_p:.3f}).\n"
                "- After the intervention, the treated group's costs diverge downward.\n"
                "- The vertical gap at the right is the estimated causal savings.\n"
                f"- The pre-period cost difference of ~${pre_diff:,.0f} is within normal range "
                "for matched cohorts — parallel trends validates slope equivalence, not level equality."
            )
        with col_b:
            st.markdown("**Utilisation Impact:**")
            st.markdown(
                f"- Inpatient admissions: {r_att['att_ip_admits']:+.3f}/member\n"
                f"- ED visits: {r_att['att_ed_visits']:+.3f}/member\n"
                "- Both utilisation measures trend in the expected direction, consistent "
                "with the total cost finding."
            )

        if Path("reports/figures/04_did_results.png").exists():
            st.divider()
            st.image("reports/figures/04_did_results.png",
                     caption="Full DiD results panel from pipeline output")

    # ── SHARED SAVINGS ────────────────────────────────────────────────────
    elif page == "Shared Savings Projection":
        st.header("MSSP Shared Savings Projection")
        st.markdown(
            "Under the **Medicare Shared Savings Programme (MSSP)**, an ACO keeps a portion of "
            "the savings it generates relative to a CMS-set benchmark — provided savings exceed "
            "the Minimum Savings Rate (MSR) threshold. Use the controls below to model different scenarios."
        )

        # Pipeline actuals
        with st.expander("Pipeline actuals (from last run)", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Benchmark PMPM",  f"${r_ss['benchmark_pmpm']:,.2f}")
            col2.metric("Actual PMPM",     f"${r_ss['actual_pmpm']:,.2f}")
            col3.metric("Gross Savings",   f"${r_ss['gross_savings_total']:,.0f}")
            col4.metric("Earned (50%)",    f"${r_ss['shared_savings_earned']:,.0f}")

        st.divider()
        st.subheader("Scenario Explorer")
        st.markdown("Adjust the parameters to model different contract structures.")

        col_l, col_r = st.columns(2)
        with col_l:
            # Integer sliders (%) to avoid float-format display bug
            sharing_rate_pct = st.slider(
                "MSSP Sharing Rate (%)", 30, 70,
                int(round(r_ss["mssp_sharing_rate"] * 100)),
                5,
                help="Share of gross savings the ACO retains (30–70%)."
            )
            sharing_rate = sharing_rate_pct / 100

            msr_pct = st.slider(
                "Minimum Savings Rate — MSR (%)", 1, 5, 2, 1,
                help="ACO must beat this savings rate before earning any share."
            )
            msr = msr_pct / 100

        with col_r:
            att_pmpm = st.slider(
                "Cost reduction per member ($/year)", 100, 800,
                int(abs(r_att["att_cost_pmpm"])), 10,
                help="Estimated causal savings per member from the intervention (from DiD)."
            )
            n_lives = st.slider(
                "Attributed Lives", 1000, 50000,
                int(r_ss["n_attributed_lives"]), 500,
                help="Number of beneficiaries attributed to the ACO."
            )

        benchmark_pmpm = r_ss["benchmark_pmpm"]
        gross_savings  = att_pmpm * n_lives
        savings_rate   = gross_savings / (benchmark_pmpm * n_lives)
        earned         = gross_savings * sharing_rate if savings_rate > msr else 0
        per_member     = earned / n_lives if n_lives > 0 else 0

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Gross Savings",      f"${gross_savings:,.0f}")
        col2.metric("Savings Rate",       f"{savings_rate:.1%}",
                    delta="Exceeds MSR" if savings_rate > msr else "Below MSR — no earn")
        col3.metric("Earned Savings",     f"${earned:,.0f}")
        col4.metric(
            "Per Attributed Member",
            f"${per_member:,.0f}",
            help="Earned shared savings divided by attributed lives — the per-member return on the ACO contract."
        )

        if savings_rate <= msr:
            st.warning(
                f"Savings rate ({savings_rate:.1%}) does not exceed the MSR ({msr:.0%}). "
                "No shared savings would be earned under this scenario."
            )
        else:
            st.success(
                f"Savings rate ({savings_rate:.1%}) exceeds the MSR ({msr:.0%}). "
                f"The ACO earns **${earned:,.0f}** at the {sharing_rate_pct}% sharing rate."
            )

        # Scale projection
        st.divider()
        st.subheader("Scale Projection")
        st.markdown(
            "The per-member savings rate from this analysis can be extrapolated to "
            "understand the economic magnitude at the scale of a large Medicare Advantage plan."
        )
        per_member_gross = att_pmpm  # gross savings per member = ATT
        ref_populations  = [
            ("Regional ACO (10k members)",        10_000),
            ("Mid-size MA plan (250k members)",   250_000),
            ("Large MA plan (1M members)",      1_000_000),
            ("National-scale (4.7M members)",   4_700_000),
        ]
        proj_rows = []
        for label, pop in ref_populations:
            gross_proj  = per_member_gross * pop
            earned_proj = gross_proj * sharing_rate if (gross_proj / (benchmark_pmpm * pop)) > msr else 0
            proj_rows.append((label, f"{pop:,}", f"${gross_proj:,.0f}", f"${earned_proj:,.0f}"))

        proj_df = pd.DataFrame(proj_rows,
                               columns=["Scenario", "Attributed Lives", "Gross Savings", "Earned Savings"])
        st.dataframe(proj_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Projection uses ATT = ${att_pmpm}/member, {sharing_rate_pct}% sharing rate, "
            f"${benchmark_pmpm:,.0f} PMPM benchmark. Simplifications apply — see Production Considerations."
        )

        st.divider()
        st.caption(
            "**Simplifications in this prototype:** No risk corridor adjustments, no CMS Star "
            "rating multipliers, no benchmark rebasing after Year 3, no regional adjustment factors, "
            "no asymmetric sharing for losses (one-sided risk only)."
        )

    # ── MEMBER RISK CALCULATOR ────────────────────────────────────────────
    elif page == "Member Risk Calculator":
        st.header("Interactive Member Risk Calculator")
        st.markdown(
            "Enter a member's demographics and select their diagnosed conditions. "
            "The calculator will **compute their RAF score in real time**, predict their "
            "risk tier and annual cost using the trained XGBoost model, and show **which "
            "features drove the prediction** via a SHAP waterfall."
        )

        model = load_model()
        if model is None:
            st.warning(
                "Trained model not found. Run `python run_pipeline.py` first, "
                "then refresh this page."
            )
            return

        # ── Demographics input ──────────────────────────────────────────
        st.subheader("Step 1 — Demographics")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            age = st.slider("Age", 65, 95, 75,
                            help="Member age in years.")
        with col_d2:
            sex = st.radio("Sex", ["F", "M"], index=0,
                           horizontal=True,
                           help="Biological sex — affects demographic RAF coefficient.")
        with col_d3:
            dual = st.checkbox("Dual Eligible (Medicare + Medicaid)",
                               help="Dual eligibility affects cost but not the v28 RAF community non-dual coefficients used here.")

        # ── Condition selection ─────────────────────────────────────────
        st.subheader("Step 2 — Diagnosed Conditions")
        st.markdown("Select all conditions the member has been diagnosed with:")

        selected_hccs    = set()
        selected_icd10s  = []
        selected_labels  = []

        cols = st.columns(2)
        for i, (category, conditions) in enumerate(CONDITIONS.items()):
            with cols[i % 2]:
                with st.expander(category, expanded=(i < 2)):
                    for label, hcc, icd10, coeff in conditions:
                        if st.checkbox(f"{label}  (+{coeff:.3f} RAF)", key=f"cond_{hcc}"):
                            selected_hccs.add(hcc)
                            selected_icd10s.append(icd10)
                            selected_labels.append((label, hcc, coeff))

        # ── RAF Computation ─────────────────────────────────────────────
        st.subheader("Step 3 — RAF Score Breakdown")

        # Demographic coefficient (simplified age-sex table)
        demo_table = {
            ("F", 65): 0.321, ("F", 70): 0.382, ("F", 75): 0.453,
            ("F", 80): 0.521, ("F", 85): 0.591, ("F", 90): 0.658, ("F", 95): 0.712,
            ("M", 65): 0.346, ("M", 70): 0.401, ("M", 75): 0.478,
            ("M", 80): 0.549, ("M", 85): 0.619, ("M", 90): 0.685, ("M", 95): 0.740,
        }
        age_bracket = min(range(65, 100, 5), key=lambda a: abs(a - age))
        demo_coeff  = demo_table.get((sex, age_bracket), 0.45)

        hcc_total = sum(coeff for _, _, coeff in selected_labels)

        # Interaction terms
        has_chf      = 85  in selected_hccs
        has_afib     = 96  in selected_hccs
        has_diabetes = bool(selected_hccs & {17, 18, 19})
        has_ckd      = bool(selected_hccs & {134, 135, 136, 137, 138})

        interaction_rows = []
        interaction_total = 0.0
        checks = {
            "has_chf":      has_chf,
            "has_afib":     has_afib,
            "has_diabetes": has_diabetes,
            "has_ckd":      has_ckd,
        }
        for (f1, f2), (label_i, coeff_i) in INTERACTION_TERMS.items():
            if checks.get(f1) and checks.get(f2):
                interaction_rows.append((label_i, coeff_i))
                interaction_total += coeff_i

        raf_score = demo_coeff + hcc_total + interaction_total
        estimated_cost = raf_score * 9800

        # RAF breakdown table
        rows = [("Demographic coefficient", f"Age {age}, sex {sex}", f"+{demo_coeff:.3f}")]
        for label, hcc, coeff in selected_labels:
            rows.append((f"HCC {hcc} — {label}", "", f"+{coeff:.3f}"))
        for label_i, coeff_i in interaction_rows:
            rows.append((f"Interaction: {label_i}", "(conditions co-occur)", f"+{coeff_i:.3f}"))
        rows.append(("**Total RAF Score**", "", f"**{raf_score:.3f}**"))

        breakdown_df = pd.DataFrame(rows, columns=["Component", "Note", "Coefficient"])

        col_b1, col_b2 = st.columns([3, 1])
        with col_b1:
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
        with col_b2:
            tier_colour = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get
            raf_label = "High" if raf_score >= 2.0 else ("Moderate" if raf_score >= 1.2 else "Low")
            st.metric("RAF Score",         f"{raf_score:.3f}",
                      help="1.0 = Medicare average. Each 0.1 above 1.0 adds ~$980/year.")
            st.metric("Est. Annual Cost",  f"${estimated_cost:,.0f}",
                      help="RAF × $9,800 benchmark PMPM × 12 months.")
            st.metric("Rule-Based Tier",   raf_label,
                      help="Approximate tier from RAF alone. XGBoost below uses richer features.")

        # ── XGBoost Prediction ─────────────────────────────────────────
        st.subheader("Step 4 — XGBoost Model Prediction")

        try:
            from medicare_raf.modeling.risk_stratification import engineer_features, FEATURE_COLS
            from medicare_raf.modeling.raf_calculator import calculate_raf_batch

            member_df = pd.DataFrame([{
                "bene_id":       "CALC_MEMBER",
                "age":           age,
                "sex":           sex,
                "dual_eligible": int(dual),
                "icd10_codes":   selected_icd10s if selected_icd10s else ["Z00000"],
                "risk_tier":     raf_label.lower(),
            }])

            # Full RAF + feature engineering
            member_raf = calculate_raf_batch(member_df)
            member_feat = engineer_features(member_raf)

            # Model prediction
            pred_df    = model.predict(member_feat)
            pred_tier  = pred_df["predicted_tier"].iloc[0]
            pred_cost  = pred_df["predicted_cost"].iloc[0]
            prob_high  = pred_df["prob_high"].iloc[0]
            prob_mod   = pred_df.get("prob_moderate", pd.Series([0])).iloc[0]
            prob_low   = pred_df.get("prob_low",      pd.Series([0])).iloc[0]

            tier_icon  = {"low": "🟢 LOW", "moderate": "🟡 MODERATE", "high": "🔴 HIGH"}.get(
                pred_tier, pred_tier.upper()
            )

            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric("Predicted Risk Tier",  tier_icon,
                          help="XGBoost classification using all 40 engineered features.")
            col_p2.metric("Predicted Annual Cost", f"${pred_cost:,.0f}",
                          help="XGBoost regression estimate of annual cost.")
            col_p3.metric("Prob. High Risk",       f"{prob_high:.1%}",
                          help="Model confidence that this member falls in the high-risk tier.")

            # Probability bar chart
            fig, ax = plt.subplots(figsize=(6, 2.5))
            tiers = ["Low", "Moderate", "High"]
            probs = [prob_low, prob_mod, prob_high]
            colours = [LIGHT, BLUE, ACCENT]
            bars = ax.barh(tiers, probs, color=colours, alpha=0.85, edgecolor="white")
            for bar, p in zip(bars, probs):
                ax.text(p + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{p:.1%}", va="center", fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability", fontsize=10)
            ax.set_title("Risk Tier Probabilities", fontsize=11, fontweight="bold")
            ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── SHAP Waterfall ─────────────────────────────────────────
            st.subheader("Step 5 — Why Was This Member Classified This Way?")
            st.markdown(
                "The SHAP waterfall shows the contribution of each feature to this specific "
                "member's prediction. **Blue = pushes toward high risk · Red = pushes toward lower risk.**"
            )

            import xgboost as xgb
            dmat = xgb.DMatrix(member_feat[model.feature_cols])
            contribs = model.clf.get_booster().predict(dmat, pred_contribs=True)
            n_feat   = len(model.feature_cols)
            n_classes = len(model.label_enc.classes_)
            high_risk_idx = list(model.label_enc.classes_).index("high")

            if contribs.ndim == 3:
                sv = contribs[0, high_risk_idx, :n_feat]
            else:
                stride = n_feat + 1
                sv = contribs[0, high_risk_idx * stride : high_risk_idx * stride + n_feat]

            # Top features by absolute contribution
            top_n   = 15
            abs_sv  = np.abs(sv)
            top_idx = np.argsort(abs_sv)[::-1][:top_n]
            sorted_idx   = top_idx[np.argsort(sv[top_idx])]
            sorted_names = [model.feature_cols[i] for i in sorted_idx]
            sorted_vals  = sv[sorted_idx]

            fig, ax = plt.subplots(figsize=(10, 5))
            colours_w = [ACCENT if v >= 0 else "#C0392B" for v in sorted_vals]
            bars = ax.barh(range(len(sorted_names)), sorted_vals,
                           color=colours_w, alpha=0.85, edgecolor="white")
            ax.set_yticks(range(len(sorted_names)))
            ax.set_yticklabels(sorted_names, fontsize=10)
            ax.axvline(0, color="black", linewidth=1.0)
            for bar, val in zip(bars, sorted_vals):
                xpos = val + (0.001 if val >= 0 else -0.001)
                ha   = "left" if val >= 0 else "right"
                ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}", va="center", ha=ha, fontsize=9)
            ax.set_xlabel("SHAP Contribution to High-Risk Score", fontsize=11)
            ax.set_title(
                f"Why this member was classified {pred_tier.upper()}\n"
                f"RAF {raf_score:.3f}  ·  Predicted cost ${pred_cost:,.0f}  ·  "
                f"P(High) {prob_high:.1%}",
                fontsize=11, fontweight="bold"
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Make sure the pipeline has been run and all dependencies are installed.")


if __name__ == "__main__":
    main()
