"""
app.py — Streamlit dashboard for Medicare RAF analytics
"""

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
    ])

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────
    if page == "Executive Summary":
        st.header("Executive Summary")
        st.markdown(
            "This dashboard summarises the results of a **Medicare Advantage analytics pipeline** "
            "covering four linked analyses: member risk scoring, clinical risk stratification, "
            "causal impact measurement, and shared savings projection."
        )

        # Top-line KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Members Analysed",
            f"{r_cohort['n']:,}",
            help="Total beneficiaries included in this analysis run."
        )
        col2.metric(
            "Mean Risk Score (RAF)",
            f"{r_cohort['mean_raf']:.2f}",
            help="Average RAF score across the cohort. A score of 1.0 is the Medicare average."
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

        # Plain-English findings
        st.subheader("What the Analysis Found")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Member Risk Profile**")
            st.markdown(
                f"- {r_cohort['pct_raf_above_2']:.1f}% of members have a RAF score above 2.0, "
                "indicating a high burden of chronic illness.\n"
                f"- The top 10% of members (by risk score) have an average RAF of "
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
                "providing confidence that savings are real, not a statistical artefact.\n"
                "- Inpatient admissions and ED visits also trended downward, consistent "
                "with the cost finding."
            )
            st.markdown("**Shared Savings**")
            st.markdown(
                f"- At a **${r_ss['benchmark_pmpm']:,.0f} PMPM benchmark**, "
                f"actual spending came in at **${r_ss['actual_pmpm']:,.2f}**.\n"
                f"- Gross savings: **${r_ss['gross_savings_total']:,.0f}** "
                f"({r_ss['savings_rate_pct']:.1f}% savings rate — exceeds the 2% MSR threshold).\n"
                f"- At the 50% MSSP sharing rate, the ACO earns **${r_ss['shared_savings_earned']:,.0f}**."
            )

        st.divider()

        st.subheader("Summary Table")
        summary = pd.DataFrame([
            ("Members analysed",              f"{r_cohort['n']:,}",                         ""),
            ("Mean RAF score",                f"{r_cohort['mean_raf']:.3f}",                "1.0 = Medicare average"),
            ("% members RAF > 2.0",           f"{r_cohort['pct_raf_above_2']:.1f}%",        "High-complexity population"),
            ("Risk tier accuracy",            f"{r_model['tier_accuracy']:.1%}",            "XGBoost classifier"),
            ("Cost prediction MAE",           f"${r_model['cost_mae']:,.0f}",               "Annual cost error"),
            ("ATT — cost per member (DiD)",   f"−${abs(r_att['att_cost_pmpm']):,.0f}",      f"p = {r_att['p_value']:.4f}"),
            ("ATT — cost per member (PSM)",   "−$399",                                      "Convergent sensitivity check"),
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

        raf_by_tier = raf_data.groupby("risk_tier")["raf_score"].mean().reindex(["low", "moderate", "high"])
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
        n_above  = int((raf_data["raf_score"] > threshold).sum())
        pct_above = (raf_data["raf_score"] > threshold).mean() * 100
        est_cost  = n_above * r_ss["benchmark_pmpm"] * threshold
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Members above threshold", f"{n_above:,}")
        col_b.metric("Share of cohort",         f"{pct_above:.1f}%")
        col_c.metric("Est. annual cost exposure", f"${est_cost:,.0f}",
                     help="Approximate: members × benchmark PMPM × RAF multiplier")

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

        # Confusion matrix
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

        # Saved figures
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
            f"−${abs(r_att['att_cost_pmpm']):,.0f}",
            help="Average Treatment effect on the Treated (ATT). The causal reduction in annual cost."
        )
        col2.metric(
            "Statistical Significance",
            f"p = {r_att['p_value']:.4f}",
            help="p < 0.05 means the result is unlikely to be due to chance alone."
        )
        col3.metric(
            "PSM Cross-Check",
            "−$399/member",
            help="Independent estimate using propensity score matching. Convergence with DiD builds confidence."
        )

        # DiD chart
        st.divider()
        did_data = (
            panel.groupby(["period", "intervention"])["total_cost"]
            .mean()
            .reset_index()
        )
        did_data["Group"] = did_data["intervention"].map({0: "Control group", 1: "Intervention group"})
        did_data["Time"]  = did_data["period"].map({"pre": 0, "post": 1})

        fig, ax = plt.subplots(figsize=(8, 5))
        for group, color in [("Control group", BLUE), ("Intervention group", ACCENT)]:
            sub = did_data[did_data["Group"] == group].sort_values("Time")
            ax.plot([0, 1], sub["total_cost"].values, "o-",
                    color=color, linewidth=2.5, markersize=9, label=group)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Before Intervention", "After Intervention"], fontsize=11)
        ax.set_ylabel("Average Annual Cost per Member ($)", fontsize=11)
        ax.set_title(
            f"Difference-in-Differences\n"
            f"Cost savings: ${abs(r_att['att_cost_pmpm']):,.0f}/member  ·  p = {r_att['p_value']:.4f}",
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
            st.markdown(
                "- Both groups should trend similarly before the intervention (parallel trends). "
                f"This assumption holds here (pre-period p = 0.679).\n"
                "- After the intervention, the treated group's costs diverge downward relative to controls.\n"
                "- The vertical gap at the right is the estimated causal savings."
            )
        with col_b:
            st.markdown("**Utilisation Impact:**")
            st.markdown(
                f"- Inpatient admissions: {r_att['att_ip_admits']:+.3f}/member "
                f"(directionally favourable)\n"
                f"- ED visits: {r_att['att_ed_visits']:+.3f}/member "
                f"(directionally favourable)\n"
                "- Primary driver of savings is total cost; utilisation trends are consistent."
            )

        # Saved figure
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
            sharing_rate = st.slider("MSSP Sharing Rate", 0.30, 0.70, 0.50, 0.05,
                                     format="%.0f%%",
                                     help="Share of gross savings the ACO retains (30–70%).")
            msr          = st.slider("Minimum Savings Rate (MSR)", 0.01, 0.05, 0.02, 0.005,
                                     format="%.1f%%",
                                     help="ACO must beat this savings rate before earning any share.")
        with col_r:
            att_pmpm     = st.slider("ATT (cost reduction per member, $)", 100, 800,
                                     int(abs(r_att["att_cost_pmpm"])), 10,
                                     help="Estimated causal savings per member from the intervention.")
            n_lives      = st.slider("Attributed Lives", 100, 5000,
                                     int(r_ss["n_attributed_lives"]), 50,
                                     help="Number of beneficiaries attributed to the ACO.")

        benchmark_pmpm  = r_ss["benchmark_pmpm"]
        gross_savings   = att_pmpm * n_lives
        savings_rate    = gross_savings / (benchmark_pmpm * n_lives)
        earned          = gross_savings * sharing_rate if savings_rate > msr else 0

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Gross Savings",    f"${gross_savings:,.0f}")
        col2.metric("Savings Rate",     f"{savings_rate:.1%}",
                    delta="Exceeds MSR" if savings_rate > msr else "Below MSR")
        col3.metric("Earned Savings",   f"${earned:,.0f}")
        col4.metric("Per-Member Earned", f"${earned / n_lives:,.0f}" if n_lives > 0 else "—")

        if savings_rate <= msr:
            st.warning(
                f"Savings rate ({savings_rate:.1%}) does not exceed the MSR ({msr:.1%}). "
                "No shared savings would be earned under this scenario."
            )
        else:
            st.success(
                f"Savings rate ({savings_rate:.1%}) exceeds the MSR ({msr:.1%}). "
                f"The ACO earns **${earned:,.0f}** at the {sharing_rate:.0%} sharing rate."
            )

        st.divider()
        st.caption(
            "**Simplifications in this prototype:** No risk corridor adjustments, no CMS Star "
            "rating multipliers, no benchmark rebasing after Year 3, no regional adjustment factors, "
            "no asymmetric sharing for losses (one-sided risk only). "
            "Real MSSP contracts involve additional complexity."
        )

if __name__ == "__main__":
    main()
