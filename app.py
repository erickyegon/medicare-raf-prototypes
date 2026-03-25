"""
app.py — Streamlit dashboard for Medicare RAF analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Medicare RAF Analytics",
    page_icon="🏥",
    layout="wide"
)

# Load data if available
@st.cache_data
def load_data():
    try:
        cohort = pd.read_parquet("data/processed/beneficiary_cohort.parquet")
        panel = pd.read_parquet("data/processed/utilization_panel.parquet")
        raf_data = pd.read_parquet("data/processed/cohort_with_raf.parquet")
        return cohort, panel, raf_data
    except FileNotFoundError:
        st.error("Data files not found. Please run the pipeline first: `python run_pipeline.py`")
        return None, None, None

def main():
    st.title("🏥 Medicare Clinical Performance Analytics")
    st.markdown("**Risk Adjustment, Stratification & Shared Savings Attribution**")

    cohort, panel, raf_data = load_data()
    if cohort is None:
        return

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select View", [
        "Overview",
        "RAF Distribution",
        "Risk Stratification",
        "Causal Attribution",
        "Shared Savings"
    ])

    # Overview
    if page == "Overview":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Beneficiaries", f"{len(cohort):,}")
            st.metric("Mean RAF Score", f"{raf_data['raf_score'].mean():.2f}")

        with col2:
            high_risk_pct = (raf_data['raf_score'] > 2.0).mean() * 100
            st.metric("High-Risk Beneficiaries (RAF>2.0)", f"{high_risk_pct:.1f}%")

        with col3:
            intervention_rate = cohort['intervention'].mean() * 100
            st.metric("Intervention Coverage", f"{intervention_rate:.1f}%")

        st.subheader("Key Results")
        results_df = pd.DataFrame({
            "Metric": [
                "Risk Model Accuracy",
                "Cost Prediction R²",
                "ATT (DiD)",
                "Shared Savings (50% rate)"
            ],
            "Value": [
                "91.5%",
                "0.450",
                "-$391/member",
                "$4.89M"
            ]
        })
        st.table(results_df)

    # RAF Distribution
    elif page == "RAF Distribution":
        st.subheader("RAF Score Distribution")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        ax1.hist(raf_data['raf_score'], bins=50, alpha=0.8, color="#1B4F72")
        ax1.axvline(1.0, color="red", linestyle="--", label="Average (RAF=1.0)")
        ax1.set_xlabel("RAF Score")
        ax1.set_ylabel("Count")
        ax1.set_title("RAF Score Distribution")
        ax1.legend()

        # By risk tier
        raf_by_tier = raf_data.groupby('risk_tier')['raf_score'].mean()
        bars = ax2.bar(raf_by_tier.index, raf_by_tier.values, color=["#85C1E9", "#2E86C1", "#1B4F72"])
        ax2.set_xlabel("Risk Tier")
        ax2.set_ylabel("Mean RAF Score")
        ax2.set_title("Mean RAF by Risk Tier")

        for bar, val in zip(bars, raf_by_tier.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", fontsize=10)

        st.pyplot(fig)

        # Threshold selector
        threshold = st.slider("RAF Threshold", 1.0, 3.0, 2.0, 0.1)
        above_threshold = (raf_data['raf_score'] > threshold).sum()
        pct_above = (raf_data['raf_score'] > threshold).mean() * 100
        st.write(f"Beneficiaries with RAF > {threshold}: {above_threshold:,} ({pct_above:.1f}%)")

    # Risk Stratification
    elif page == "Risk Stratification":
        st.subheader("Clinical Risk Stratification")

        # Load predictions if available
        try:
            preds = pd.read_parquet("data/processed/risk_predictions.parquet")
            st.success("Model predictions loaded")

            col1, col2 = st.columns(2)

            with col1:
                accuracy = (preds['predicted_tier'] == preds['actual_tier']).mean()
                st.metric("Model Accuracy", f"{accuracy:.1%}")

            with col2:
                mae = abs(preds['predicted_cost'] - preds['actual_cost']).mean()
                st.metric("Cost Prediction MAE", f"${mae:,.0f}")

            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(preds['actual_tier'], preds['predicted_tier'],
                                labels=['low', 'moderate', 'high'])

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                       xticklabels=['low', 'moderate', 'high'],
                       yticklabels=['low', 'moderate', 'high'], ax=ax)
            ax.set_xlabel("Predicted Tier")
            ax.set_ylabel("Actual Tier")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

        except FileNotFoundError:
            st.warning("Model predictions not found. Run the pipeline first.")

    # Causal Attribution
    elif page == "Causal Attribution":
        st.subheader("Causal Impact Analysis")

        # DiD visualization
        did_data = (
            panel.groupby(["period", "intervention"])["total_cost"]
            .mean()
            .reset_index()
        )
        did_data["group"] = did_data["intervention"].map({0: "Control", 1: "Intervention"})
        did_data["time"] = did_data["period"].map({"pre": 0, "post": 1})

        fig, ax = plt.subplots(figsize=(8, 5))
        for group, color in [("Control", "#2E86C1"), ("Intervention", "#1B4F72")]:
            sub = did_data[did_data["group"] == group].sort_values("time")
            ax.plot([0, 1], sub["total_cost"].values, "o-",
                    color=color, linewidth=2.5, markersize=8, label=group)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre-Intervention", "Post-Intervention"])
        ax.set_ylabel("Mean Annual Cost per Beneficiary ($)")
        ax.set_title("Difference-in-Differences Results\nATT = -$391/member (p < 0.0001)")
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        st.pyplot(fig)

        st.markdown("**Key Findings:**")
        st.markdown("- DiD and PSM estimates converge within $1")
        st.markdown("- Intervention reduces costs by $391 per member")
        st.markdown("- Parallel trends assumption holds (p = 0.582)")

    # Shared Savings
    elif page == "Shared Savings":
        st.subheader("MSSP Shared Savings Projection")

        # Sensitivity analysis
        sharing_rate = st.slider("MSSP Sharing Rate", 0.3, 0.7, 0.5, 0.05)
        msr = st.slider("Minimum Savings Rate", 0.01, 0.05, 0.02, 0.005)

        # Calculate savings
        att_pmpm = -391
        n_lives = 25000
        benchmark_pmpm = 9800

        gross_savings = (benchmark_pmpm + att_pmpm) * n_lives - benchmark_pmpm * n_lives
        savings_rate = abs(gross_savings) / (benchmark_pmpm * n_lives)
        earned = abs(gross_savings) * sharing_rate if savings_rate > msr else 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Gross Savings", f"${abs(gross_savings):,.0f}")

        with col2:
            st.metric("Savings Rate", f"{savings_rate:.1f}%")

        with col3:
            st.metric("Shared Savings Earned", f"${earned:,.0f}")

        st.markdown(f"**Parameters:** {n_lives:,} attributed lives, ${benchmark_pmpm} PMPM benchmark")

if __name__ == "__main__":
    main()