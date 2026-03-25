"""
run_pipeline.py
---------------
End-to-end Medicare RAF + risk stratification + shared savings pipeline.

Usage:
    python run_pipeline.py

Stages:
    1. Generate synthetic 50,000-beneficiary CMS-style claims cohort
    2. Calculate HCC/RAF scores for all beneficiaries
    3. Train XGBoost risk stratification model
    4. Run causal attribution (DiD + PSM) for shared savings
    5. Export results to reports/
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from medicare_raf.data.data_generator import generate_beneficiary_cohort, generate_utilization_panel
from medicare_raf.modeling.raf_calculator import calculate_raf_batch, summarise_cohort_raf
from medicare_raf.modeling.risk_stratification import engineer_features, train_and_evaluate, FEATURE_COLS
from medicare_raf.inference.causal_attribution import run_full_attribution

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
ACCENT = "#1B4F72"
PALETTE = ["#1B4F72", "#2E86C1", "#85C1E9", "#D5E8F0"]


def stage_banner(n: int, title: str):
    print(f"\n{'='*60}")
    print(f"  STAGE {n}: {title}")
    print(f"{'='*60}")


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)

    t0 = time.time()

    # ── STAGE 1: DATA GENERATION ─────────────────────────────────
    stage_banner(1, "SYNTHETIC DATA GENERATION")
    # N_BENE env var lets Streamlit Cloud run a fast 1,000-member demo
    n_bene = int(os.environ.get("N_BENE", 50_000))
    print(f"Generating {n_bene:,}-beneficiary Medicare cohort...")
    cohort = generate_beneficiary_cohort(n=n_bene)
    panel  = generate_utilization_panel(cohort, intervention_effect_pmpm=-420.0)

    cohort.to_parquet("data/processed/beneficiary_cohort.parquet", index=False)
    panel.to_parquet("data/processed/utilization_panel.parquet", index=False)
    print(f"  Cohort: {cohort.shape[0]:,} beneficiaries")
    print(f"  Panel:  {panel.shape[0]:,} bene-year records (2 years)")
    print(f"  Intervention arm: {cohort['intervention'].sum():,} ({cohort['intervention'].mean():.1%})")
    print(f"  Risk tier: {cohort['risk_tier'].value_counts().to_dict()}")

    # ── STAGE 2: RAF SCORING ──────────────────────────────────────
    stage_banner(2, "HCC / RAF SCORING")
    print(f"Calculating RAF scores for all {len(cohort):,} beneficiaries...")
    cohort_raf = calculate_raf_batch(cohort)

    raf_summary = summarise_cohort_raf(cohort_raf)
    print(f"  Mean RAF:           {raf_summary['mean_raf']:.3f}")
    print(f"  Median RAF:         {raf_summary['median_raf']:.3f}")
    print(f"  P90 RAF:            {raf_summary['p90_raf']:.3f}")
    print(f"  % RAF > 2.0:        {raf_summary['pct_raf_above_2']:.1f}%")
    print(f"  Estimated total cost: ${raf_summary['estimated_total_cost']:,.0f}")

    # Serialise complex columns for parquet
    cohort_raf["hccs"] = cohort_raf["hccs"].apply(lambda x: str(x) if isinstance(x, (list, set)) else x)
    cohort_raf["hcc_labels"] = cohort_raf["hcc_labels"].apply(lambda x: str(x) if isinstance(x, list) else x)
    cohort_raf["hcc_details"] = cohort_raf["hcc_details"].apply(lambda x: str(x) if isinstance(x, dict) else x)
    cohort_raf["icd10_codes"] = cohort_raf["icd10_codes"].apply(lambda x: str(x) if isinstance(x, list) else x)
    cohort_raf.to_parquet("data/processed/cohort_with_raf.parquet", index=False)

    # Plot 1: RAF distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(cohort_raf["raf_score"], bins=60, color=ACCENT, alpha=0.8, edgecolor="white")
    axes[0].axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="Average (RAF=1.0)")
    axes[0].set_xlabel("RAF Score", fontsize=11)
    axes[0].set_ylabel("Beneficiary Count", fontsize=11)
    axes[0].set_title(f"RAF Score Distribution (N={len(cohort):,})", fontsize=12, fontweight="bold")
    axes[0].legend()

    raf_by_tier = cohort_raf.groupby("risk_tier")["raf_score"].mean().reindex(
        ["low", "moderate", "high"]
    )
    bars = axes[1].bar(raf_by_tier.index, raf_by_tier.values, color=PALETTE[:3], edgecolor="white")
    for bar, val in zip(bars, raf_by_tier.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", fontsize=10)
    axes[1].set_xlabel("Risk Tier", fontsize=11)
    axes[1].set_ylabel("Mean RAF Score", fontsize=11)
    axes[1].set_title("Mean RAF Score by Risk Tier", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/01_raf_distribution.png", dpi=150, bbox_inches="tight")
    plt.savefig("docs/figures/01_raf_distribution.png",    dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Figure saved: reports/figures/01_raf_distribution.png")

    # ── STAGE 3: RISK STRATIFICATION ─────────────────────────────
    stage_banner(3, "RISK STRATIFICATION MODEL (XGBoost)")
    results = train_and_evaluate(cohort_raf, panel)
    metrics = results["metrics"]
    fi      = results["feature_importance"]
    shap_fi = results["shap_importance"]
    shap_values = results["shap_values"]
    X_test  = results["X_test"]
    preds   = results["test_predictions"]

    preds.to_parquet("data/processed/risk_predictions.parquet", index=False)

    import joblib
    joblib.dump(results["model"], "data/processed/risk_model.joblib")
    print("  → Model saved: data/processed/risk_model.joblib")

    # Plot 2: Feature importance (XGBoost gain)
    fig, ax = plt.subplots(figsize=(9, 5))
    fi_top = fi.head(12)
    ax.barh(fi_top["feature"][::-1], fi_top["importance"][::-1], color=ACCENT, alpha=0.85)
    ax.set_xlabel("Feature Importance (XGBoost)", fontsize=11)
    ax.set_title("Top 12 Features — XGBoost Gain Importance", fontsize=12, fontweight="bold")
    ax.axvline(fi_top["importance"].mean(), color="red", linestyle="--",
               linewidth=1, alpha=0.7, label="Mean importance")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("reports/figures/02a_xgboost_importance.png", dpi=150, bbox_inches="tight")
    plt.savefig("docs/figures/02a_xgboost_importance.png",    dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Figure saved: reports/figures/02a_xgboost_importance.png")

    # Plot 2b: SHAP feature importance
    fig, ax = plt.subplots(figsize=(9, 5))
    shap_top = shap_fi.head(12)
    ax.barh(shap_top["feature"][::-1], shap_top["shap_importance"][::-1], color="#2E86C1", alpha=0.85)
    ax.set_xlabel("Feature Importance (SHAP)", fontsize=11)
    ax.set_title("Top 12 Features — SHAP Importance", fontsize=12, fontweight="bold")
    ax.axvline(shap_top["shap_importance"].mean(), color="red", linestyle="--",
               linewidth=1, alpha=0.7, label="Mean importance")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("reports/figures/02b_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.savefig("docs/figures/02b_shap_importance.png",    dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Figure saved: reports/figures/02b_shap_importance.png")

    # Plot 2c: SHAP beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap.Explanation(values=shap_values, 
                                         data=X_test[results["model"].feature_cols].values,
                                         feature_names=results["model"].feature_cols),
                        max_display=12, show=False)
    plt.title("SHAP Beeswarm Plot — High-Risk Prediction", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/figures/02c_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.savefig("docs/figures/02c_shap_beeswarm.png",    dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Figure saved: reports/figures/02c_shap_beeswarm.png")

    # Plot 2d: Per-member SHAP waterfall — highest-risk member in test set
    feature_cols = results["model"].feature_cols
    high_risk_idx_list = preds.index[preds["actual_tier"] == "high"].tolist()
    if high_risk_idx_list:
        top_member_pos = preds.loc[high_risk_idx_list, "predicted_cost"].idxmax()
        member_pos = preds.index.get_loc(top_member_pos)
        sv = shap_values[member_pos]                        # (n_features,)
        feat_vals = X_test[feature_cols].iloc[member_pos]

        top_n = 12
        abs_sv   = np.abs(sv)
        top_idx  = np.argsort(abs_sv)[::-1][:top_n]
        # Sort by contribution value for waterfall readability
        sorted_idx   = top_idx[np.argsort(sv[top_idx])]
        sorted_names = [feature_cols[i] for i in sorted_idx]
        sorted_vals  = sv[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [ACCENT if v >= 0 else "#C0392B" for v in sorted_vals]
        bars = ax.barh(range(len(sorted_names)), sorted_vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=10)
        ax.axvline(0, color="black", linewidth=1.0)
        for bar, val in zip(bars, sorted_vals):
            xpos = val + (0.001 if val >= 0 else -0.001)
            ha   = "left" if val >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=9)
        raf_val = feat_vals.get("raf_score", float("nan"))
        ax.set_xlabel("SHAP Contribution to High-Risk Score", fontsize=11)
        ax.set_title(
            f"SHAP Waterfall — Example High-Risk Member  |  RAF {raf_val:.3f}\n"
            "Blue bars push toward HIGH risk · Red bars push toward lower risk",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig("reports/figures/02d_shap_waterfall.png", dpi=150, bbox_inches="tight")
        plt.savefig("docs/figures/02d_shap_waterfall.png",    dpi=150, bbox_inches="tight")
        plt.close()
        print("  → Figure saved: reports/figures/02d_shap_waterfall.png")

    # Plot 3: Predicted vs Actual cost
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = preds.sample(min(3000, len(preds)), random_state=42)
    ax.scatter(sample["actual_cost"], sample["predicted_cost"],
               alpha=0.3, s=8, color=ACCENT)
    lims = [0, max(sample["actual_cost"].max(), sample["predicted_cost"].max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Annual Cost ($)", fontsize=11)
    ax.set_ylabel("Predicted Annual Cost ($)", fontsize=11)
    ax.set_title(f"Predicted vs Actual Cost  |  R²={metrics['cost_r2']:.3f}  MAE=${metrics['cost_mae']:,.0f}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("reports/figures/03_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.savefig("docs/figures/03_predicted_vs_actual.png",    dpi=150, bbox_inches="tight")
    plt.close()
    print("  → Figure saved: reports/figures/03_predicted_vs_actual.png")

    # ── STAGE 4: CAUSAL ATTRIBUTION ──────────────────────────────
    stage_banner(4, "CAUSAL ATTRIBUTION (DiD + PSM)")

    # Merge panel with features
    panel_features = panel.merge(
        cohort[["bene_id", "dual_eligible", "risk_tier"]],
        on="bene_id",
        how="left",
    )
    attribution = run_full_attribution(panel_features)

    # Plot 4: DiD visualisation
    did_data = (
        panel_features.groupby(["period", "intervention"])["total_cost"]
        .mean()
        .reset_index()
    )
    did_data["group"] = did_data["intervention"].map({0: "Control", 1: "Intervention"})
    did_data["time"]  = did_data["period"].map({"pre": 0, "post": 1})

    fig, ax = plt.subplots(figsize=(7, 5))
    for group, color in [("Control", "#2E86C1"), ("Intervention", ACCENT)]:
        sub = did_data[did_data["group"] == group].sort_values("time")
        ax.plot([0, 1], sub["total_cost"].values, "o-",
                color=color, linewidth=2.5, markersize=8, label=group)
        for _, r in sub.iterrows():
            ax.annotate(f"${r['total_cost']:,.0f}",
                        xy=(r["time"], r["total_cost"]),
                        xytext=(8, 4), textcoords="offset points",
                        fontsize=9, color=color)

    att = attribution["did_cost"]["att"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre-Intervention", "Post-Intervention"], fontsize=11)
    ax.set_ylabel("Mean Annual Cost per Beneficiary ($)", fontsize=11)
    ax.set_title(
        f"Difference-in-Differences\nATT = ${att:,.0f}/member  "
        f"(p={attribution['did_cost']['p_value']:.4f})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    plt.savefig("reports/figures/04_did_results.png", dpi=150, bbox_inches="tight")
    plt.savefig("docs/figures/04_did_results.png",    dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  → Figure saved: reports/figures/04_did_results.png")

    # ── SUMMARY REPORT ────────────────────────────────────────────
    stage_banner(5, "SUMMARY REPORT")
    savings = attribution["savings"]
    elapsed = round(time.time() - t0, 1)

    print(f"""
┌─────────────────────────────────────────────────────────┐
│         MEDICARE RAF + SHARED SAVINGS PIPELINE           │
│                     RESULTS SUMMARY                      │
├─────────────────────────────────────────────────────────┤
│  COHORT                                                  │
│  Beneficiaries:          50,000                          │
│  Intervention arm:       {cohort['intervention'].sum():>6,}                         │
│  Mean RAF score:         {raf_summary['mean_raf']:.3f}                          │
│  % High-risk (RAF>2):    {raf_summary['pct_raf_above_2']:.1f}%                         │
├─────────────────────────────────────────────────────────┤
│  RISK MODEL (XGBoost)                                    │
│  Tier accuracy:          {metrics['tier_accuracy']:.1%}                         │
│  Cost MAE:               ${metrics['cost_mae']:>7,.0f}                        │
│  Cost R²:                {metrics['cost_r2']:.3f}                          │
├─────────────────────────────────────────────────────────┤
│  CAUSAL ATTRIBUTION (DiD)                                │
│  ATT — total cost:       ${attribution['did_cost']['att']:>7,.0f}/member            │
│  p-value:                {attribution['did_cost']['p_value']:.4f}                          │
│  ATT — IP admits:        {attribution['did_ip']['att']:>+.3f}/member             │
│  ATT — ED visits:        {attribution['did_ed']['att']:>+.3f}/member             │
├─────────────────────────────────────────────────────────┤
│  SHARED SAVINGS (MSSP PROJECTION)                        │
│  Attributed lives:       {savings['n_attributed_lives']:>6,}                         │
│  Gross savings:          ${savings['gross_savings_total']:>10,.0f}                  │
│  Savings rate:           {savings['savings_rate_pct']:.1f}%                          │
│  Shared savings earned:  ${savings['shared_savings_earned']:>10,.0f}                  │
├─────────────────────────────────────────────────────────┤
│  Runtime: {elapsed}s                                        │
└─────────────────────────────────────────────────────────┘
    """)

    # Save summary JSON
    import json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            return super().default(obj)
    summary = {
        "cohort":       raf_summary,
        "model":        metrics,
        "attribution": {
            "att_cost_pmpm": attribution["did_cost"]["att"],
            "p_value":       attribution["did_cost"]["p_value"],
            "att_ip_admits": attribution["did_ip"]["att"],
            "att_ed_visits": attribution["did_ed"]["att"],
        },
        "shared_savings": savings,
    }
    with open("reports/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print("  → Full results: reports/results_summary.json")
    print("  → Figures:      reports/figures/")

    return summary


if __name__ == "__main__":
    main()
