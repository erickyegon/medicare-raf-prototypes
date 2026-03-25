"""
causal_attribution.py — fast vectorized PSM via numpy
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def difference_in_differences(panel, outcome="total_cost", covariates=None):
    df = panel.copy()
    df["post"]       = (df["year"] == 1).astype(int)
    df["treat"]      = df["intervention"]
    df["post_treat"] = df["post"] * df["treat"]
    cov_str = (" + " + " + ".join(covariates)) if covariates else ""
    formula = f"{outcome} ~ post + treat + post_treat{cov_str}"
    try:
        model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["bene_id"]})
    except Exception:
        model = smf.ols(formula, data=df).fit()
    att    = model.params["post_treat"]
    ci     = model.conf_int().loc["post_treat"]
    p_val  = model.pvalues["post_treat"]
    se     = model.bse["post_treat"]
    df_pre = df[df["year"] == 0]
    pt_p   = smf.ols(f"{outcome} ~ treat", data=df_pre).fit().pvalues.get("treat", np.nan)
    return {
        "method": "DiD (TWFE)", "outcome": outcome,
        "att": round(float(att), 2), "att_se": round(float(se), 2),
        "ci_95_low": round(float(ci[0]), 2), "ci_95_high": round(float(ci[1]), 2),
        "p_value": round(float(p_val), 4), "significant": bool(p_val < 0.05),
        "n_treated": int((df["treat"]==1).sum()//2), "n_control": int((df["treat"]==0).sum()//2),
        "parallel_trends_p": round(float(pt_p), 4),
        "model_summary": model.summary().as_text(),
    }


def propensity_score_matching(panel, outcome="total_cost", caliper=0.05):
    print("   [PSM] Estimating propensity scores...")
    pre = panel[panel["year"] == 0].copy().reset_index(drop=True)
    pre["age_scaled"]    = (pre["age"] - 72) / 10
    pre["risk_high"]     = (pre["risk_tier"] == "high").astype(int) if "risk_tier" in pre.columns else 0
    pre["risk_moderate"] = (pre["risk_tier"] == "moderate").astype(int) if "risk_tier" in pre.columns else 0
    ps_feats = [f for f in ["age_scaled","risk_high","risk_moderate","dual_eligible","ip_admits","ed_visits"] if f in pre.columns]
    X  = StandardScaler().fit_transform(pre[ps_feats].fillna(0).values)
    lr = LogisticRegression(max_iter=500, random_state=42).fit(X, pre["intervention"].values)
    pre["ps"] = lr.predict_proba(X)[:, 1]

    treated = pre[pre["intervention"] == 1].reset_index(drop=True)
    control = pre[pre["intervention"] == 0].reset_index(drop=True)
    t_ps    = treated["ps"].values
    c_ps    = control["ps"].values

    print(f"   [PSM] Matching {len(treated):,} treated to {len(control):,} controls (vectorized)...")

    # Vectorized greedy 1:1 NN matching with caliper
    used      = np.zeros(len(c_ps), dtype=bool)
    t_matched = []
    c_matched = []

    # Sort treated by PS to improve greedy quality
    t_order = np.argsort(t_ps)
    for ti in t_order:
        ps_t    = t_ps[ti]
        dists   = np.abs(c_ps - ps_t)
        dists[used] = np.inf
        best    = np.argmin(dists)
        if dists[best] <= caliper:
            t_matched.append(treated["bene_id"].iloc[ti])
            c_matched.append(control["bene_id"].iloc[best])
            used[best] = True

    n_pairs = len(t_matched)
    print(f"   [PSM] Matched pairs: {n_pairs:,}")

    if n_pairs < 50:
        return {"method": "PSM", "outcome": outcome, "error": f"Only {n_pairs} matches. Try larger caliper."}

    post_out  = panel[panel["year"]==1].set_index("bene_id")[outcome]
    pre_out   = panel[panel["year"]==0].set_index("bene_id")[outcome]
    t_post    = post_out.reindex(t_matched).values
    c_post    = post_out.reindex(c_matched).values
    t_pre_v   = pre_out.reindex(t_matched).values
    c_pre_v   = pre_out.reindex(c_matched).values
    diffs     = (t_post - t_pre_v) - (c_post - c_pre_v)
    att       = float(np.nanmean(diffs))
    se        = float(np.nanstd(diffs) / np.sqrt(np.sum(~np.isnan(diffs))))
    p_val     = float(2*(1 - stats.t.cdf(abs(att/se), df=n_pairs-1)))
    ctrl_m    = control[control["bene_id"].isin(c_matched)]
    smd_age   = abs(treated["age"].mean() - ctrl_m["age"].mean()) / treated["age"].std()
    return {
        "method": "PSM (1:1 NN vectorized)", "outcome": outcome,
        "att": round(att, 2), "att_se": round(se, 2),
        "ci_95_low": round(att - 1.96*se, 2), "ci_95_high": round(att + 1.96*se, 2),
        "p_value": round(p_val, 4), "significant": bool(p_val < 0.05),
        "n_matched_pairs": n_pairs, "caliper": caliper,
        "smd_age_post_match": round(float(smd_age), 3),
        "matched_treated_ids": t_matched,
        "matched_control_ids": c_matched,
        "treated_pre": treated,
        "control_pre": control,
        "treated_matched": treated[treated["bene_id"].isin(t_matched)],
        "control_matched": control[control["bene_id"].isin(c_matched)],
    }


def generate_love_plot(treated_pre: pd.DataFrame, control_pre: pd.DataFrame,
                      treated_matched: pd.DataFrame, control_matched: pd.DataFrame,
                      covariates: list, save_path: str = None):
    """
    Generate Love Plot showing covariate balance before and after matching.

    Parameters
    ----------
    treated_pre : pd.DataFrame
        Treated group before matching
    control_pre : pd.DataFrame
        Control group before matching
    treated_matched : pd.DataFrame
        Treated group after matching
    control_matched : pd.DataFrame
        Control group after matching
    covariates : list
        List of covariate column names
    save_path : str, optional
        Path to save the plot
    """
    def standardized_mean_difference(group1, group2, var):
        """Calculate SMD for a variable."""
        mean1, mean2 = group1[var].mean(), group2[var].mean()
        std1, std2 = group1[var].std(), group2[var].std()
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    balance_data = []

    for cov in covariates:
        if cov in treated_pre.columns and cov in control_pre.columns:
            # Before matching
            smd_before = standardized_mean_difference(treated_pre, control_pre, cov)

            # After matching
            smd_after = standardized_mean_difference(treated_matched, control_matched, cov)

            balance_data.append({
                'covariate': cov,
                'smd_before': abs(smd_before),
                'smd_after': abs(smd_after)
            })

    balance_df = pd.DataFrame(balance_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(balance_df))
    ax.barh(y_pos - 0.2, balance_df['smd_before'], 0.4, label='Before Matching',
            color='#2E86C1', alpha=0.7)
    ax.barh(y_pos + 0.2, balance_df['smd_after'], 0.4, label='After Matching',
            color='#1B4F72', alpha=0.7)

    ax.axvline(0.1, color='red', linestyle='--', linewidth=1, label='SMD = 0.1 (good balance)')
    ax.axvline(0.2, color='orange', linestyle='--', linewidth=1, label='SMD = 0.2 (acceptable)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_df['covariate'])
    ax.set_xlabel('Absolute Standardized Mean Difference')
    ax.set_title('Covariate Balance: Love Plot (Before vs After Propensity Score Matching)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return balance_df


def project_shared_savings(att_pmpm, n_attributed_lives, benchmark_pmpm=9800.0,
                            mssp_sharing_rate=0.50, minimum_savings_rate=0.02):
    total_actual    = (benchmark_pmpm + att_pmpm) * n_attributed_lives
    total_benchmark = benchmark_pmpm * n_attributed_lives
    gross_savings   = total_benchmark - total_actual
    savings_rate    = gross_savings / total_benchmark
    qualifies       = bool(gross_savings > minimum_savings_rate * total_benchmark)
    shared_savings  = gross_savings * mssp_sharing_rate if qualifies else 0.0
    return {
        "n_attributed_lives": int(n_attributed_lives),
        "benchmark_pmpm": round(float(benchmark_pmpm), 2),
        "actual_pmpm": round(float(benchmark_pmpm + att_pmpm), 2),
        "att_pmpm": round(float(att_pmpm), 2),
        "gross_savings_total": round(float(gross_savings), 0),
        "savings_rate_pct": round(float(savings_rate * 100), 2),
        "exceeds_msr": qualifies,
        "shared_savings_earned": round(float(shared_savings), 0),
        "mssp_sharing_rate": mssp_sharing_rate,
    }


def run_full_attribution(panel, n_attributed_lives=None, benchmark_pmpm=9800.0):
    print("=" * 60)
    print("CAUSAL ATTRIBUTION ANALYSIS")
    print("=" * 60)
    if n_attributed_lives is None:
        n_attributed_lives = panel[panel["intervention"]==1]["bene_id"].nunique()
    available_covs = [c for c in ["age","dual_eligible"] if c in panel.columns]

    print("\n1. Difference-in-Differences (Primary Estimator)")
    print("-" * 40)
    did_cost = difference_in_differences(panel, "total_cost", available_covs)
    print(f"   ATT (total cost): ${did_cost['att']:,.2f}/member (p={did_cost['p_value']:.4f})")
    print(f"   95% CI: [${did_cost['ci_95_low']:,.2f}, ${did_cost['ci_95_high']:,.2f}]")
    print(f"   Significant: {did_cost['significant']}")
    print(f"   Parallel trends p-value: {did_cost['parallel_trends_p']:.3f}")
    did_ip = difference_in_differences(panel, "ip_admits")
    print(f"\n   ATT (IP admits): {did_ip['att']:.3f}/member (p={did_ip['p_value']:.4f})")
    did_ed = difference_in_differences(panel, "ed_visits")
    print(f"   ATT (ED visits): {did_ed['att']:.3f}/member (p={did_ed['p_value']:.4f})")

    print("\n2. Propensity Score Matching (Sensitivity Analysis)")
    print("-" * 40)
    psm_cost = propensity_score_matching(panel, "total_cost")
    if "att" in psm_cost:
        print(f"   ATT (PSM): ${psm_cost['att']:,.2f}/member (p={psm_cost['p_value']:.4f})")
        print(f"   Matched pairs: {psm_cost['n_matched_pairs']:,}")
        print(f"   Post-match SMD (age): {psm_cost['smd_age_post_match']:.3f} (target <0.10)")

        # Generate Love Plot for covariate balance
        covariates = ["age", "dual_eligible", "ip_admits", "ed_visits"]
        if all(cov in psm_cost['treated_pre'].columns for cov in covariates):
            balance_df = generate_love_plot(
                psm_cost['treated_pre'], psm_cost['control_pre'],
                psm_cost['treated_matched'], psm_cost['control_matched'],
                covariates, save_path="reports/figures/05_love_plot.png"
            )
            print(f"   → Love Plot saved: reports/figures/05_love_plot.png")
    else:
        print(f"   PSM: {psm_cost.get('error', 'Failed')}")

    print("\n3. Shared Savings Projection (MSSP framework)")
    print("-" * 40)
    savings = project_shared_savings(did_cost["att"], n_attributed_lives, benchmark_pmpm)
    print(f"   Attributed lives:      {savings['n_attributed_lives']:,}")
    print(f"   Benchmark PMPM:        ${savings['benchmark_pmpm']:,.2f}")
    print(f"   Actual PMPM:           ${savings['actual_pmpm']:,.2f}")
    print(f"   Gross savings:         ${savings['gross_savings_total']:,.0f}")
    print(f"   Savings rate:          {savings['savings_rate_pct']:.1f}%")
    print(f"   Exceeds MSR (2%):      {savings['exceeds_msr']}")
    print(f"   Shared savings earned: ${savings['shared_savings_earned']:,.0f}")
    return {"did_cost": did_cost, "did_ip": did_ip, "did_ed": did_ed, "psm_cost": psm_cost, "savings": savings}
