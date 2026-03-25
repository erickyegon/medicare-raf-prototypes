"""
risk_stratification.py
-----------------------
Clinical risk stratification model for Medicare beneficiaries.

Predicts:
  1. High-cost risk tier (high / moderate / low)
  2. Continuous predicted annual cost

Pipeline:
  - Feature engineering from HCC flags + demographics + utilization history
  - XGBoost classifier + regressor
  - SHAP-based feature importance
  - Threshold-calibrated stratification bands

For production deployment this would consume CMS claims data
(Part A, B, D) processed through the HCC pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    mean_absolute_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings("ignore")

from ..modeling.hcc_mapper import HCC_COEFFICIENTS_V28, get_hcc_label
from ..modeling.raf_calculator import calculate_raf_batch, estimate_pmpm_cost


def engineer_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from beneficiary cohort.

    Features:
      - Demographics (age, sex, dual eligibility)
      - RAF score and components
      - HCC flag columns (one-hot for key HCCs)
      - HCC count
      - Age × sex interactions
    """
    df = cohort.copy().reset_index(drop=True)

    # Restore hccs to list if serialised as string
    if len(df) > 0 and isinstance(df.get("hccs", pd.Series([None])).iloc[0], str):
        import ast
        df["hccs"] = df["hccs"].apply(
            lambda x: list(ast.literal_eval(x)) if isinstance(x, str) else (list(x) if x else [])
        )

    # Calculate RAF only if not already present
    if "raf_score" not in df.columns:
        df_raf = calculate_raf_batch(df).reset_index(drop=True)
    else:
        df_raf = df.reset_index(drop=True)
        # Ensure hccs is a proper list column for membership tests
        if "hccs" not in df_raf.columns:
            df_raf["hccs"] = [[] for _ in range(len(df_raf))]

    # HCC flag features for top HCCs
    key_hccs = [8, 9, 11, 12, 17, 18, 19, 22, 40, 58, 79, 85, 86, 96,
                107, 108, 111, 134, 135, 136, 137, 138]

    for hcc in key_hccs:
        col = f"hcc_{hcc}"
        df_raf[col] = df_raf["hccs"].apply(lambda h: int(hcc in h))

    # Aggregate HCC features
    df_raf["hcc_count"]    = df_raf["hccs"].apply(len)
    df_raf["has_cancer"]   = df_raf["hccs"].apply(lambda h: int(bool(set(h) & {8, 9, 10, 11, 12})))
    df_raf["has_chf"]      = df_raf["hcc_85"]
    df_raf["has_diabetes"] = df_raf[["hcc_17", "hcc_18", "hcc_19"]].max(axis=1)
    df_raf["has_ckd"]      = df_raf[["hcc_134", "hcc_135", "hcc_136",
                                      "hcc_137", "hcc_138"]].max(axis=1)
    df_raf["has_copd"]     = df_raf["hcc_111"]
    df_raf["has_afib"]     = df_raf["hcc_96"]

    # Demographic features
    df_raf["age_scaled"]   = (df_raf["age"] - 72) / 10
    df_raf["is_female"]    = (df_raf["sex"] == "F").astype(int)
    df_raf["age_sq"]       = df_raf["age_scaled"] ** 2

    # Interaction features
    df_raf["chf_afib"]     = df_raf["has_chf"] * df_raf["has_afib"]
    df_raf["chf_diabetes"] = df_raf["has_chf"] * df_raf["has_diabetes"]
    df_raf["ckd_diabetes"] = df_raf["has_ckd"] * df_raf["has_diabetes"]
    df_raf["cancer_age"]   = df_raf["has_cancer"] * df_raf["age_scaled"]

    # Estimated cost from RAF
    df_raf["predicted_cost_raf"] = df_raf["raf_score"].apply(estimate_pmpm_cost)

    return df_raf


FEATURE_COLS = (
    ["age_scaled", "age_sq", "is_female", "dual_eligible",
     "raf_score", "demographic_raf", "hcc_raf", "hcc_count",
     "has_cancer", "has_chf", "has_diabetes", "has_ckd",
     "has_copd", "has_afib",
     "chf_afib", "chf_diabetes", "ckd_diabetes", "cancer_age"] +
    [f"hcc_{h}" for h in [8, 9, 11, 12, 17, 18, 19, 22, 40,
                           58, 79, 85, 86, 96, 107, 108, 111,
                           134, 135, 136, 137, 138]]
)


class RiskStratificationModel:
    """
    End-to-end risk stratification pipeline.

    Outputs:
      - Risk tier prediction (high / moderate / low)
      - Predicted annual cost (continuous)
      - Beneficiary-level risk scores for ACO outreach prioritisation
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.label_enc = LabelEncoder()
        self.clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state,
            verbosity=0,
        )
        self.reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
        )
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y_tier: pd.Series,
            y_cost: pd.Series) -> "RiskStratificationModel":
        y_enc = self.label_enc.fit_transform(y_tier)
        self.clf.fit(X[self.feature_cols], y_enc)
        self.reg.fit(X[self.feature_cols], y_cost)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fitted, "Model must be fitted first."
        tier_enc    = self.clf.predict(X[self.feature_cols])
        tier_proba  = self.clf.predict_proba(X[self.feature_cols])
        cost_pred   = self.reg.predict(X[self.feature_cols])

        return pd.DataFrame({
            "predicted_tier":      self.label_enc.inverse_transform(tier_enc),
            "prob_high":           tier_proba[:, list(self.label_enc.classes_).index("high")]
                                   if "high" in self.label_enc.classes_ else 0,
            "predicted_cost":      np.round(cost_pred, 2),
            "risk_score":          np.round(tier_proba[:, -1], 4),  # highest risk class prob
        })

    def evaluate(self, X: pd.DataFrame, y_tier: pd.Series,
                 y_cost: pd.Series) -> dict:
        preds = self.predict(X)
        y_enc = self.label_enc.transform(y_tier)
        tier_enc = self.label_enc.transform(preds["predicted_tier"])

        return {
            "tier_accuracy": round((preds["predicted_tier"].values == y_tier.values).mean(), 4),
            "cost_mae":      round(mean_absolute_error(y_cost.values, preds["predicted_cost"].values), 2),
            "cost_r2":       round(r2_score(y_cost.values, preds["predicted_cost"].values), 4),
        }

    def feature_importance(self) -> pd.DataFrame:
        fi = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.clf.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi["rank"] = range(1, len(fi) + 1)
        return fi.head(20)


def train_and_evaluate(cohort: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """
    Full training and evaluation pipeline.

    Parameters
    ----------
    cohort : beneficiary cohort DataFrame
    panel  : utilization panel DataFrame (pre-period for cost labels)

    Returns
    -------
    dict with model, metrics, feature importance, predictions
    """
    print("Engineering features...")
    X = engineer_features(cohort)

    # Cost labels from pre-period utilisation
    pre_costs = (
        panel[panel["year"] == 0]
        .set_index("bene_id")["total_cost"]
    )
    X["actual_cost"] = X["bene_id"].map(pre_costs)
    X = X.dropna(subset=["actual_cost"])

    y_tier = X["risk_tier"]
    y_cost = X["actual_cost"]

    X_train, X_test, yt_train, yt_test, yc_train, yc_test = train_test_split(
        X, y_tier, y_cost, test_size=0.20, random_state=42, stratify=y_tier
    )

    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

    model = RiskStratificationModel()
    print("Training XGBoost classifier + regressor...")
    model.fit(X_train, yt_train, yc_train)

    metrics = model.evaluate(X_test, yt_test, yc_test)
    print(f"\n  Tier accuracy : {metrics['tier_accuracy']:.1%}")
    print(f"  Cost MAE      : ${metrics['cost_mae']:,.0f}")
    print(f"  Cost R²       : {metrics['cost_r2']:.3f}")

    # Log to MLflow
    with mlflow.start_run(run_name="medicare_raf_risk_model"):
        # Log parameters
        mlflow.log_param("n_estimators", model.clf.n_estimators)
        mlflow.log_param("max_depth", model.clf.max_depth)
        mlflow.log_param("learning_rate", model.clf.learning_rate)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Log metrics
        mlflow.log_metric("tier_accuracy", metrics['tier_accuracy'])
        mlflow.log_metric("cost_mae", metrics['cost_mae'])
        mlflow.log_metric("cost_r2", metrics['cost_r2'])

        # Log feature importance
        fi = model.feature_importance()
        fi_dict = dict(zip(fi['feature'], fi['importance']))
        mlflow.log_dict(fi_dict, "feature_importance.json")

        # Log models
        mlflow.xgboost.log_model(model.clf, "classifier")
        mlflow.xgboost.log_model(model.reg, "regressor")

        print("  → MLflow run logged")

    print(f"\n  Top 5 features:\n{fi[['rank','feature','importance']].head(5).to_string(index=False)}")

    test_preds = model.predict(X_test)
    test_preds["bene_id"]     = X_test["bene_id"].values
    test_preds["actual_tier"] = yt_test.values
    test_preds["actual_cost"] = yc_test.values
    test_preds["raf_score"]   = X_test["raf_score"].values

    # SHAP analysis for explainability
    # Use XGBoost's native pred_contribs to avoid SHAP/XGBoost 3.x version incompatibility
    # (SHAP's XGBTreeModelLoader can't parse multiclass base_score vectors in XGBoost 3.x)
    print("Computing SHAP values...")
    import xgboost as xgb
    X_feat = X_test[model.feature_cols]
    dmat = xgb.DMatrix(X_feat)
    contribs = model.clf.get_booster().predict(dmat, pred_contribs=True)
    # XGBoost 3.x multiclass: shape (n_samples, n_classes, n_features+1)
    # Older versions: shape (n_samples, n_classes * (n_features+1))
    high_risk_idx = list(model.label_enc.classes_).index("high")
    n_feat = len(model.feature_cols)
    if contribs.ndim == 3:
        shap_high = contribs[:, high_risk_idx, :n_feat]
    else:
        # Flat layout: [class0_f0..fn_bias, class1_f0..fn_bias, ...]
        n_classes = len(model.label_enc.classes_)
        stride = n_feat + 1
        shap_high = contribs[:, high_risk_idx * stride : high_risk_idx * stride + n_feat]
    
    # Get feature importance from SHAP
    shap_importance = np.abs(shap_high).mean(axis=0)
    shap_fi = pd.DataFrame({
        "feature": model.feature_cols,
        "shap_importance": shap_importance
    }).sort_values("shap_importance", ascending=False)
    shap_fi["rank"] = range(1, len(shap_fi) + 1)

    return {
        "model":              model,
        "metrics":            metrics,
        "feature_importance": fi,
        "shap_importance":    shap_fi,
        "shap_values":        shap_high,
        "X_test":             X_test,
        "test_predictions":   test_preds,
    }
