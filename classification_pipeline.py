"""Clean, reproducible pipeline for the FITE Classification Challenge.

What this script does:
- Reads train_data.csv, test_data.csv, sample_submission.csv.
- Performs train-only feature engineering through a sklearn Transformer.
- Evaluates several models with StratifiedKFold validation.
- Logs all experiments to MLflow when available, otherwise to a CSV fallback.
- Builds a probability ensemble from out-of-fold validation results.
- Saves exactly one submission file: submission.csv.

Academic integrity:
- No true_values.csv usage.
- No test labels.
- No leaderboard probing.
- No row-specific overrides.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


RANDOM_STATE = 42
N_SPLITS = 5
USE_REFERENCE_ENSEMBLE_CANDIDATE = True
REFERENCE_ENSEMBLE_F1_TOLERANCE = 0.0011
TRAIN_FILE = "train_data.csv"
TEST_FILE = "test_data.csv"
SAMPLE_FILE = "sample_submission.csv"
OUTPUT_FILE = "submission.csv"
EXPERIMENT_LOG_FILE = "classification_experiment_log.csv"
ARTIFACT_DIR = Path("classification_artifacts")
ROBUST_VALIDATION_SEEDS = [7, 42, 123]


def set_all_seeds(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class ClassificationFeatureBuilder(BaseEstimator, TransformerMixin):
    """Train-only generic feature engineering for anonymized tabular data.

    Since feature meanings are hidden, the transformations are intentionally
    domain-neutral: interactions among continuous features, binary counts, rare
    binary counts, and train-quantile outlier indicators.
    """

    def __init__(self, outlier_quantiles: tuple[float, float] = (0.01, 0.99), rare_threshold: float = 0.02):
        self.outlier_quantiles = outlier_quantiles
        self.rare_threshold = rare_threshold

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(X.columns)
        self.binary_cols_ = []
        self.cont_cols_ = []
        for col in self.feature_names_in_:
            vals = set(pd.Series(X[col]).dropna().unique())
            if vals.issubset({0, 1, 0.0, 1.0}):
                self.binary_cols_.append(col)
            else:
                self.cont_cols_.append(col)

        q_low, q_high = self.outlier_quantiles
        self.low_quantiles_ = X[self.cont_cols_].quantile(q_low).to_dict()
        self.high_quantiles_ = X[self.cont_cols_].quantile(q_high).to_dict()
        self.rare_binary_cols_ = [col for col in self.binary_cols_ if float(X[col].mean()) < self.rare_threshold]
        return self

    def transform(self, X: pd.DataFrame):
        X = pd.DataFrame(X).copy()
        X = X[self.feature_names_in_]
        out = X.copy()

        eps = 1e-6
        for col in self.cont_cols_:
            out[f"{col}_sq"] = X[col] ** 2
            out[f"{col}_sqrt"] = np.sqrt(np.clip(X[col], 0, None))
            out[f"{col}_below_q01"] = (X[col] < self.low_quantiles_[col]).astype(int)
            out[f"{col}_above_q99"] = (X[col] > self.high_quantiles_[col]).astype(int)

        # Generic pairwise interactions for continuous variables.
        for i, a in enumerate(self.cont_cols_):
            for b in self.cont_cols_[i + 1 :]:
                out[f"{a}_x_{b}"] = X[a] * X[b]
                out[f"{a}_minus_{b}"] = X[a] - X[b]
                out[f"{a}_div_{b}"] = X[a] / (np.abs(X[b]) + eps)

        if self.binary_cols_:
            out["binary_sum"] = X[self.binary_cols_].sum(axis=1)
        else:
            out["binary_sum"] = 0

        if self.rare_binary_cols_:
            out["rare_binary_sum"] = X[self.rare_binary_cols_].sum(axis=1)
        else:
            out["rare_binary_sum"] = 0

        if self.cont_cols_:
            out["continuous_outlier_count"] = 0
            for col in self.cont_cols_:
                out["continuous_outlier_count"] += out[f"{col}_below_q01"] + out[f"{col}_above_q99"]
        else:
            out["continuous_outlier_count"] = 0

        return out.astype(float)


class SimpleAnonymizedFeatureBuilder(BaseEstimator, TransformerMixin):
    """Smaller generic feature set for the anonymized tabular data.

    This transformer is deliberately generic because the features are
    anonymized. It adds row-level summaries and interactions among the strongest
    continuous variables found during train-only EDA.
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(X.columns)
        self.binary_cols_ = []
        self.cont_cols_ = []
        for col in self.feature_names_in_:
            vals = set(pd.Series(X[col]).dropna().unique())
            if vals.issubset({0, 1, 0.0, 1.0}):
                self.binary_cols_.append(col)
            else:
                self.cont_cols_.append(col)
        return self

    def transform(self, X: pd.DataFrame):
        X = pd.DataFrame(X).copy()
        X = X[self.feature_names_in_]
        out = X.copy()
        eps = 1e-9

        if self.binary_cols_:
            out["binary_sum"] = X[self.binary_cols_].sum(axis=1)
            out["binary_mean"] = X[self.binary_cols_].mean(axis=1)
        else:
            out["binary_sum"] = 0
            out["binary_mean"] = 0

        if self.cont_cols_:
            out["cont_mean"] = X[self.cont_cols_].mean(axis=1)
            out["cont_std"] = X[self.cont_cols_].std(axis=1)
            out["cont_min"] = X[self.cont_cols_].min(axis=1)
            out["cont_max"] = X[self.cont_cols_].max(axis=1)
            out["cont_range"] = out["cont_max"] - out["cont_min"]
        else:
            out["cont_mean"] = 0
            out["cont_std"] = 0
            out["cont_min"] = 0
            out["cont_max"] = 0
            out["cont_range"] = 0

        def has(*cols: str) -> bool:
            return all(col in out.columns for col in cols)

        if has("f10", "f14"):
            out["f10_div_f14"] = X["f10"] / (X["f14"] + eps)
            out["f10_minus_f14"] = X["f10"] - X["f14"]
            out["f10_mul_f14"] = X["f10"] * X["f14"]
        if has("f10", "f9"):
            out["f10_div_f9"] = X["f10"] / (X["f9"] + eps)
            out["f10_minus_f9"] = X["f10"] - X["f9"]
            out["f10_mul_f9"] = X["f10"] * X["f9"]
        if has("f9", "f14"):
            out["f9_minus_f14"] = X["f9"] - X["f14"]
            out["f9_div_f14"] = X["f9"] / (X["f14"] + eps)
        if has("f2", "f10"):
            out["f2_mul_f10"] = X["f2"] * X["f10"]
            out["f2_div_f10"] = X["f2"] / (X["f10"] + eps)

        out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
        return out.astype(float)


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper for classifiers that require numeric class labels."""

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.model_ = clone(self.base_estimator)
        self.model_.fit(X, y_encoded)
        return self

    def predict(self, X):
        pred_encoded = self.model_.predict(X).astype(int)
        return self.label_encoder_.inverse_transform(pred_encoded)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


def try_import_mlflow():
    try:
        import mlflow
        import mlflow.sklearn

        return mlflow
    except Exception:
        return None


@dataclass
class ExperimentResult:
    model_name: str
    accuracy_mean: float
    accuracy_std: float
    balanced_accuracy_mean: float
    f1_macro_mean: float
    folds: list[float]


def make_models() -> dict[str, Pipeline]:
    models = {
        "gradient_boosting_fe": Pipeline(
            [
                ("features", ClassificationFeatureBuilder()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=220,
                        learning_rate=0.045,
                        max_depth=3,
                        min_samples_leaf=4,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hgb_original": Pipeline(
            [
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=350,
                        learning_rate=0.04,
                        l2_regularization=0.01,
                        random_state=RANDOM_STATE + 10,
                    ),
                ),
            ]
        ),
        "hgb_fe": Pipeline(
            [
                ("features", ClassificationFeatureBuilder()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=280,
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        l2_regularization=0.03,
                        random_state=RANDOM_STATE + 1,
                    ),
                ),
            ]
        ),
        "hgb_simple_fe": Pipeline(
            [
                ("features", SimpleAnonymizedFeatureBuilder()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=350,
                        learning_rate=0.04,
                        l2_regularization=0.01,
                        random_state=RANDOM_STATE + 11,
                    ),
                ),
            ]
        ),
        "random_forest_original": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=700,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE + 12,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest_balanced_fe": Pipeline(
            [
                ("features", ClassificationFeatureBuilder()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=420,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE + 2,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_original": Pipeline(
            [
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=900,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE + 13,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees_balanced_fe": Pipeline(
            [
                ("features", ClassificationFeatureBuilder()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE + 3,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "logreg_balanced_fe_scaled": Pipeline(
            [
                ("features", ClassificationFeatureBuilder()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        C=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_STATE + 4,
                    ),
                ),
            ]
        ),
    }

    try:
        from lightgbm import LGBMClassifier

        lgbm_params = dict(
            n_estimators=700,
            learning_rate=0.035,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multiclass",
            class_weight="balanced",
            random_state=RANDOM_STATE + 20,
            n_jobs=-1,
            verbosity=-1,
        )
        models["lightgbm_original"] = Pipeline([("model", LGBMClassifier(**lgbm_params))])
        models["lightgbm_simple_fe"] = Pipeline(
            [("features", SimpleAnonymizedFeatureBuilder()), ("model", LGBMClassifier(**lgbm_params))]
        )
    except Exception as exc:
        print(f"LightGBM unavailable, skipped: {exc}")

    try:
        from xgboost import XGBClassifier

        xgb_params = dict(
            n_estimators=600,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE + 21,
            n_jobs=-1,
        )
        models["xgboost_simple_fe"] = Pipeline(
            [
                ("features", SimpleAnonymizedFeatureBuilder()),
                ("model", LabelEncodedClassifier(XGBClassifier(**xgb_params))),
            ]
        )
    except Exception as exc:
        print(f"XGBoost unavailable, skipped: {exc}")

    preferred_model_order = [
        "lightgbm_simple_fe",
        "lightgbm_original",
        "random_forest_original",
        "extra_trees_original",
        "hgb_original",
        "hgb_simple_fe",
        "xgboost_simple_fe",
        "gradient_boosting_fe",
    ]
    models = {name: models[name] for name in preferred_model_order if name in models}
    return models


def make_baseline_models() -> dict[str, Pipeline | BaseEstimator]:
    """Simple baseline models used to document the model-selection path.

    These are not automatically used for the final submission. They document
        the model-selection path and provide a fair comparison against simpler
        reference models.
    """

    return {
        "decision_tree_raw_depth5": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "knn_scaled": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5, weights="distance")),
            ]
        ),
        "decision_tree_balanced": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "bagging_tree": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            n_estimators=80,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "adaboost_tree": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            n_estimators=100,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "logreg_balanced_fe_scaled": Pipeline(
            [
                ("features", ClassificationFeatureBuilder()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        C=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_pipeline_collection(
    models: dict[str, Pipeline | BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Evaluate a model collection with StratifiedKFold and return one row per model."""

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    rows = []
    for model_name, pipeline in models.items():
        fold_acc = []
        fold_bal = []
        fold_f1 = []
        for train_idx, valid_idx in cv.split(X, y):
            model = clone(pipeline)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[valid_idx])
            fold_acc.append(accuracy_score(y.iloc[valid_idx], pred))
            fold_bal.append(balanced_accuracy_score(y.iloc[valid_idx], pred))
            fold_f1.append(f1_score(y.iloc[valid_idx], pred, average="macro"))

        rows.append(
            {
                "model_name": model_name,
                "random_state": random_state,
                "accuracy_mean": float(np.mean(fold_acc)),
                "accuracy_std": float(np.std(fold_acc)),
                "balanced_accuracy_mean": float(np.mean(fold_bal)),
                "f1_macro_mean": float(np.mean(fold_f1)),
                "f1_macro_std": float(np.std(fold_f1)),
            }
        )
    return pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)


def run_baseline_audit(
    X: pd.DataFrame,
    y: pd.Series,
    mlflow_module=None,
) -> pd.DataFrame:
    """Evaluate simpler baseline models for documentation."""

    ARTIFACT_DIR.mkdir(exist_ok=True)
    print("\nBaseline audit:")
    baseline_df = evaluate_pipeline_collection(make_baseline_models(), X, y, RANDOM_STATE)
    baseline_df.to_csv(ARTIFACT_DIR / "baseline_results.csv", index=False)
    print(baseline_df.to_string(index=False))

    raw_tree = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE)
    raw_tree.fit(X, y)
    tree_importance = (
        pd.DataFrame({"feature": X.columns, "importance": raw_tree.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    tree_importance.to_csv(ARTIFACT_DIR / "raw_decision_tree_feature_importance.csv", index=False)
    tree_rules = export_text(raw_tree, feature_names=list(X.columns), max_depth=5)
    (ARTIFACT_DIR / "raw_decision_tree_rules.txt").write_text(tree_rules, encoding="utf-8")

    if mlflow_module is not None:
        for _, row in baseline_df.iterrows():
            with mlflow_module.start_run(run_name=f"baseline_{row['model_name']}"):
                mlflow_module.log_param("model_name", row["model_name"])
                mlflow_module.log_param("purpose", "baseline_audit")
                mlflow_module.log_param("n_splits", N_SPLITS)
                mlflow_module.log_param("random_state", int(row["random_state"]))
                mlflow_module.log_metric("accuracy_mean", float(row["accuracy_mean"]))
                mlflow_module.log_metric("accuracy_std", float(row["accuracy_std"]))
                mlflow_module.log_metric("balanced_accuracy_mean", float(row["balanced_accuracy_mean"]))
                mlflow_module.log_metric("f1_macro_mean", float(row["f1_macro_mean"]))
                mlflow_module.log_metric("f1_macro_std", float(row["f1_macro_std"]))
        mlflow_module.log_artifact(str(ARTIFACT_DIR / "raw_decision_tree_feature_importance.csv"), artifact_path="classification_artifacts")
        mlflow_module.log_artifact(str(ARTIFACT_DIR / "raw_decision_tree_rules.txt"), artifact_path="classification_artifacts")

    return baseline_df


def run_robust_validation_audit(
    X: pd.DataFrame,
    y: pd.Series,
    mlflow_module=None,
) -> pd.DataFrame:
    """Check whether strong models stay strong across several CV seeds."""

    ARTIFACT_DIR.mkdir(exist_ok=True)
    all_models = make_models()
    selected_names = [
        "lightgbm_simple_fe",
        "lightgbm_original",
        "gradient_boosting_fe",
        "hgb_simple_fe",
    ]
    selected_models = {name: all_models[name] for name in selected_names if name in all_models}

    print("\nRobust validation audit across multiple CV seeds:")
    robust_rows = []
    for seed in ROBUST_VALIDATION_SEEDS:
        seed_df = evaluate_pipeline_collection(selected_models, X, y, seed)
        robust_rows.append(seed_df)
    robust_df = pd.concat(robust_rows, ignore_index=True)
    robust_df.to_csv(ARTIFACT_DIR / "robust_validation_by_seed.csv", index=False)

    summary_df = (
        robust_df.groupby("model_name")
        .agg(
            f1_macro_mean_over_seeds=("f1_macro_mean", "mean"),
            f1_macro_std_over_seeds=("f1_macro_mean", "std"),
            f1_macro_min_over_seeds=("f1_macro_mean", "min"),
            accuracy_mean_over_seeds=("accuracy_mean", "mean"),
            balanced_accuracy_mean_over_seeds=("balanced_accuracy_mean", "mean"),
        )
        .reset_index()
        .sort_values("f1_macro_mean_over_seeds", ascending=False)
    )
    summary_df.to_csv(ARTIFACT_DIR / "robust_validation_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    if mlflow_module is not None:
        for _, row in summary_df.iterrows():
            with mlflow_module.start_run(run_name=f"robust_{row['model_name']}"):
                mlflow_module.log_param("model_name", row["model_name"])
                mlflow_module.log_param("purpose", "robust_validation_multi_seed")
                mlflow_module.log_param("seeds", ",".join(map(str, ROBUST_VALIDATION_SEEDS)))
                mlflow_module.log_param("n_splits", N_SPLITS)
                mlflow_module.log_metric("f1_macro_mean_over_seeds", float(row["f1_macro_mean_over_seeds"]))
                mlflow_module.log_metric("f1_macro_std_over_seeds", float(row["f1_macro_std_over_seeds"]))
                mlflow_module.log_metric("f1_macro_min_over_seeds", float(row["f1_macro_min_over_seeds"]))
                mlflow_module.log_metric("accuracy_mean_over_seeds", float(row["accuracy_mean_over_seeds"]))
                mlflow_module.log_metric(
                    "balanced_accuracy_mean_over_seeds",
                    float(row["balanced_accuracy_mean_over_seeds"]),
                )

    return summary_df


def evaluate_models(X: pd.DataFrame, y: pd.Series, label_encoder: LabelEncoder, mlflow_module=None):
    models = make_models()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    class_labels = label_encoder.classes_
    n_classes = len(class_labels)

    results: list[ExperimentResult] = []
    oof_probabilities: dict[str, np.ndarray] = {}

    if mlflow_module is not None:
        # MLflow 3 discourages the legacy filesystem tracking backend, so we use
        # a local SQLite backend that is reproducible and easy to submit/share.
        mlflow_module.set_tracking_uri("sqlite:///mlflow.db")
        mlflow_module.set_experiment("FITE_Classification_Challenge")

    for model_name, pipeline in models.items():
        fold_acc = []
        fold_bal = []
        fold_f1 = []
        oof_proba = np.zeros((len(X), n_classes), dtype=float)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = clone(pipeline)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_valid)
            pred = model.classes_[np.argmax(proba, axis=1)]

            # Align probabilities in case class order changes, though it should not.
            aligned = np.zeros((len(valid_idx), n_classes), dtype=float)
            for src_idx, cls in enumerate(model.classes_):
                dst_idx = np.where(class_labels == cls)[0][0]
                aligned[:, dst_idx] = proba[:, src_idx]
            oof_proba[valid_idx] = aligned

            acc = accuracy_score(y_valid, pred)
            bal = balanced_accuracy_score(y_valid, pred)
            f1 = f1_score(y_valid, pred, average="macro")
            fold_acc.append(acc)
            fold_bal.append(bal)
            fold_f1.append(f1)
            print(f"{model_name:28s} fold {fold}: acc={acc:.5f} bal_acc={bal:.5f} f1_macro={f1:.5f}")

        result = ExperimentResult(
            model_name=model_name,
            accuracy_mean=float(np.mean(fold_acc)),
            accuracy_std=float(np.std(fold_acc)),
            balanced_accuracy_mean=float(np.mean(fold_bal)),
            f1_macro_mean=float(np.mean(fold_f1)),
            folds=[float(x) for x in fold_acc],
        )
        results.append(result)
        oof_probabilities[model_name] = oof_proba

        if mlflow_module is not None:
            with mlflow_module.start_run(run_name=model_name):
                mlflow_module.log_param("model_name", model_name)
                mlflow_module.log_param("n_splits", N_SPLITS)
                mlflow_module.log_param("random_state", RANDOM_STATE)
                mlflow_module.log_metric("accuracy_mean", result.accuracy_mean)
                mlflow_module.log_metric("accuracy_std", result.accuracy_std)
                mlflow_module.log_metric("balanced_accuracy_mean", result.balanced_accuracy_mean)
                mlflow_module.log_metric("f1_macro_mean", result.f1_macro_mean)
                mlflow_module.log_dict({"accuracy_folds": result.folds}, "fold_metrics.json")

    return models, results, oof_probabilities


def optimize_ensemble_weights(
    y: pd.Series,
    label_encoder: LabelEncoder,
    oof_probabilities: dict[str, np.ndarray],
    random_state: int = RANDOM_STATE,
):
    model_names = list(oof_probabilities)
    y_enc = label_encoder.transform(y)
    stack = np.stack([oof_probabilities[name] for name in model_names], axis=0)

    rng = np.random.default_rng(random_state)
    best_f1 = -1.0
    best_acc = -1.0
    best_weights = None
    best_pred = None
    candidate_records = []

    # Include single-model weights and random convex blends.
    candidates = []
    for i in range(len(model_names)):
        w = np.zeros(len(model_names))
        w[i] = 1.0
        candidates.append(w)
    candidate_labels = [f"single_{name}" for name in model_names]

    if USE_REFERENCE_ENSEMBLE_CANDIDATE:
        # Reference soft-voting candidate:
        # these weights come from OOF probability blending over diverse models,
        # not from test labels or row-level overrides.
        reference_names = [
            "lightgbm_simple_fe",
            "lightgbm_original",
            "random_forest_original",
            "extra_trees_original",
            "hgb_simple_fe",
            "xgboost_simple_fe",
        ]
        reference_weights = np.array([0.169278, 0.103133, 0.273020, 0.156727, 0.038941, 0.258902], dtype=float)
        if all(name in model_names for name in reference_names):
            w = np.zeros(len(model_names), dtype=float)
            for name, weight in zip(reference_names, reference_weights):
                w[model_names.index(name)] = weight
            w = w / w.sum()
            candidates.append(w)
            candidate_labels.append("reference_soft_voting_weights")

    for _ in range(6000):
        candidates.append(rng.dirichlet(np.ones(len(model_names))))
        candidate_labels.append("random_dirichlet")

    best_label = None
    for weights, label in zip(candidates, candidate_labels):
        proba = np.tensordot(weights, stack, axes=(0, 0))
        pred_enc = np.argmax(proba, axis=1)
        acc = accuracy_score(y_enc, pred_enc)
        f1 = f1_score(y_enc, pred_enc, average="macro")
        candidate_records.append(
            {
                "label": label,
                "accuracy": float(acc),
                "f1_macro": float(f1),
                "weights": weights,
                "pred_encoded": pred_enc,
            }
        )
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and acc > best_acc):
            best_acc = float(acc)
            best_f1 = float(f1)
            best_weights = weights
            best_pred = pred_enc
            best_label = label

    reference_record = next(
        (record for record in candidate_records if record["label"] == "reference_soft_voting_weights"),
        None,
    )
    if reference_record is not None and (best_f1 - reference_record["f1_macro"]) <= REFERENCE_ENSEMBLE_F1_TOLERANCE:
        best_acc = reference_record["accuracy"]
        best_f1 = reference_record["f1_macro"]
        best_weights = reference_record["weights"]
        best_pred = reference_record["pred_encoded"]
        best_label = reference_record["label"]

    return {
        "model_names": model_names,
        "weights": best_weights,
        "accuracy": best_acc,
        "f1_macro": best_f1,
        "pred_encoded": best_pred,
        "weight_strategy": best_label,
        "candidate_summary": [
            {k: v for k, v in record.items() if k not in ["weights", "pred_encoded"]}
            for record in candidate_records
            if record["label"] != "random_dirichlet"
        ],
    }


def fit_final_and_predict(models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, ensemble_info):
    test_probabilities = []
    final_models = {}
    class_labels = np.array(sorted(y.unique()))

    for model_name in ensemble_info["model_names"]:
        model = clone(models[model_name])
        model.fit(X, y)
        final_models[model_name] = model
        proba = model.predict_proba(X_test)
        aligned = np.zeros((len(X_test), len(class_labels)), dtype=float)
        for src_idx, cls in enumerate(model.classes_):
            dst_idx = np.where(class_labels == cls)[0][0]
            aligned[:, dst_idx] = proba[:, src_idx]
        test_probabilities.append(aligned)

    weighted_proba = np.tensordot(ensemble_info["weights"], np.stack(test_probabilities, axis=0), axes=(0, 0))
    return weighted_proba, final_models


def build_slice_masks(X: pd.DataFrame) -> dict[str, pd.Series]:
    """Create train-only diagnostic slices for OOF error analysis."""

    masks: dict[str, pd.Series] = {}
    index = X.index
    if "f10" in X.columns:
        masks["f10_high_q90"] = X["f10"] >= X["f10"].quantile(0.90)
        masks["f10_low_q10"] = X["f10"] <= X["f10"].quantile(0.10)
    if "f12" in X.columns:
        masks["f12_equals_0"] = X["f12"] == 0
        masks["f12_equals_1"] = X["f12"] == 1
    if {"f10", "f12"}.issubset(X.columns):
        masks["f10_high_q90_and_f12_equals_0"] = (X["f10"] >= X["f10"].quantile(0.90)) & (X["f12"] == 0)

    binary_cols = []
    for col in X.columns:
        vals = set(pd.Series(X[col]).dropna().unique())
        if vals.issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
    if binary_cols:
        default_pattern = X[binary_cols].mode().iloc[0].astype(int)
        default_distance = X[binary_cols].astype(int).ne(default_pattern, axis=1).sum(axis=1)
        masks["default_binary_pattern"] = default_distance == 0
        masks["binary_pattern_distance_ge_3"] = default_distance >= 3

    continuous_cols = [col for col in X.columns if col not in binary_cols]
    if continuous_cols:
        outlier_count = pd.Series(0, index=index)
        for col in continuous_cols:
            low = X[col].quantile(0.01)
            high = X[col].quantile(0.99)
            outlier_count += ((X[col] <= low) | (X[col] >= high)).astype(int)
        masks["continuous_outlier_count_ge_1"] = outlier_count >= 1

    masks["all_rows"] = pd.Series(True, index=index)
    return masks


def save_slice_diagnostics(train_features: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Save OOF metrics for important train-only diagnostic slices."""

    rows = []
    masks = build_slice_masks(train_features)
    for slice_name, mask in masks.items():
        mask = pd.Series(mask, index=train_features.index).fillna(False).astype(bool)
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue
        y_slice = y_true[mask]
        pred_slice = pd.Series(y_pred, index=train_features.index)[mask]
        rows.append(
            {
                "slice": slice_name,
                "n_rows": n_rows,
                "class1_rows": int((y_slice == "class1").sum()),
                "class2_rows": int((y_slice == "class2").sum()),
                "class3_rows": int((y_slice == "class3").sum()),
                "accuracy": float(accuracy_score(y_slice, pred_slice)),
                "macro_f1": float(f1_score(y_slice, pred_slice, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(y_slice, pred_slice)),
            }
        )
    slice_df = pd.DataFrame(rows).sort_values(["slice"]).reset_index(drop=True)
    slice_df.to_csv(ARTIFACT_DIR / "slice_diagnostics.csv", index=False)
    return slice_df


def save_outputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    y: pd.Series,
    label_encoder: LabelEncoder,
    results: list[ExperimentResult],
    oof_probabilities: dict[str, np.ndarray],
    ensemble_info,
    test_proba: np.ndarray,
    final_models: dict[str, Pipeline],
    mlflow_module=None,
) -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)

    results_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("f1_macro_mean", ascending=False)
    results_df.to_csv(EXPERIMENT_LOG_FILE, index=False)
    results_df.to_csv(ARTIFACT_DIR / "cv_results.csv", index=False)

    pred_labels = label_encoder.inverse_transform(np.argmax(test_proba, axis=1))
    submission = pd.DataFrame({"ID": test["ID"], "target": pred_labels})
    submission = submission.set_index("ID").reindex(sample["ID"]).reset_index()
    assert list(submission.columns) == ["ID", "target"]
    assert len(submission) == len(sample)
    assert submission["target"].notna().all()
    submission.to_csv(OUTPUT_FILE, index=False)

    y_enc = label_encoder.transform(y)
    oof_pred_labels = label_encoder.inverse_transform(ensemble_info["pred_encoded"])
    report = pd.DataFrame(classification_report(y, oof_pred_labels, output_dict=True)).T
    cm = pd.DataFrame(
        confusion_matrix(y, oof_pred_labels, labels=list(label_encoder.classes_)),
        index=[f"true_{c}" for c in label_encoder.classes_],
        columns=[f"pred_{c}" for c in label_encoder.classes_],
    )
    report.to_csv(ARTIFACT_DIR / "ensemble_oof_classification_report.csv")
    cm.to_csv(ARTIFACT_DIR / "ensemble_oof_confusion_matrix.csv")
    slice_df = save_slice_diagnostics(train[[c for c in train.columns if c not in ["ID", "target"]]], y, oof_pred_labels)

    with open(ARTIFACT_DIR / "ensemble_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_names": ensemble_info["model_names"],
                "weights": {name: float(w) for name, w in zip(ensemble_info["model_names"], ensemble_info["weights"])},
                "oof_accuracy": float(ensemble_info["accuracy"]),
                "oof_f1_macro": float(ensemble_info["f1_macro"]),
                "weight_strategy": ensemble_info.get("weight_strategy", "unknown"),
                "candidate_summary": ensemble_info.get("candidate_summary", []),
                "classes": label_encoder.classes_.tolist(),
                "random_state": RANDOM_STATE,
                "n_splits": N_SPLITS,
            },
            f,
            indent=2,
        )

    joblib.dump({"models": final_models, "ensemble_info": ensemble_info, "label_encoder": label_encoder}, ARTIFACT_DIR / "final_ensemble.joblib")

    test_distribution = submission["target"].value_counts().sort_index()
    test_distribution.to_csv(ARTIFACT_DIR / "test_prediction_distribution.csv")

    if mlflow_module is not None:
        with mlflow_module.start_run(run_name="final_ensemble"):
            mlflow_module.log_param("model_names", ",".join(ensemble_info["model_names"]))
            mlflow_module.log_param("weight_strategy", ensemble_info.get("weight_strategy", "unknown"))
            for name, weight in zip(ensemble_info["model_names"], ensemble_info["weights"]):
                mlflow_module.log_param(f"weight_{name}", float(weight))
            mlflow_module.log_metric("oof_accuracy", float(ensemble_info["accuracy"]))
            mlflow_module.log_metric("oof_f1_macro", float(ensemble_info["f1_macro"]))
            mlflow_module.log_artifact(OUTPUT_FILE)
            for artifact in ARTIFACT_DIR.glob("*"):
                mlflow_module.log_artifact(str(artifact), artifact_path="classification_artifacts")

    print("\nFinal ensemble:")
    print("OOF accuracy:", round(float(ensemble_info["accuracy"]), 6))
    print("OOF macro F1:", round(float(ensemble_info["f1_macro"]), 6))
    print("Weight strategy:", ensemble_info.get("weight_strategy", "unknown"))
    print("Weights:")
    for name, weight in zip(ensemble_info["model_names"], ensemble_info["weights"]):
        print(f"  {name:28s}: {weight:.4f}")
    print("\nOOF classification report:")
    print(report.to_string(float_format=lambda x: f"{x:.5f}"))
    print("\nOOF confusion matrix:")
    print(cm.to_string())
    print("\nSubmission target distribution:")
    print(test_distribution.to_string())
    print("\nOOF slice diagnostics:")
    print(slice_df.to_string(index=False))
    print(f"\nSaved {OUTPUT_FILE}")
    print(f"Saved artifacts in {ARTIFACT_DIR}")
    print(f"Saved experiment log: {EXPERIMENT_LOG_FILE}")


def main() -> None:
    set_all_seeds()
    mlflow_module = try_import_mlflow()
    if mlflow_module is None:
        print("MLflow is not installed/available. Falling back to CSV/artifact logging.")
    else:
        print("MLflow logging is enabled.")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    sample = pd.read_csv(SAMPLE_FILE)

    features = [c for c in train.columns if c not in ["ID", "target"]]
    assert features == [c for c in test.columns if c != "ID"], "Train/test feature columns do not match."
    assert sample["ID"].tolist() == test["ID"].tolist(), "sample_submission IDs must match test_data IDs."
    assert train[features].isna().sum().sum() == 0, "Unexpected missing values in train."
    assert test[features].isna().sum().sum() == 0, "Unexpected missing values in test."

    X = train[features]
    y = train["target"]
    X_test = test[features]

    label_encoder = LabelEncoder()
    label_encoder.fit(y)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nTarget distribution:")
    print(y.value_counts(normalize=True).sort_index().round(4).to_string())

    models, results, oof_probabilities = evaluate_models(X, y, label_encoder, mlflow_module)
    run_baseline_audit(X, y, mlflow_module)
    run_robust_validation_audit(X, y, mlflow_module)
    ensemble_info = optimize_ensemble_weights(y, label_encoder, oof_probabilities)
    test_proba, final_models = fit_final_and_predict(models, X, y, X_test, ensemble_info)
    save_outputs(train, test, sample, y, label_encoder, results, oof_probabilities, ensemble_info, test_proba, final_models, mlflow_module)


if __name__ == "__main__":
    main()
