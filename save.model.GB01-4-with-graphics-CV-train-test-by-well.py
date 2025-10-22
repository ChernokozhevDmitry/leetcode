#!/usr/bin/env python3
"""
Train and evaluate a Gradient Boosting model with grouped cross-validation by well,
optionally perform a grouped holdout test split, generate graphics, and save
artifacts (model, metrics, predictions, plots).

Usage example:

    python save.model.GB01-4-with-graphics-CV-train-test-by-well.py \
        --data data.csv \
        --target target_column \
        --group-col well \
        --outdir artifacts/GB01-4 \
        --cv 5 --model-type auto --random-state 42

Install dependencies (if needed):

    pip install pandas numpy scikit-learn matplotlib seaborn joblib

"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


# ------------------------------
# Utilities
# ------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def infer_problem_type(y: pd.Series, model_type_arg: str = "auto") -> str:
    if model_type_arg in {"regression", "classification"}:
        return model_type_arg
    # auto inference
    if y.dtype.kind in {"i", "b"}:  # integers or boolean -> likely classification
        unique_vals = y.dropna().unique()
        if len(unique_vals) <= max(20, int(0.05 * len(y))):
            return "classification"
        # Many unique ints -> treat as regression
        return "regression"
    if y.dtype.kind in {"f"}:  # floats
        return "regression"
    # object/categorical -> classification
    return "classification"


def get_feature_name_list_from_preprocessor(preprocessor: ColumnTransformer) -> Optional[List[str]]:
    try:
        # sklearn >= 1.0
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        try:
            # Fallback: build from transformers
            names: List[str] = []
            for name, transformer, cols in preprocessor.transformers_:
                if name == 'remainder' and transformer == 'drop':
                    continue
                if hasattr(transformer, 'get_feature_names_out'):
                    sub_names = list(transformer.get_feature_names_out(cols))
                elif isinstance(transformer, Pipeline):
                    last_step = transformer.steps[-1][1]
                    if hasattr(last_step, 'get_feature_names_out'):
                        sub_names = list(last_step.get_feature_names_out(cols))
                    else:
                        sub_names = list(cols)
                else:
                    sub_names = list(cols)
                names.extend([f"{name}__{c}" for c in sub_names])
            return names
        except Exception:
            return None


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100.0
    if np.isnan(mape):
        mape = float('nan')
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

    metrics: Dict[str, object] = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "classification_report": classification_report(y_true, y_pred, target_names=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
                ap = average_precision_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                # For AP in multi-class, use macro-average of per-class AP
                y_true_bin = pd.get_dummies(y_true).to_numpy()
                ap = average_precision_score(y_true_bin, y_proba, average='macro')
            metrics.update({
                "roc_auc": float(auc),
                "avg_precision": float(ap),
            })
        except Exception:
            pass

    return metrics


# ------------------------------
# Plotting helpers
# ------------------------------

def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=20, alpha=0.6)
    min_val = np.nanmin([y_true, y_pred])
    max_val = np.nanmax([y_true, y_pred])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residuals_hist(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str) -> None:
    residuals = y_pred - y_true
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=40, kde=True)
    plt.xlabel("Residual (Pred - Actual)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, out_path: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curve_binary(y_true: np.ndarray, y_proba: np.ndarray, title: str, out_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    auc = roc_auc_score(y_true, y_proba[:, 1])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curve_binary(y_true: np.ndarray, y_proba: np.ndarray, title: str, out_path: str) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
    ap = average_precision_score(y_true, y_proba[:, 1])
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_calibration_binary(y_true: np.ndarray, y_proba: np.ndarray, title: str, out_path: str) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_proba[:, 1], n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Predicted probability")
    plt.ylabel("True probability")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(model: Pipeline, out_path: str, top_n: int = 30) -> None:
    try:
        gb = model.named_steps.get('gb')
        pre = model.named_steps.get('preprocessor')
        if gb is None or pre is None:
            return
        importances = getattr(gb, 'feature_importances_', None)
        if importances is None:
            return
        names = get_feature_name_list_from_preprocessor(pre)
        if names is None:
            names = [f"f{i}" for i in range(len(importances))]
        fi = pd.DataFrame({"feature": names, "importance": importances})
        fi.sort_values("importance", ascending=False, inplace=True)
        fi_top = fi.head(top_n)
        plt.figure(figsize=(8, max(4, int(0.25 * len(fi_top)))))
        sns.barplot(data=fi_top, x="importance", y="feature")
        plt.title("Feature importance (Gradient Boosting)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        # Also save CSV of full importances
        fi.to_csv(os.path.splitext(out_path)[0] + ".csv", index=False)
    except Exception:
        pass


# ------------------------------
# Core training/evaluation
# ------------------------------

def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    problem_type: str,
    params: Dict,
) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=params.get("num_impute", "median"))),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    if problem_type == "regression":
        model = GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3),
            subsample=params.get("subsample", 1.0),
            max_features=params.get("max_features", None),
            random_state=params.get("random_state", 42),
            validation_fraction=params.get("validation_fraction", 0.1),
            n_iter_no_change=params.get("n_iter_no_change", None),
            tol=params.get("tol", 1e-4),
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3),
            subsample=params.get("subsample", 1.0),
            max_features=params.get("max_features", None),
            random_state=params.get("random_state", 42),
            validation_fraction=params.get("validation_fraction", 0.1),
            n_iter_no_change=params.get("n_iter_no_change", None),
            tol=params.get("tol", 1e-4),
        )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("gb", model),
    ])
    return pipe


def grouped_cv_predictions(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    problem_type: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict[str, float]]]:
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = np.zeros_like(y, dtype=float if problem_type == 'regression' else int)
    y_proba: Optional[np.ndarray] = None
    if problem_type == 'classification':
        # We'll infer number of classes from y
        n_classes = len(np.unique(y))
        y_proba = np.zeros((len(y), n_classes), dtype=float)

    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        m = clone(model)
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]
        m.fit(X_tr, y_tr)
        if problem_type == 'regression':
            preds = m.predict(X_va)
            y_pred[val_idx] = preds
            fold_metrics.append(compute_regression_metrics(y_va, preds))
        else:
            preds = m.predict(X_va)
            y_pred[val_idx] = preds
            try:
                proba = m.predict_proba(X_va)
                # Map classes order to indices of y unique values encountered in training
                # We'll align to m.classes_
                class_order = list(m.named_steps['gb'].classes_)
                # Create a y_proba matrix with columns ordered by unique(y) sorted
                # But for metrics we'll store in y_proba[val_idx] consistent with m.classes_
                # We'll build a proba array of full shape aligned to np.unique(y)
                unique_sorted = sorted(np.unique(y))
                proba_full = np.zeros((len(val_idx), len(unique_sorted)), dtype=float)
                for i, cls in enumerate(class_order):
                    j = unique_sorted.index(cls)
                    proba_full[:, j] = proba[:, i]
                y_proba[val_idx, :] = proba_full
            except Exception:
                pass
            fold_metrics.append({
                "accuracy": float(accuracy_score(y_va, preds)),
                "f1_macro": float(f1_score(y_va, preds, average='macro', zero_division=0)),
            })

    return y_pred, y_proba, fold_metrics


def fit_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    problem_type: str,
    features: Optional[List[str]],
    cv: int,
    params: Dict,
    test_groups: Optional[List[str]] = None,
    test_size: float = 0.0,
    random_state: int = 42,
    id_cols: Optional[List[str]] = None,
    outdir: str = ".",
) -> Dict[str, object]:
    # Prepare data
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in data.")

    df = df.copy()

    # Drop rows with missing target or group
    df = df[~df[target_col].isna() & ~df[group_col].isna()].reset_index(drop=True)

    if id_cols is None:
        id_cols = []

    # Select features
    if features:
        feature_cols = [c for c in features if c in df.columns]
    else:
        # All columns except target, group, and ids
        exclude_cols = set([target_col, group_col] + id_cols)
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Split numeric vs categorical
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    X = df[feature_cols]
    y_raw = df[target_col]
    groups = df[group_col].astype(str).to_numpy()

    # Encode target for classification
    label_encoder: Optional[LabelEncoder] = None
    class_labels: Optional[List[str]] = None
    if problem_type == 'classification':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        class_labels = [str(cls) for cls in label_encoder.classes_]
    else:
        y = y_raw.to_numpy(dtype=float)

    # Define train/test split by groups
    if test_groups is not None and len(test_groups) > 0:
        test_mask = df[group_col].astype(str).isin(set(test_groups)).to_numpy()
    elif test_size and test_size > 0:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        # Need X and y index array
        idx = np.arange(len(df))
        train_idx, test_idx = next(gss.split(idx, y, groups))
        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[test_idx] = True
    else:
        test_mask = np.zeros(len(df), dtype=bool)

    train_mask = ~test_mask

    X_train, y_train, groups_train = X[train_mask], y[train_mask], groups[train_mask]
    X_test, y_test, groups_test = X[test_mask], y[test_mask], groups[test_mask]

    # Build pipeline
    pipe = build_pipeline(numeric_features, categorical_features, problem_type, params)

    # Grouped CV predictions on training set
    cv_pred, cv_proba, fold_metrics = grouped_cv_predictions(pipe, X_train, y_train, groups_train, cv, problem_type)

    # CV metrics aggregated
    if problem_type == 'regression':
        cv_metrics = compute_regression_metrics(y_train, cv_pred)
    else:
        cv_metrics = compute_classification_metrics(y_train, cv_pred, cv_proba, labels=class_labels)

    # Fit final model on full training data
    pipe.fit(X_train, y_train)

    # Test evaluation (if any test set)
    test_results: Dict[str, object] = {}
    y_test_pred: Optional[np.ndarray] = None
    y_test_proba: Optional[np.ndarray] = None
    if len(X_test) > 0:
        y_test_pred = pipe.predict(X_test)
        if problem_type == 'regression':
            test_results = compute_regression_metrics(y_test, y_test_pred)
        else:
            try:
                y_test_proba = pipe.predict_proba(X_test)
            except Exception:
                y_test_proba = None
            test_results = compute_classification_metrics(y_test, y_test_pred, y_test_proba, labels=class_labels)

    # Save artifacts
    ensure_dir(outdir)

    # Save model
    model_path = os.path.join(outdir, "model.joblib")
    joblib.dump({
        "pipeline": pipe,
        "problem_type": problem_type,
        "target_col": target_col,
        "group_col": group_col,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "label_encoder_classes": (label_encoder.classes_.tolist() if label_encoder is not None else None),
        "class_labels": class_labels,
        "params": params,
        "created_at": timestamp(),
        "cv": cv,
    }, model_path)

    # Save metrics
    results = {
        "problem_type": problem_type,
        "cv_metrics": cv_metrics,
        "fold_metrics": fold_metrics,
        "test_metrics": test_results,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "groups_train_unique": sorted(list(set(groups_train.tolist()))),
        "groups_test_unique": sorted(list(set(groups_test.tolist()))) if len(groups_test) else [],
    }
    save_json(os.path.join(outdir, "metrics.json"), results)

    # Save predictions (CV on train)
    cv_pred_df = pd.DataFrame({
        "index": np.arange(len(y_train)),
        "group": groups_train,
        "y_true": y_train,
        "y_pred": cv_pred,
    })
    if problem_type == 'classification' and cv_proba is not None and class_labels is not None:
        for i, lbl in enumerate(sorted(np.unique(y_train))):
            cv_pred_df[f"proba_{class_labels[i]}"] = cv_proba[:, i]
    cv_pred_df.to_csv(os.path.join(outdir, "cv_predictions.csv"), index=False)

    # Save test predictions (if any)
    if len(X_test) > 0:
        test_pred_df = pd.DataFrame({
            "index": np.arange(len(y_test)),
            "group": groups_test,
            "y_true": y_test,
            "y_pred": y_test_pred,
        })
        if problem_type == 'classification' and y_test_proba is not None and class_labels is not None:
            unique_sorted = sorted(np.unique(y_train))
            for i, lbl in enumerate(unique_sorted):
                test_pred_df[f"proba_{class_labels[i]}"] = y_test_proba[:, i]
        test_pred_df.to_csv(os.path.join(outdir, "test_predictions.csv"), index=False)

    # Plots
    if problem_type == 'regression':
        plot_regression_scatter(y_train, cv_pred, "CV Predictions (Train)", os.path.join(outdir, "cv_pred_scatter.png"))
        if len(X_test) > 0 and y_test_pred is not None:
            plot_regression_scatter(y_test, y_test_pred, "Test Predictions (Holdout)", os.path.join(outdir, "test_pred_scatter.png"))
        plot_residuals_hist(y_train, cv_pred, "Residuals (CV, Train)", os.path.join(outdir, "cv_residuals_hist.png"))
    else:
        # Confusion matrix on train CV
        cm = confusion_matrix(y_train, cv_pred)
        labels_to_plot = class_labels if class_labels is not None else [str(i) for i in sorted(np.unique(y_train))]
        plot_confusion_matrix(cm, labels_to_plot, "Confusion Matrix (CV, Train)", os.path.join(outdir, "cv_confusion_matrix.png"))
        # ROC/PR/Calibration for binary
        if cv_proba is not None and cv_proba.shape[1] == 2:
            plot_roc_curve_binary(y_train, cv_proba, "ROC (CV, Train)", os.path.join(outdir, "cv_roc.png"))
            plot_pr_curve_binary(y_train, cv_proba, "PR (CV, Train)", os.path.join(outdir, "cv_pr.png"))
            plot_calibration_binary(y_train, cv_proba, "Calibration (CV, Train)", os.path.join(outdir, "cv_calibration.png"))
        # Test curves (binary)
        if len(X_test) > 0 and (y_test_proba is not None) and y_test_proba.shape[1] == 2:
            plot_roc_curve_binary(y_test, y_test_proba, "ROC (Test)", os.path.join(outdir, "test_roc.png"))
            plot_pr_curve_binary(y_test, y_test_proba, "PR (Test)", os.path.join(outdir, "test_pr.png"))
            plot_calibration_binary(y_test, y_test_proba, "Calibration (Test)", os.path.join(outdir, "test_calibration.png"))

    # Feature importance from final model
    try:
        plot_feature_importance(pipe, os.path.join(outdir, "feature_importance.png"), top_n=30)
    except Exception:
        pass

    return {
        "model_path": model_path,
        "outdir": outdir,
        "results": results,
        "class_labels": class_labels,
    }


# ------------------------------
# CLI
# ------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradient Boosting with grouped CV by well; saves model, metrics, and plots.")

    # Data
    p.add_argument("--data", type=str, required=True, help="Path to input CSV file")
    p.add_argument("--target", type=str, required=True, help="Target column name")
    p.add_argument("--group-col", type=str, default="well", help="Group column (e.g., well)")
    p.add_argument("--features", type=str, default=None, help="Comma-separated list of feature columns. If omitted, use all non-target/group columns.")
    p.add_argument("--id-cols", type=str, default=None, help="Comma-separated list of ID columns to carry to outputs (optional)")

    # Problem
    p.add_argument("--model-type", type=str, default="auto", choices=["auto", "regression", "classification"], help="Problem type")

    # CV and split
    p.add_argument("--cv", type=int, default=5, help="Number of GroupKFold splits")
    p.add_argument("--test-groups", type=str, default=None, help="Comma-separated group names to use as test holdout")
    p.add_argument("--test-groups-file", type=str, default=None, help="Path to file with one group name per line for test holdout")
    p.add_argument("--test-size", type=float, default=0.0, help="GroupShuffleSplit test size fraction if explicit test groups not provided")

    # Model hyperparameters
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--max-features", type=str, default=None, help="max_features for trees (e.g., sqrt, log2, or int)")

    # Training behavior
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--num-impute", type=str, default="median", choices=["mean", "median", "most_frequent"], help="Numeric imputation strategy")
    p.add_argument("--early-stopping", action="store_true", help="Enable early stopping via n_iter_no_change")
    p.add_argument("--validation-fraction", type=float, default=0.1, help="Validation fraction for early stopping")
    p.add_argument("--tol", type=float, default=1e-4, help="Tolerance for early stopping")

    # Output
    p.add_argument("--outdir", type=str, default=None, help="Output directory to save artifacts")

    args = p.parse_args(argv)

    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Load data
    df = pd.read_csv(args.data)

    # Parse features and ids
    features = [c.strip() for c in args.features.split(",")] if args.features else None
    id_cols = [c.strip() for c in args.id_cols.split(",")] if args.id_cols else []

    # Test groups
    test_groups: Optional[List[str]] = None
    if args.test_groups_file:
        with open(args.test_groups_file, "r", encoding="utf-8") as f:
            test_groups = [line.strip() for line in f if line.strip()]
    elif args.test_groups:
        test_groups = [x.strip() for x in args.test_groups.split(",") if x.strip()]

    # Problem type
    problem_type = infer_problem_type(df[args.target], args.model_type)

    # Output directory
    outdir = args.outdir
    if not outdir:
        outdir = os.path.join("artifacts", f"GB01-4_{problem_type}_{timestamp()}")
    ensure_dir(outdir)

    # Params
    max_features: Optional[object]
    try:
        # try parse int
        max_features = int(args.max_features) if args.max_features is not None else None
    except Exception:
        max_features = args.max_features  # 'sqrt', 'log2', or None

    params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "max_features": max_features,
        "random_state": args.random_state,
        "num_impute": args.num_impute,
        "validation_fraction": args.validation_fraction,
        "n_iter_no_change": 5 if args.early_stopping else None,
        "tol": args.tol,
    }

    # Fit + Evaluate
    result = fit_and_evaluate(
        df=df,
        target_col=args.target,
        group_col=args.group_col,
        problem_type=problem_type,
        features=features,
        cv=args.cv,
        params=params,
        test_groups=test_groups,
        test_size=float(args.test_size) if args.test_size else 0.0,
        random_state=args.random_state,
        id_cols=id_cols,
        outdir=outdir,
    )

    # Final console output
    print(json.dumps({
        "outdir": result["outdir"],
        "model_path": result["model_path"],
        "problem_type": problem_type,
        "metrics_json": os.path.join(result["outdir"], "metrics.json"),
        "cv_predictions_csv": os.path.join(result["outdir"], "cv_predictions.csv"),
        "test_predictions_csv": os.path.join(result["outdir"], "test_predictions.csv") if os.path.exists(os.path.join(result["outdir"], "test_predictions.csv")) else None,
    }, indent=2))


if __name__ == "__main__":
    main()
