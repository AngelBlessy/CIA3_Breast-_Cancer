"""
Standalone training pipeline for the Breast Cancer Risk Screening model.

Extracted from CIA_3_ML_breast_cancer.ipynb (Sections 5-13) into a plain script.
Running this file reproduces the same preprocessing, model pipelines, tuning,
stacking ensemble, and saved artifacts as the notebook.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def load_data(csv_path="breast_cancer.csv", random_state=55):
    df = pd.read_csv(csv_path)

    # Binary target from the raw diagnosis label.
    df["target"] = (df["diagnosis"] == "malignant").astype(int)

    # Simulate realistic measurement noise (Section 5).
    feat_cols = [c for c in df.columns if c not in ("diagnosis", "target")]
    rng = np.random.RandomState(random_state)
    for c in feat_cols:
        std = df[c].std()
        df[c] = (df[c] + rng.normal(0, 0.6 * std, size=len(df))).clip(lower=0)

    # Feature engineering (Section 7).
    df["radius_growth_ratio"] = df["worst radius"] / df["mean radius"].replace(0, 1e-6)
    q1, q2 = df["mean radius"].quantile([1 / 3, 2 / 3])
    df["size_category"] = pd.cut(
        df["mean radius"], bins=[0, q1, q2, 1000], labels=["Small", "Medium", "Large"]
    )

    return df


def build_preprocessor(X):
    numeric_features = [c for c in X.columns if c != "size_category"]
    categorical_features = ["size_category"]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def evaluate(name, model, X_tr, y_tr, X_te, y_te, results, confmats, threshold=0.5, fit=True):
    if fit:
        model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    results[name] = {
        "ROC-AUC": roc_auc_score(y_te, y_proba),
        "Accuracy": accuracy_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred),
        "Recall": recall_score(y_te, y_pred),
    }
    confmats[name] = confusion_matrix(y_te, y_pred)
    print(f"--- {name} (threshold={threshold}) ---")
    for metric, val in results[name].items():
        print(f"{metric}: {val:.4f}")
    print("Confusion matrix:\n", confmats[name])
    return model


def build_and_train_pipelines(X_train, y_train, X_test, y_test, preprocessor):
    results, confmats = {}, {}

    # 10.1 Baseline - Logistic Regression.
    logreg_pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    logreg_pipe = evaluate(
        "Baseline (LogReg)", logreg_pipe, X_train, y_train, X_test, y_test, results, confmats
    )

    # 10.2 Bagging - Random Forest (tuned).
    rf_pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )
    rf_param_dist = {
        "clf__n_estimators": [100, 200, 300, 500, 800],
        "clf__max_depth": [3, 4, 5, 6, 8, 10, None],
        "clf__min_samples_leaf": [1, 2, 3, 4],
        "clf__max_features": ["sqrt", "log2", 0.5, 0.7],
        "clf__criterion": ["gini", "entropy"],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_param_dist, n_iter=40, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    print("Best RF params:", rf_search.best_params_)
    evaluate(
        "Bagging (Random Forest)",
        rf_search.best_estimator_,
        X_train,
        y_train,
        X_test,
        y_test,
        results,
        confmats,
        threshold=0.38,
        fit=False,
    )

    # 10.3 Boosting - XGBoost (tuned).
    xgb_pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clf", XGBClassifier(eval_metric="logloss", random_state=42)),
        ]
    )
    xgb_param_dist = {
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [2, 3, 4, 5],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb_pipe, xgb_param_dist, n_iter=15, cv=5, scoring="roc_auc", random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    print("Best XGB params:", xgb_search.best_params_)
    best_xgb = evaluate(
        "Boosting (XGBoost)",
        xgb_search.best_estimator_,
        X_train,
        y_train,
        X_test,
        y_test,
        results,
        confmats,
        threshold=0.2,
        fit=False,
    )

    # 10.4 Stacking Ensemble (LogReg + RF + XGBoost, LogReg meta-learner).
    stacking_model = StackingClassifier(
        estimators=[
            ("logreg", logreg_pipe),
            ("rf", rf_search.best_estimator_),
            ("xgb", xgb_search.best_estimator_),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1,
    )
    stacking_model = evaluate(
        "Stacking Ensemble",
        stacking_model,
        X_train,
        y_train,
        X_test,
        y_test,
        results,
        confmats,
        threshold=0.25,
    )

    return {
        "logreg_pipe": logreg_pipe,
        "rf_best": rf_search.best_estimator_,
        "xgb_best": best_xgb,
        "stacking_model": stacking_model,
        "results": results,
        "confmats": confmats,
    }


def save_artifacts(models, X_train, numeric_features, categorical_features):
    joblib.dump(models["logreg_pipe"], "bc_model_logreg.pkl")
    joblib.dump(models["rf_best"], "bc_model_rf.pkl")
    joblib.dump(models["xgb_best"], "bc_model_xgb_for_shap.pkl")
    joblib.dump(models["stacking_model"], "bc_model_stacking.pkl")

    metadata = {
        "numeric_features": {},
        "categorical_features": {},
        "thresholds": {"rf": 0.38, "xgb": 0.2, "stacking": 0.25},
    }
    for c in numeric_features:
        metadata["numeric_features"][c] = {
            "min": float(X_train[c].min()),
            "max": float(X_train[c].max()),
            "median": float(X_train[c].median()),
        }
    for c in categorical_features:
        metadata["categorical_features"][c] = sorted(X_train[c].astype(str).unique().tolist())

    with open("bc_feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(
        "Saved bc_model_logreg.pkl, bc_model_rf.pkl, bc_model_xgb_for_shap.pkl, "
        "bc_model_stacking.pkl, bc_feature_metadata.json"
    )


def main():
    df = load_data()

    X = df.drop(columns=["diagnosis", "target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    models = build_and_train_pipelines(X_train, y_train, X_test, y_test, preprocessor)

    comparison_df = pd.DataFrame(models["results"]).T.round(4)
    print("\nModel Comparison - same held-out test set")
    print(comparison_df)

    save_artifacts(models, X_train, numeric_features, categorical_features)


if __name__ == "__main__":
    main()
