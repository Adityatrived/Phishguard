"""Milestone 4: train and compare candidate models, export the winner.

Trains Logistic Regression, Random Forest, and XGBoost on the feature
matrix from ml/features.py, evaluates each on a held-out test set, and
exports the best-performing model (highest F1) to ml/models/model.joblib
so the backend can load it and run inference without ever training.

Run with: .venv/bin/python -m ml.train   (must run as a module, from the
repo root, so `ml` resolves as a package the same way it will for the
backend later)
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.features import FEATURE_NAMES, build_feature_matrix

DATA_PATH = Path(__file__).parent / "data" / "processed" / "dataset.csv"
MODELS_DIR = Path(__file__).parent / "models"
REPORT_PATH = Path(__file__).parent / "MODEL_REPORT.md"
SEED = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = build_feature_matrix(df["url"])
    y = df["label"]
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def candidate_models():
    return {
        # Logistic Regression needs scaled inputs (features here range from
        # 0/1 flags to url_length in the hundreds) -- tree-based models
        # don't care about feature scale, so only this one gets a scaler.
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
            ]
        ),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, eval_metric="logloss", random_state=SEED, n_jobs=-1
        ),
    }


# Randomized search spaces for the two tree models -- Logistic Regression
# isn't competitive on this feature set (see MODEL_REPORT.md) so it's left
# untuned as the baseline comparison point.
SEARCH_SPACES = {
    "Random Forest": {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    },
    "XGBoost": {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    },
}
N_SEARCH_ITER = 40


def tune(name, model, X_train, y_train):
    """Randomized hyperparameter search over SEARCH_SPACES, scored on F1
    via 5-fold CV. Returns the best estimator found, refit on the full
    training set. Skipped for models with no search space (e.g. Logistic
    Regression, which isn't competitive here anyway)."""
    space = SEARCH_SPACES.get(name)
    if space is None:
        model.fit(X_train, y_train)
        return model

    search = RandomizedSearchCV(
        model,
        param_distributions=space,
        n_iter=N_SEARCH_ITER,
        scoring="f1",
        cv=5,
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  best params for {name}: {search.best_params_} (cv f1={search.best_score_:.4f})")
    return search.best_estimator_


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        # [[TN, FP], [FN, TP]] since labels are sorted [0, 1]
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def top_features(model, n=8):
    """Feature importances (trees) or |coefficient| (logistic regression)."""
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(clf, "feature_importances_"):
        scores = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        scores = abs(clf.coef_[0])
    else:
        return []
    ranked = sorted(zip(FEATURE_NAMES, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:n]


def write_report(results, winner_name, winner_features):
    lines = [
        "# PhishGuard — Model Comparison Report",
        "",
        f"Trained on `{DATA_PATH.name}` (130k URLs, balanced), 80/20 "
        f"stratified train/test split, seed={SEED}.",
        "",
        "| Model | Accuracy | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for name, metrics in results.items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} "
            f"| {metrics['recall']:.4f} | {metrics['f1']:.4f} |"
        )
    lines += [
        "",
        f"**Winner: {winner_name}** (highest F1 on the held-out test set).",
        "",
        f"Confusion matrix `[[TN, FP], [FN, TP]]`: " f"{results[winner_name]['confusion_matrix']}",
        "",
        "Top features for the winning model:",
        "",
    ]
    for feat, score in winner_features:
        lines.append(f"- `{feat}`: {score:.4f}")
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main():
    X_train, X_test, y_train, y_test = load_data()
    models = candidate_models()

    results = {}
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...")
        fitted_model = tune(name, model, X_train, y_train)
        fitted[name] = fitted_model
        results[name] = evaluate(fitted_model, X_test, y_test)

    print("\nResults:")
    for name, metrics in results.items():
        print(
            f"  {name:20s} acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )

    winner_name = max(results, key=lambda name: results[name]["f1"])
    winner_model = fitted[winner_name]
    print(f"\nWinner: {winner_name} (f1={results[winner_name]['f1']:.4f})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(winner_model, model_path)
    print(f"Saved winning model to {model_path}")

    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps({"winner": winner_name, "results": results}, indent=2))

    winner_features = top_features(winner_model)
    write_report(results, winner_name, winner_features)
    print(f"Wrote comparison report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
