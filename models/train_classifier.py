"""
train_classifier.py
--------------------
Week 3 Steps 4-6: train a simple TF-IDF classifier, compare it against
other simple models, and evaluate properly (accuracy/precision/recall/F1 +
confusion matrix).

Run: python models/train_classifier.py
Reads:  data/dataset.csv          (text,label)
Writes: models/tfidf_vectorizer.joblib
        models/best_model.joblib
        models/model_metadata.json
        models/evaluation_report.json
        models/confusion_matrix.png
        models/model_comparison.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocess import clean_text, normalize_for_classification  # noqa: E402

DATA_PATH = ROOT / "data" / "dataset.csv"
MODELS_DIR = ROOT / "models"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_dataset.py` first "
            "(or drop in your own text,label CSV of real documents)."
        )
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["text"].apply(lambda t: normalize_for_classification(clean_text(t)))
    return df


def train_and_compare(df: pd.DataFrame) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "linear_svm": LinearSVC(),
        "naive_bayes": MultinomialNB(),
    }

    results = {}
    trained_models = {}

    for name, model in candidates.items():
        model.fit(x_train_vec, y_train)
        preds = model.predict(x_test_vec)

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
            "classification_report": classification_report(y_test, preds, zero_division=0, output_dict=True),
        }
        trained_models[name] = model
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, zero_division=0))

    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    best_model = trained_models[best_name]
    best_preds = best_model.predict(x_test_vec)

    print(f"\nBest model by macro F1: {best_name} "
          f"(F1={results[best_name]['f1_macro']:.3f})")

    # --- confusion matrix for the best model ---
    labels_sorted = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, best_preds, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels_sorted).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- bar chart comparing all models on macro F1 ---
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    names = list(results.keys())
    f1_scores = [results[n]["f1_macro"] for n in names]
    bars = ax2.bar(names, f1_scores, color=["#4C72B0", "#55A868", "#C44E52"])
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Macro F1-score")
    ax2.set_title("Model comparison")
    for bar, score in zip(bars, f1_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.2f}", ha="center")
    plt.xticks(rotation=15)
    fig2.tight_layout()
    fig2.savefig(MODELS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig2)

    # --- persist everything the app needs at inference time ---
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    metadata = {
        "best_model": best_name,
        "labels": labels_sorted,
        "test_size": len(y_test),
        "train_size": len(y_train),
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    eval_report = {
        "best_model": best_name,
        "comparison": {n: {k: v for k, v in r.items() if k != "classification_report"}
                       for n, r in results.items()},
        "best_model_full_report": results[best_name]["classification_report"],
        "confusion_matrix": {
            "labels": labels_sorted,
            "matrix": cm.tolist(),
        },
    }
    (MODELS_DIR / "evaluation_report.json").write_text(json.dumps(eval_report, indent=2))

    return eval_report


def main() -> None:
    df = load_dataset()
    print(f"Loaded {len(df)} rows. Class balance:\n{df['label'].value_counts()}\n")
    train_and_compare(df)
    print(f"\nSaved model, vectorizer, metadata, evaluation report, and plots to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
