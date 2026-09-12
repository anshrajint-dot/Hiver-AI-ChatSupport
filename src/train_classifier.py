import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


INPUT_FILE = "data/processed/golden_set.csv"
OUTPUT_DIR = "data/processed"


def main():

    print("Loading golden set...")

    # ------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"Total rows loaded: {len(df):,}")

    # ------------------------------------------------
    # CLEAN INTENT COLUMN
    # ------------------------------------------------

    df["intent"] = (
        df["intent"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Keep only labelled examples
    df = df[df["intent"] != ""].copy()

    print(f"Total labeled examples: {len(df):,}")

    # ------------------------------------------------
    # CLEAN CUSTOMER TEXT
    # ------------------------------------------------

    df["customer_text"] = (
        df["customer_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove empty customer messages
    df = df[df["customer_text"] != ""].copy()

    X = df["customer_text"]
    y = df["intent"]

    # ------------------------------------------------
    # INTENT DISTRIBUTION
    # ------------------------------------------------

    print("\n===================================")
    print("INTENT DISTRIBUTION")
    print("===================================")

    print(y.value_counts())

    # ------------------------------------------------
    # TRAIN / TEST SPLIT
    # ------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\n===================================")
    print("DATA SPLIT")
    print("===================================")

    print(f"Training examples: {len(X_train)}")
    print(f"Testing examples:  {len(X_test)}")

    # =================================================
    # BASELINE 1
    # MAJORITY CLASS
    # =================================================

    majority_intent = y_train.value_counts().idxmax()

    baseline_predictions = [
        majority_intent
        for _ in range(len(y_test))
    ]

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    baseline_macro_f1 = f1_score(
        y_test,
        baseline_predictions,
        average="macro",
        zero_division=0
    )

    print("\n===================================")
    print("BASELINE 1 — MAJORITY CLASS")
    print("===================================")

    print(f"Majority intent: {majority_intent}")
    print(f"Accuracy: {baseline_accuracy:.4f}")
    print(f"Macro F1: {baseline_macro_f1:.4f}")

    # =================================================
    # BASELINE 2
    # TF-IDF + LOGISTIC REGRESSION
    # =================================================

    print("\n===================================")
    print("BASELINE 2 — TF-IDF + LOGISTIC REGRESSION")
    print("===================================")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    print(
        f"TF-IDF train shape: {X_train_tfidf.shape}"
    )

    # ------------------------------------------------
    # TRAIN MODEL
    # ------------------------------------------------

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    print("\nTraining Logistic Regression...")

    classifier.fit(
        X_train_tfidf,
        y_train
    )

    # ------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------

    predictions = classifier.predict(
        X_test_tfidf
    )

    # ------------------------------------------------
    # METRICS
    # ------------------------------------------------

    ml_accuracy = accuracy_score(
        y_test,
        predictions
    )

    ml_macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    ml_weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n===================================")
    print("TF-IDF + LOGISTIC REGRESSION RESULTS")
    print("===================================")

    print(f"Accuracy:    {ml_accuracy:.4f}")
    print(f"Macro F1:    {ml_macro_f1:.4f}")
    print(f"Weighted F1: {ml_weighted_f1:.4f}")

    # ------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------

    print("\n===================================")
    print("CLASSIFICATION REPORT")
    print("===================================")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)

    # ------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------

    labels = sorted(y.unique())

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    cm_file = (
        f"{OUTPUT_DIR}/confusion_matrix.csv"
    )

    cm_df.to_csv(cm_file)

    # ------------------------------------------------
    # SAVE PREDICTIONS
    # ------------------------------------------------

    results = pd.DataFrame({
        "customer_text": X_test.values,
        "true_intent": y_test.values,
        "predicted_intent": predictions
    })

    predictions_file = (
        f"{OUTPUT_DIR}/classifier_predictions.csv"
    )

    results.to_csv(
        predictions_file,
        index=False
    )

    # ------------------------------------------------
    # SAVE METRICS
    # ------------------------------------------------

    metrics = pd.DataFrame([
        {
            "model": "Majority Class",
            "accuracy": baseline_accuracy,
            "macro_f1": baseline_macro_f1
        },
        {
            "model": "TF-IDF + Logistic Regression",
            "accuracy": ml_accuracy,
            "macro_f1": ml_macro_f1,
            "weighted_f1": ml_weighted_f1
        }
    ])

    metrics_file = (
        f"{OUTPUT_DIR}/classifier_metrics.csv"
    )

    metrics.to_csv(
        metrics_file,
        index=False
    )

    # ------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------

    print("\n===================================")
    print("CLASSIFIER TRAINING COMPLETE")
    print("===================================")

    print(f"Confusion matrix:")
    print(cm_file)

    print(f"\nPredictions:")
    print(predictions_file)

    print(f"\nMetrics:")
    print(metrics_file)

    print("\nDone!")


if __name__ == "__main__":
    main()