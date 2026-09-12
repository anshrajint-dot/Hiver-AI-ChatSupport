import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

TRAIN_FILE = Path("data/processed/training_data.csv")
GOLDEN_FILE = Path("data/processed/golden_set_final.csv")

METRICS_FILE = Path("data/processed/baseline_metrics.csv")
PREDICTIONS_FILE = Path("data/processed/baseline_predictions.csv")
CONFUSION_FILE = Path("data/processed/baseline_confusion_matrix.csv")


print("=" * 60)
print("BASELINE EVALUATION")
print("=" * 60)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
golden = pd.read_csv(GOLDEN_FILE)

train["customer_text"] = (
    train["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

golden["customer_text"] = (
    golden["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

train["intent"] = (
    train["intent"]
    .fillna("")
    .astype(str)
    .str.strip()
)

golden["intent"] = (
    golden["intent"]
    .fillna("")
    .astype(str)
    .str.strip()
)

print(f"Training examples : {len(train)}")
print(f"Golden examples   : {len(golden)}")

X_train = train["customer_text"]
y_train = train["intent"]

X_test = golden["customer_text"]
y_test = golden["intent"]

# ---------------------------------------------------------
# Baseline 1: Majority Class
# ---------------------------------------------------------

majority_class = y_train.value_counts().idxmax()

majority_predictions = [majority_class] * len(y_test)

majority_accuracy = accuracy_score(
    y_test,
    majority_predictions
)

majority_macro_f1 = f1_score(
    y_test,
    majority_predictions,
    average="macro",
    zero_division=0
)

majority_weighted_f1 = f1_score(
    y_test,
    majority_predictions,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("BASELINE 1 — MAJORITY CLASS")
print("=" * 60)

print(f"Majority class : {majority_class}")
print(f"Accuracy       : {majority_accuracy:.4f}")
print(f"Macro F1       : {majority_macro_f1:.4f}")
print(f"Weighted F1    : {majority_weighted_f1:.4f}")

# ---------------------------------------------------------
# Baseline 2: TF-IDF + Logistic Regression
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE 2 — TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=30000,
    sublinear_tf=True,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

classifier.fit(
    X_train_tfidf,
    y_train
)

lr_predictions = classifier.predict(X_test_tfidf)

lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)

lr_macro_f1 = f1_score(
    y_test,
    lr_predictions,
    average="macro",
    zero_division=0
)

lr_weighted_f1 = f1_score(
    y_test,
    lr_predictions,
    average="weighted",
    zero_division=0
)

print(f"Accuracy       : {lr_accuracy:.4f}")
print(f"Macro F1       : {lr_macro_f1:.4f}")
print(f"Weighted F1    : {lr_weighted_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        lr_predictions,
        zero_division=0
    )
)

# ---------------------------------------------------------
# Save metrics
# ---------------------------------------------------------

metrics = pd.DataFrame([
    {
        "baseline": "Majority Class",
        "accuracy": majority_accuracy,
        "macro_f1": majority_macro_f1,
        "weighted_f1": majority_weighted_f1,
    },
    {
        "baseline": "TF-IDF + Logistic Regression",
        "accuracy": lr_accuracy,
        "macro_f1": lr_macro_f1,
        "weighted_f1": lr_weighted_f1,
    },
])

metrics.to_csv(
    METRICS_FILE,
    index=False
)

# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

predictions = golden[
    [
        "customer_tweet_id",
        "customer_text",
        "intent",
    ]
].copy()

predictions["majority_prediction"] = majority_predictions
predictions["tfidf_lr_prediction"] = lr_predictions

predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)

# ---------------------------------------------------------
# Save confusion matrix
# ---------------------------------------------------------

labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    lr_predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(
    CONFUSION_FILE
)

# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(METRICS_FILE)
print(PREDICTIONS_FILE)
print(CONFUSION_FILE)

print("\nBaseline comparison:")
print(metrics.to_string(index=False))

print("\n" + "=" * 60)
print("BASELINE EVALUATION COMPLETE")
print("=" * 60)