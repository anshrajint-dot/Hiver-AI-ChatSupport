import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

MODEL_FILE = Path("data/processed/intent_model.joblib")
GOLDEN_FILE = Path("data/processed/golden_set_final.csv")

METRICS_FILE = Path("data/processed/model_metrics.csv")
PREDICTIONS_FILE = Path("data/processed/model_predictions.csv")
CONFUSION_FILE = Path("data/processed/model_confusion_matrix.csv")


print("=" * 60)
print("TESTING INTENT MODEL")
print("=" * 60)


# =========================================================
# 1. LOAD TRAINED MODEL
# =========================================================

model = joblib.load(MODEL_FILE)

print(f"Loaded model: {MODEL_FILE}")


# =========================================================
# 2. LOAD GOLDEN SET
# =========================================================

golden = pd.read_csv(GOLDEN_FILE)

golden["customer_text"] = (
    golden["customer_text"]
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

X_test = golden["customer_text"]
y_test = golden["intent"]

print(f"Golden examples: {len(golden)}")


# =========================================================
# 3. PREDICTION
# =========================================================

print("\nMaking predictions...")

predictions = model.predict(X_test)


# =========================================================
# 4. CALCULATE METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


# =========================================================
# 5. PRINT RESULTS
# =========================================================

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


# =========================================================
# 6. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# =========================================================
# 7. CONFUSION MATRIX
# =========================================================

labels = sorted(y_test.unique())

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

cm_df.to_csv(
    CONFUSION_FILE
)


# =========================================================
# 8. SAVE PREDICTIONS
# =========================================================

result = golden[
    [
        "customer_tweet_id",
        "customer_text",
        "intent",
    ]
].copy()

result["prediction"] = predictions

result["correct"] = (
    result["intent"] == result["prediction"]
)

result.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# =========================================================
# 9. SAVE METRICS
# =========================================================

metrics = pd.DataFrame([
    {
        "model": "Improved TF-IDF + Logistic Regression",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
])

metrics.to_csv(
    METRICS_FILE,
    index=False
)


# =========================================================
# 10. SHOW WRONG PREDICTIONS
# =========================================================

wrong = result[
    result["correct"] == False
].copy()

print("\n" + "=" * 60)
print(f"INCORRECT PREDICTIONS: {len(wrong)}")
print("=" * 60)

for _, row in wrong.head(15).iterrows():

    print("\nCustomer:")
    print(row["customer_text"])

    print(f"Actual     : {row['intent']}")
    print(f"Prediction : {row['prediction']}")


# =========================================================
# 11. SAVE SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(f"Metrics     : {METRICS_FILE}")
print(f"Predictions : {PREDICTIONS_FILE}")
print(f"Confusion   : {CONFUSION_FILE}")


print("\n" + "=" * 60)
print("MODEL TEST COMPLETE")
print("=" * 60)