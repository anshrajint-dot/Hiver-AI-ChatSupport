from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent

GOLDEN_FILE = BASE_DIR / "data" / "processed" / "golden_set_final_v2.csv"
MODEL_FILE = BASE_DIR / "data" / "processed" / "intent_model.joblib"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "evaluation_results.csv"


# ============================================================
# LOAD GOLDEN SET
# ============================================================

df = pd.read_csv(GOLDEN_FILE)

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["intent"] = (
    df["intent"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df = df[
    (df["customer_text"] != "") &
    (df["intent"] != "")
].copy()

print("=" * 60)
print("HIVER AI SUPPORT AGENT - EVALUATION")
print("=" * 60)

print(f"\nGolden examples: {len(df)}")


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)

print(f"Model classes: {len(model.classes_)}")


# ============================================================
# MODEL PREDICTIONS
# ============================================================

predictions = model.predict(df["customer_text"])

df["predicted_intent"] = predictions

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(df["customer_text"])
    df["confidence"] = probabilities.max(axis=1)
else:
    df["confidence"] = 0.0


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    df["intent"],
    df["predicted_intent"]
)

macro_f1 = f1_score(
    df["intent"],
    df["predicted_intent"],
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    df["intent"],
    df["predicted_intent"],
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"\nAccuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


# ============================================================
# MAJORITY BASELINE
# ============================================================

majority_class = df["intent"].value_counts().idxmax()

majority_predictions = [
    majority_class
] * len(df)

majority_accuracy = accuracy_score(
    df["intent"],
    majority_predictions
)

majority_macro_f1 = f1_score(
    df["intent"],
    majority_predictions,
    average="macro",
    zero_division=0
)

majority_weighted_f1 = f1_score(
    df["intent"],
    majority_predictions,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("BASELINE 1 - MAJORITY CLASS")
print("=" * 60)

print(f"\nMajority intent: {majority_class}")
print(f"Accuracy       : {majority_accuracy:.4f}")
print(f"Macro F1       : {majority_macro_f1:.4f}")
print(f"Weighted F1    : {majority_weighted_f1:.4f}")


# ============================================================
# RANDOM BASELINE
# ============================================================

# Stratified random baseline:
# randomly samples intents according to their frequency
# in the golden set.

import numpy as np

rng = np.random.default_rng(42)

classes = df["intent"].value_counts(normalize=True)

random_predictions = rng.choice(
    classes.index,
    size=len(df),
    p=classes.values
)

random_accuracy = accuracy_score(
    df["intent"],
    random_predictions
)

random_macro_f1 = f1_score(
    df["intent"],
    random_predictions,
    average="macro",
    zero_division=0
)

random_weighted_f1 = f1_score(
    df["intent"],
    random_predictions,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("BASELINE 2 - STRATIFIED RANDOM")
print("=" * 60)

print(f"\nAccuracy    : {random_accuracy:.4f}")
print(f"Macro F1    : {random_macro_f1:.4f}")
print(f"Weighted F1 : {random_weighted_f1:.4f}")


# ============================================================
# IMPROVEMENT OVER BASELINES
# ============================================================

print("\n" + "=" * 60)
print("IMPROVEMENT")
print("=" * 60)

print(
    f"\nAccuracy vs Majority : "
    f"{accuracy - majority_accuracy:+.4f}"
)

print(
    f"Macro F1 vs Majority : "
    f"{macro_f1 - majority_macro_f1:+.4f}"
)

print(
    f"Accuracy vs Random   : "
    f"{accuracy - random_accuracy:+.4f}"
)

print(
    f"Macro F1 vs Random   : "
    f"{macro_f1 - random_macro_f1:+.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("PER-INTENT PERFORMANCE")
print("=" * 60)

print(
    classification_report(
        df["intent"],
        df["predicted_intent"],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    set(df["intent"]) |
    set(df["predicted_intent"])
)

cm = confusion_matrix(
    df["intent"],
    df["predicted_intent"],
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm_df)


# ============================================================
# SAVE EXAMPLE-LEVEL RESULTS
# ============================================================

df[
    [
        "customer_text",
        "intent",
        "predicted_intent",
        "confidence",
    ]
].to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print(f"\nResults saved to:")
print(OUTPUT_FILE)