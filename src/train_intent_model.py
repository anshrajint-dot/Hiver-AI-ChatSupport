import pandas as pd
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TRAIN_FILE = Path("data/processed/training_data.csv")
MODEL_FILE = Path("data/processed/intent_model.joblib")


print("=" * 60)
print("TRAINING INTENT MODEL")
print("=" * 60)

# ---------------------------------------------------------
# Load training data
# ---------------------------------------------------------

df = pd.read_csv(TRAIN_FILE)

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

print(f"Training examples: {len(df)}")
print(f"Intent classes: {df['intent'].nunique()}")

print("\nTraining distribution:")
print(df["intent"].value_counts())

# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    max_features=50000,
)

# ---------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------

classifier = LogisticRegression(
    max_iter=1500,
    class_weight="balanced",
    C=3.0,
    solver="liblinear",
    random_state=42,
)

model = Pipeline([
    ("tfidf", vectorizer),
    ("classifier", classifier),
])

print("\nTraining model...")

model.fit(
    df["customer_text"],
    df["intent"]
)

# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

joblib.dump(
    model,
    MODEL_FILE
)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)

print(f"Saved model: {MODEL_FILE}")

print("\nClasses:")
for cls in classifier.classes_:
    print(f"- {cls}")

print("\nNext step:")
print("Run src/test_intent_model.py")