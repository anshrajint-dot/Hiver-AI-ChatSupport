from pathlib import Path
import pandas as pd
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent

THREADS_FILE = BASE_DIR / "data" / "processed" / "applesupport_threads.csv"
MODEL_FILE = BASE_DIR / "data" / "processed" / "intent_model.joblib"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "resolution_index.csv"


def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


print("=" * 60)
print("BUILDING INTENT-AWARE RESOLUTION INDEX")
print("=" * 60)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(THREADS_FILE)

print(f"Historical conversations: {len(df)}")

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
)

df["support_reply"] = (
    df["support_reply"]
    .fillna("")
    .astype(str)
)

df = df[
    (df["customer_text"].str.strip() != "") &
    (df["support_reply"].str.strip() != "")
].copy()

# ---------------------------------------------------------
# Load trained intent model
# ---------------------------------------------------------

model = joblib.load(MODEL_FILE)

print(
    f"Intent classes: {len(model.classes_)}"
)

# ---------------------------------------------------------
# Predict historical intents
# ---------------------------------------------------------

print("\nClassifying historical conversations...")

cleaned_text = df["customer_text"].apply(clean_text)

df["intent"] = model.predict(
    cleaned_text
)

# ---------------------------------------------------------
# Clean retrieval text
# ---------------------------------------------------------

df["customer_clean"] = cleaned_text

# ---------------------------------------------------------
# Remove duplicates
# ---------------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset=[
        "customer_clean",
        "support_reply"
    ]
)

duplicates_removed = before - len(df)

# ---------------------------------------------------------
# Save index
# ---------------------------------------------------------

index_columns = [
    "customer_tweet_id",
    "customer_text",
    "support_tweet_id",
    "support_reply",
    "customer_time",
    "support_time",
    "intent",
    "customer_clean"
]

df[index_columns].to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

print()
print("=" * 60)
print("RESOLUTION INDEX CREATED")
print("=" * 60)

print(
    f"Resolution examples: {len(df)}"
)

print(
    f"Duplicates removed : {duplicates_removed}"
)

print("\nIntent distribution:")

print(
    df["intent"].value_counts()
)

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)