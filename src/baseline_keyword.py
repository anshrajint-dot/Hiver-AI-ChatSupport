from pathlib import Path
import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_FILE = BASE_DIR / "data" / "processed" / "golden_set_final_v2.csv"


# ============================================================
# KEYWORD BASELINE
# ============================================================

KEYWORDS = {
    "battery_charging": [
        "battery",
        "charging",
        "charge",
        "drain",
        "dies",
        "dying",
    ],

    "apps_app_store": [
        "app store",
        "download app",
        "download apps",
        "can't download",
        "cannot download",
        "app won't download",
        "apps",
    ],

    "purchases_billing": [
        "charged",
        "charge",
        "billing",
        "payment",
        "purchase",
        "refund",
        "subscription",
        "money",
        "invoice",
    ],

    "messages_imessage": [
        "imessage",
        "message",
        "messages",
        "text",
        "texting",
        "sms",
    ],

    "calls": [
        "call",
        "calling",
        "phone call",
        "can't call",
        "cannot call",
        "unable to call",
    ],

    "sim_cellular_network": [
        "no service",
        "network",
        "signal",
        "carrier",
        "sim",
        "cellular",
        "searching",
    ],

    "apple_id_account": [
        "apple id",
        "password",
        "sign in",
        "signin",
        "login",
        "log in",
        "account",
    ],

    "icloud_photos": [
        "icloud",
        "photos",
        "photo",
        "pictures",
        "photostream",
    ],

    "ios_update": [
        "ios update",
        "software update",
        "update ios",
        "ios 11",
        "ios 12",
        "ios 13",
        "ios 14",
        "update",
    ],

    "device_settings_features": [
        "setting",
        "settings",
        "bluetooth",
        "wifi",
        "wi-fi",
        "notification",
        "notifications",
        "screen",
        "keyboard",
        "camera",
        "location",
        "feature",
    ],

    "performance_crash": [
        "crash",
        "crashed",
        "freezing",
        "freeze",
        "frozen",
        "slow",
        "lag",
        "bug",
        "glitch",
        "not working",
        "doesn't work",
        "doesnt work",
        "stopped working",
    ],
}


# ============================================================
# CLEAN TEXT
# ============================================================

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


# ============================================================
# KEYWORD CLASSIFIER
# ============================================================

def predict_intent(text):

    text = clean_text(text)

    scores = {
        intent: 0
        for intent in KEYWORDS
    }

    for intent, keywords in KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:
                scores[intent] += 1

    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_intent]

    # No keyword match
    if best_score == 0:
        return "other_unclear"

    return best_intent


# ============================================================
# LOAD GOLDEN SET
# ============================================================

df = pd.read_csv(
    GOLDEN_FILE
)

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


# ============================================================
# PREDICTIONS
# ============================================================

df["prediction"] = df[
    "customer_text"
].apply(
    predict_intent
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    df["intent"],
    df["prediction"]
)

macro_f1 = f1_score(
    df["intent"],
    df["prediction"],
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    df["intent"],
    df["prediction"],
    average="weighted",
    zero_division=0
)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 60)
print("KEYWORD BASELINE EVALUATION")
print("=" * 60)

print(
    f"\nGolden examples: {len(df)}"
)

print("\nRESULTS")
print("-" * 60)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Macro F1    : {macro_f1:.4f}"
)

print(
    f"Weighted F1 : {weighted_f1:.4f}"
)


print("\nCLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        df["intent"],
        df["prediction"],
        zero_division=0
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "keyword_baseline_results.csv"
)

df[
    [
        "customer_text",
        "intent",
        "prediction",
    ]
].to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)