import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/targeted_20.csv")
OUTPUT_FILE = Path("data/processed/targeted_20_labeled.csv")

LABELS = {
    "2035102": "other_unclear",

    "311509": "calls",
    "273123": "calls",
    "188639": "calls",
    "386734": "calls",

    "2723667": "purchases_billing",
    "2493140": "purchases_billing",
    "2493135": "purchases_billing",
    "1434221": "purchases_billing",
    "1423459": "purchases_billing",

    "989638": "messages_imessage",
    "989636": "messages_imessage",
    "981949": "messages_imessage",
    "326660": "messages_imessage",
    "187457": "messages_imessage",

    "333090": "device_settings_features",
    "1824652": "device_settings_features",
    "237547": "device_settings_features",
    "1795269": "device_settings_features",
    "711075": "device_settings_features",
}

df = pd.read_csv(INPUT_FILE)

df["customer_tweet_id"] = (
    df["customer_tweet_id"]
    .astype(str)
    .str.replace(".0", "", regex=False)
)

df["final_intent"] = df["customer_tweet_id"].map(LABELS)

df["review_notes"] = ""

df.loc[
    df["customer_tweet_id"] == "2035102",
    "review_notes"
] = "Customer-service escalation/complaint rather than a technical calling issue."

df.loc[
    df["customer_tweet_id"].isin([
        "273123",
        "188639"
    ]),
    "review_notes"
] = "Call-related technical issue."

df.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("TARGETED 20 LABELING COMPLETE")
print("=" * 60)

print(f"Rows: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nIntent distribution:")
print(df["final_intent"].value_counts())

print("\nMissing labels:")
print(df["final_intent"].isna().sum())