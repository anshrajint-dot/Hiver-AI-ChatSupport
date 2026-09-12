import pandas as pd
from pathlib import Path

GOLDEN_FILE = Path("data/processed/golden_set.csv")
TARGETED_FILE = Path("data/processed/targeted_20_labeled.csv")
OUTPUT_FILE = Path("data/processed/golden_set_final.csv")

golden = pd.read_csv(GOLDEN_FILE)
targeted = pd.read_csv(TARGETED_FILE)

# Convert targeted structure to match golden-set structure
targeted_clean = targeted[
    ["customer_tweet_id", "customer_text", "support_reply", "final_intent", "review_notes"]
].copy()

targeted_clean = targeted_clean.rename(
    columns={
        "final_intent": "intent",
        "review_notes": "notes"
    }
)

# Keep only columns present in golden set
targeted_clean = targeted_clean[golden.columns]

# Remove accidental duplicates
existing_ids = set(
    golden["customer_tweet_id"].astype(str)
)

targeted_clean = targeted_clean[
    ~targeted_clean["customer_tweet_id"].astype(str).isin(existing_ids)
].copy()

final = pd.concat(
    [golden, targeted_clean],
    ignore_index=True
)

# Safety checks
assert len(final) == 220, f"Expected 220 rows, got {len(final)}"
assert final["intent"].notna().all(), "Some intents are missing"
assert (final["intent"].astype(str).str.strip() != "").all(), "Blank intents found"

final.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("FINAL GOLDEN SET CREATED")
print("=" * 60)

print(f"Original golden set : {len(golden)}")
print(f"Targeted examples   : {len(targeted_clean)}")
print(f"Final golden set    : {len(final)}")

print("\nIntent distribution:")
print(final["intent"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)