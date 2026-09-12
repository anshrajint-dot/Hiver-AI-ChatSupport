import pandas as pd
from pathlib import Path

CANDIDATE_FILE = Path("data/processed/underrepresented_candidates.csv")
OUTPUT_FILE = Path("data/processed/targeted_20.csv")

selected_ids = [
    # calls
    188639,
    386734,
    311509,
    273123,
    2035102,

    # purchases_billing
    2723667,
    2493140,
    2493135,
    1434221,
    1423459,

    # messages_imessage
    989638,
    989636,
    981949,
    326660,
    187457,

    # device_settings_features
    333090,
    1824652,
    237547,
    1795269,
    711075,
]

df = pd.read_csv(CANDIDATE_FILE)

df["customer_tweet_id"] = (
    df["customer_tweet_id"]
    .astype(str)
    .str.replace(".0", "", regex=False)
)

selected_ids = [str(x) for x in selected_ids]

selected = df[df["customer_tweet_id"].isin(selected_ids)].copy()

# Remove duplicates just in case
selected = selected.drop_duplicates(subset=["customer_tweet_id"])

# Keep a clean labelling structure
selected["final_intent"] = ""
selected["review_notes"] = ""

selected.to_csv(OUTPUT_FILE, index=False)

print("=" * 60)
print("TARGETED GOLDEN CANDIDATES")
print("=" * 60)
print(f"Selected: {len(selected)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nCandidates:")
for i, row in enumerate(selected.itertuples(), 1):
    print(f"\n{i}. Tweet ID: {row.customer_tweet_id}")
    print(f"Suggested category: {row.category_suggestion}")
    print(f"Customer: {row.customer_text}")

if len(selected) != 20:
    print("\nWARNING: Expected 20 candidates.")
    print("Check whether any selected Tweet IDs were not found.")
else:
    print("\nAll 20 candidates successfully selected.")