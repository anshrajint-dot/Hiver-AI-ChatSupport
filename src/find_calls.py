import pandas as pd
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/processed/applesupport_threads.csv")
MAX_CANDIDATES = 20

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING APPLESUPPORT THREADS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["support_reply"] = (
    df["support_reply"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# ============================================================
# CALLS KEYWORDS
# ============================================================

patterns = [
    "can't call",
    "cannot call",
    "cant call",
    "can't make a call",
    "cannot make a call",
    "can't make calls",
    "cannot make calls",
    "unable to call",
    "unable to make calls",
    "calls not working",
    "call not working",
    "phone calls",
    "phone call",
    "make a call",
    "make calls",
    "receive calls",
    "receive a call",
    "incoming calls",
    "incoming call",
    "outgoing calls",
    "outgoing call",
]

# ============================================================
# FIND CANDIDATES
# ============================================================

matches = []

for _, row in df.iterrows():

    text = row["customer_text"]
    lower_text = text.lower()

    matched_patterns = [
        pattern
        for pattern in patterns
        if pattern.lower() in lower_text
    ]

    if matched_patterns:

        matches.append(
            {
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": text,
                "support_reply": row["support_reply"],
                "matched": ", ".join(matched_patterns),
            }
        )

    if len(matches) >= MAX_CANDIDATES:
        break

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CALLS CANDIDATES")
print("=" * 60)

print(f"\nFound: {len(matches)}")

for i, row in enumerate(matches, start=1):

    print("\n" + "-" * 60)
    print(f"Candidate #{i}")

    print(f"\nTweet ID:")
    print(row["customer_tweet_id"])

    print(f"\nCustomer:")
    print(row["customer_text"])

    print(f"\nHistorical reply:")
    print(row["support_reply"])

    print(f"\nMatched:")
    print(row["matched"])

print("\n" + "=" * 60)
print("END")
print("=" * 60)