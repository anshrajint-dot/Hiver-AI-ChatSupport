import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/applesupport_threads.csv")
OUTPUT_FILE = Path("data/processed/underrepresented_candidates.csv")

df = pd.read_csv(INPUT_FILE)

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# ------------------------------------------------------------
# KEYWORDS
# ------------------------------------------------------------

patterns = {
    "calls": [
        "call",
        "calling",
        "called",
        "phone call",
        "phone calls",
        "make a call",
        "make calls",
        "make a phone call",
        "make phone calls",
        "can't call",
        "cannot call",
        "cant call",
        "can't make",
        "cannot make",
        "unable to call",
        "unable to make calls",
        "calls not working",
        "call not working",
        "receive a call",
        "receive calls",
        "incoming call",
        "incoming calls",
        "outgoing call",
        "outgoing calls",
    ],

    "purchases_billing": [
        "refund",
        "refunded",
        "charge",
        "charged",
        "billing",
        "payment",
        "purchase",
        "purchased",
        "subscription",
        "money back",
        "cancel subscription",
        "apple music refund",
        "app purchase",
        "in app purchase",
    ],

    "messages_imessage": [
        "imessage",
        "iMessage",
        "text message",
        "text messages",
        "messages",
        "message",
        "group chat",
        "groupchat",
        "send texts",
        "can't send messages",
        "cannot send messages",
    ],

    "device_settings_features": [
        "bluetooth",
        "wifi",
        "wi-fi",
        "airplane mode",
        "airplane",
        "siri",
        "alarm",
        "volume",
        "ringer",
        "brightness",
        "keyboard",
        "location services",
        "control center",
        "settings",
    ],
}

# ------------------------------------------------------------
# FIND CANDIDATES
# ------------------------------------------------------------

all_candidates = []

for category, keywords in patterns.items():

    category_matches = []

    for _, row in df.iterrows():

        text = row["customer_text"]
        lower_text = text.lower()

        matched = [
            keyword
            for keyword in keywords
            if keyword.lower() in lower_text
        ]

        if matched:

            candidate = {
                "category_suggestion": category,
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_text": text,
                "support_reply": row["support_reply"],
                "matched_keywords": ", ".join(matched),
            }

            category_matches.append(candidate)

            if len(category_matches) >= 10:
                break

    all_candidates.extend(category_matches)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

result = pd.DataFrame(all_candidates)

result["final_intent"] = ""
result["review_notes"] = ""

result.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# PRINT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("UNDERREPRESENTED INTENT CANDIDATES")
print("=" * 70)

for category in patterns.keys():

    subset = result[
        result["category_suggestion"] == category
    ]

    print("\n")
    print("=" * 70)
    print(f"CATEGORY: {category}")
    print("=" * 70)

    for i, (_, row) in enumerate(subset.iterrows(), start=1):

        print("\n" + "-" * 60)

        print(f"Candidate #{i}")

        print(f"Tweet ID:")
        print(row["customer_tweet_id"])

        print(f"\nCustomer:")
        print(row["customer_text"])

        print(f"\nHistorical AppleSupport reply:")
        print(row["support_reply"])

        print(f"\nMatched keywords:")
        print(row["matched_keywords"])


print("\n" + "=" * 70)
print("AUDIT FILE CREATED")
print("=" * 70)

print(f"Saved to:")
print(OUTPUT_FILE)