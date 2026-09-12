import pandas as pd
import re
from pathlib import Path

INPUT = Path("data/processed/golden_set_final.csv")
OUTPUT = Path("data/processed/golden_audit.csv")

df = pd.read_csv(INPUT)


RULES = {
    "battery_charging": [
        (r"\bbattery\b", 3),
        (r"\bcharging\b", 3),
        (r"\bcharge\b", 2),
        (r"\bdrain(ed|s|ing)?\b", 3),
        (r"\boverheat(ed|ing)?\b", 3),
        (r"swollen battery", 4),
    ],

    "sim_cellular_network": [
        (r"\bsim\b", 4),
        (r"\bcellular\b", 3),
        (r"\b4g\b", 3),
        (r"\blte\b", 3),
        (r"mobile data", 3),
        (r"\bno service\b", 4),
        (r"cell signal", 3),
    ],

    "messages_imessage": [
        (r"\bimessage\b", 4),
        (r"\bsms\b", 3),
        (r"\btext message\b", 3),
        (r"\btext(s|ing)?\b", 2),
        (r"\bmessage(s)?\b", 1),
        (r"not delivered", 3),
        (r"red exclamation", 4),
    ],

    "calls": [
        (r"\bphone call\b", 4),
        (r"\bcalling\b", 3),
        (r"\bcall(s|ing)?\b", 2),
        (r"can't call", 4),
        (r"cannot call", 4),
        (r"answering calls", 4),
    ],

    "purchases_billing": [
        (r"\brefund\b", 4),
        (r"\bpayment\b", 4),
        (r"\bcharged\b", 3),
        (r"\bbilling\b", 4),
        (r"\bsubscription\b", 4),
        (r"\bpurchase(d|s)?\b", 3),
        (r"itunes purchase", 4),
    ],

    "icloud_photos": [
        (r"\bicloud\b", 4),
        (r"icloud drive", 4),
        (r"\bphotos?\b", 2),
        (r"photo stream", 4),
        (r"\bbackup\b", 3),
        (r"sync.*photo", 4),
    ],

    "apple_id_account": [
        (r"apple id", 4),
        (r"\bpassword\b", 3),
        (r"\bsign in\b", 3),
        (r"\blogin\b", 3),
        (r"\baccount\b", 2),
        (r"verification", 3),
    ],

    "apps_app_store": [
        (r"app store", 4),
        (r"\bapps?\b", 2),
        (r"\bdownload\b", 2),
        (r"\binstall\b", 2),
        (r"update.*app", 3),
        (r"app.*update", 3),
    ],

    "device_settings_features": [
        (r"\bwi-?fi\b", 4),
        (r"\bbluetooth\b", 4),
        (r"airplane mode", 4),
        (r"control center", 4),
        (r"\bsettings\b", 2),
        (r"do not disturb", 4),
    ],

    "ios_update": [
        (r"\bios update\b", 4),
        (r"update.*ios", 4),
        (r"downgrade", 4),
        (r"\bbeta\b", 3),
        (r"software update", 4),
        (r"install.*ios", 3),
    ],

    "performance_crash": [
        (r"\bcrash(ed|es|ing)?\b", 4),
        (r"\bfreeze(s|d)?\b", 4),
        (r"\bfreezing\b", 4),
        (r"\blag(ging|gy)?\b", 3),
        (r"\bslow\b", 3),
        (r"\brestart(s|ed|ing)?\b", 3),
        (r"\breboot(s|ed|ing)?\b", 3),
        (r"\bglitch(y|ing)?\b", 3),
    ],
}


def score_text(text):
    text = str(text).lower()
    scores = {}

    for intent, patterns in RULES.items():
        score = 0

        for pattern, weight in patterns:
            if re.search(pattern, text):
                score += weight

        scores[intent] = score

    return scores


results = []

for _, row in df.iterrows():

    text = str(row["customer_text"])
    current = str(row["intent"])

    scores = score_text(text)

    suggested = max(scores, key=scores.get)
    suggested_score = scores[suggested]
    current_score = scores.get(current, 0)

    # Strong evidence + meaningful gap
    needs_review = (
        suggested != current
        and suggested_score >= 3
        and suggested_score >= current_score + 2
    )

    if len(text.strip()) < 10:
        needs_review = True
        reason = "Very short/unclear customer message"
    elif needs_review:
        reason = (
            f"Strong keyword evidence for {suggested} "
            f"({suggested_score}) vs current label "
            f"{current} ({current_score})"
        )
    else:
        reason = ""

    results.append({
        "customer_tweet_id": row["customer_tweet_id"],
        "customer_text": text,
        "current_intent": current,
        "suggested_intent": suggested,
        "suggested_score": suggested_score,
        "current_score": current_score,
        "needs_review": needs_review,
        "review_reason": reason,
    })


audit = pd.DataFrame(results)

# Suspicious examples first
audit = audit.sort_values(
    ["needs_review", "suggested_score"],
    ascending=[False, False]
)

audit.to_csv(OUTPUT, index=False)

print("=" * 60)
print("GOLDEN SET AUDIT COMPLETE")
print("=" * 60)
print(f"Total examples : {len(audit)}")
print(f"Needs review   : {audit['needs_review'].sum()}")
print(f"Looks OK       : {(~audit['needs_review']).sum()}")
print(f"Saved to       : {OUTPUT}")
print("=" * 60)

print("\nTOP SUSPICIOUS EXAMPLES:\n")

review = audit[audit["needs_review"]].head(30)

for _, row in review.iterrows():
    print("-" * 60)
    print("ID:", row["customer_tweet_id"])
    print("TEXT:", row["customer_text"])
    print("CURRENT:", row["current_intent"])
    print("SUGGESTED:", row["suggested_intent"])
    print("REASON:", row["review_reason"])