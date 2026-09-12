import pandas as pd
from pathlib import Path
import re


# ============================================================
# FILE PATHS
# ============================================================

THREADS_FILE = Path(
    "data/processed/applesupport_threads.csv"
)

# IMPORTANT:
# Use the corrected 220-example golden set.
GOLDEN_FILE = Path(
    "data/processed/golden_set_final_v2.csv"
)

OUTPUT_FILE = Path(
    "data/processed/training_data.csv"
)


# ============================================================
# WEAK-LABEL RULES
# ============================================================
# These rules create noisy/weak training labels.
# Golden evaluation examples are NEVER used for training.
#
# Rules are intentionally conservative:
# - Only strong phrases are used.
# - Ambiguous examples are discarded.
# - "other_unclear" is NOT weakly generated.
# ============================================================

RULES = {

    # --------------------------------------------------------
    # CALLS
    # --------------------------------------------------------
    "calls": [
        "can't call",
        "cannot call",
        "cant call",
        "can't make a call",
        "cannot make a call",
        "cant make a call",
        "can't make calls",
        "cannot make calls",
        "cant make calls",
        "unable to call",
        "unable to make calls",
        "make a phone call",
        "phone calls not working",
        "phone call not working",
        "incoming calls",
        "outgoing calls",
        "receive calls",
        "can't receive calls",
        "cannot receive calls",
        "can't answer calls",
        "cannot answer calls",
        "calls not working",
        "call not working",
    ],


    # --------------------------------------------------------
    # BATTERY / CHARGING
    # --------------------------------------------------------
    "battery_charging": [
        "battery drain",
        "battery draining",
        "battery drains",
        "battery dies",
        "battery dying",
        "battery life",
        "battery percentage",
        "battery percent",
        "battery drops",
        "battery dropping",
        "battery draining fast",
        "battery dies quickly",
        "battery dying quickly",
        "battery life is bad",
        "battery life is poor",
        "won't charge",
        "wont charge",
        "cannot charge",
        "can't charge",
        "not charging",
        "charging problem",
        "charging issue",
        "charging doesn't work",
        "charging does not work",
        "charger not working",
        "overheating",
        "phone overheating",
        "iphone overheating",
        "battery overheating",
        "battery swollen",
    ],


    # --------------------------------------------------------
    # IOS UPDATE
    # --------------------------------------------------------
    "ios_update": [
        "ios update",
        "ios upgrade",
        "update ios",
        "update to ios",
        "update to ios 10",
        "update to ios 11",
        "update to ios 12",
        "update to ios 13",
        "update to ios 14",
        "update to ios 15",
        "update to ios 16",
        "update to ios 17",
        "update failed",
        "ios update failed",
        "ios update error",
        "can't update ios",
        "cannot update ios",
        "cant update ios",
        "won't update ios",
        "wont update ios",
        "install ios",
        "install ios update",
        "downgrade ios",
        "downgrade from ios",
        "rollback ios",
        "revert to ios",
        "go back to ios",
        "remove ios beta",
        "ios beta",
        "ios public beta",
    ],


    # --------------------------------------------------------
    # MESSAGES / IMESSAGE
    # --------------------------------------------------------
    "messages_imessage": [
        "imessage",
        "i message",
        "text message",
        "text messages",
        "messages not delivered",
        "message not delivered",
        "message won't send",
        "messages won't send",
        "message wont send",
        "messages wont send",
        "can't send text",
        "cannot send text",
        "cant send text",
        "can't send messages",
        "cannot send messages",
        "cant send messages",
        "text not delivered",
        "texts not delivered",
        "text won't send",
        "texts won't send",
        "group chat",
        "groupchat",
        "sms",
        "send a text",
        "send text",
        "message delivery",
    ],


    # --------------------------------------------------------
    # SIM / CELLULAR / MOBILE NETWORK
    # --------------------------------------------------------
    "sim_cellular_network": [
        "no sim",
        "no sim card",
        "sim card",
        "sim card not",
        "sim not detected",
        "sim isn't detected",
        "sim is not detected",
        "cellular data",
        "cellular network",
        "mobile data",
        "mobile network",
        "4g",
        "4g not working",
        "lte",
        "lte not working",
        "network not working",
        "mobile network not working",
        "cellular not working",
        "no service",
        "no signal",
        "signal not",
        "wifi calling",
        "carrier",
        "carrier network",
    ],


    # --------------------------------------------------------
    # APPS / APP STORE
    # --------------------------------------------------------
    "apps_app_store": [
        "app store",
        "appstore",
        "download an app",
        "download app",
        "download apps",
        "install app",
        "install apps",
        "app won't install",
        "app wont install",
        "app cannot install",
        "app can't install",
        "can't download app",
        "cannot download app",
        "cant download app",
        "can't download apps",
        "cannot download apps",
        "cant download apps",
        "apps not downloading",
        "app not downloading",
        "app store not working",
        "app store won't",
        "app store wont",
        "app store cannot",
        "app store can't",
        "update an app",
        "update apps",
        "apps won't update",
        "apps wont update",
        "app won't update",
        "app wont update",
    ],


    # --------------------------------------------------------
    # APPLE ID / ACCOUNT
    # --------------------------------------------------------
    "apple_id_account": [
        "apple id",
        "appleid",
        "apple id password",
        "forgot my apple id",
        "forgot apple id",
        "forgot my password",
        "apple id password",
        "account password",
        "account locked",
        "apple account locked",
        "can't sign in to apple",
        "cannot sign in to apple",
        "cant sign in to apple",
        "can't sign into apple",
        "cannot sign into apple",
        "cant sign into apple",
        "sign in to apple id",
        "sign into apple id",
        "login to apple id",
        "log into apple id",
        "apple id verification",
        "apple id verification code",
        "apple id security",
    ],


    # --------------------------------------------------------
    # ICLOUD / PHOTOS
    # --------------------------------------------------------
    "icloud_photos": [
        "icloud photos",
        "icloud photo",
        "icloud photo library",
        "icloud photo library",
        "photos not syncing",
        "photos won't sync",
        "photos wont sync",
        "photos not synced",
        "photos disappeared from icloud",
        "photos missing from icloud",
        "icloud storage",
        "icloud backup",
        "icloud backups",
        "icloud drive",
        "icloud sync",
        "icloud not syncing",
        "icloud isn't syncing",
        "icloud is not syncing",
        "icloud storage full",
        "icloud full",
        "photo library not syncing",
        "pictures not syncing to icloud",
    ],


    # --------------------------------------------------------
    # PURCHASES / BILLING
    # --------------------------------------------------------
    "purchases_billing": [
        "refund",
        "refund request",
        "refund my",
        "charged for",
        "charged me",
        "charged twice",
        "charged twice for",
        "unexpected charge",
        "unauthorized charge",
        "payment method",
        "payment declined",
        "payment rejected",
        "payment failed",
        "billing",
        "billing issue",
        "billing problem",
        "purchase",
        "purchase history",
        "purchased",
        "subscription",
        "subscription charge",
        "itunes purchase",
        "app store purchase",
        "charged on itunes",
        "charged by apple",
        "payment information",
        "credit card charged",
    ],


    # --------------------------------------------------------
    # DEVICE SETTINGS / FEATURES
    # --------------------------------------------------------
    "device_settings_features": [
        "bluetooth",
        "bluetooth not working",
        "bluetooth won't connect",
        "bluetooth wont connect",
        "airplane mode",
        "control center",
        "brightness",
        "brightness setting",
        "volume",
        "volume control",
        "ringer",
        "ringer volume",
        "alarm",
        "alarm not working",
        "alarm doesn't work",
        "alarm does not work",
        "siri",
        "siri not working",
        "wifi",
        "wi-fi",
        "wifi not working",
        "wi-fi not working",
        "keyboard",
        "keyboard not working",
        "touch bar",
        "location services",
        "location service",
        "orientation",
        "screen rotation",
        "auto lock",
        "screen auto lock",
        "airplay",
        "carplay",
    ],


    # --------------------------------------------------------
    # PERFORMANCE / CRASH
    # --------------------------------------------------------
    "performance_crash": [
        "crash",
        "crashes",
        "crashing",
        "freeze",
        "freezes",
        "freezing",
        "frozen",
        "phone is frozen",
        "iphone is frozen",
        "glitch",
        "glitching",
        "glitchy",
        "slow",
        "slower",
        "slow iphone",
        "iphone slow",
        "phone slow",
        "lag",
        "lagging",
        "reboot",
        "rebooting",
        "restarts",
        "restarting",
        "random restart",
        "random restarting",
        "keeps restarting",
        "keeps rebooting",
        "black screen",
        "screen goes black",
        "shuts off",
        "keeps shutting off",
        "touch not working",
        "touchscreen not working",
        "touch screen not working",
        "keyboard glitch",
        "keyboard glitching",
    ],
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    """
    Normalize tweet text for rule matching.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(text):
    """
    Weakly classify one customer message.

    Returns:
        (intent, score)

    If the message is ambiguous, return:
        (None, 0)
    """

    text = normalize(text)

    matches = []

    for intent, keywords in RULES.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        if score > 0:
            matches.append(
                (intent, score)
            )

    # No matching intent
    if not matches:
        return None, 0

    # Sort highest score first
    matches.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_intent, top_score = matches[0]

    # --------------------------------------------------------
    # Ambiguity handling
    # --------------------------------------------------------
    #
    # If two intents have the same score,
    # don't force a label.
    #
    if len(matches) > 1:

        second_intent, second_score = matches[1]

        if top_score == second_score:
            return None, 0

    # If the winning signal is weak and multiple
    # intents are present, discard it.
    if len(matches) >= 3 and top_score == 1:
        return None, 0

    return top_intent, top_score


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("CREATING TRAINING DATA")
print("=" * 60)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

threads = pd.read_csv(
    THREADS_FILE
)

golden = pd.read_csv(
    GOLDEN_FILE
)


# ------------------------------------------------------------
# Normalize tweet IDs
# ------------------------------------------------------------

threads["customer_tweet_id"] = (
    threads["customer_tweet_id"]
    .astype(str)
    .str.replace(
        ".0",
        "",
        regex=False
    )
)

golden["customer_tweet_id"] = (
    golden["customer_tweet_id"]
    .astype(str)
    .str.replace(
        ".0",
        "",
        regex=False
    )
)


# ------------------------------------------------------------
# Golden IDs
# ------------------------------------------------------------

golden_ids = set(
    golden["customer_tweet_id"]
)


# ------------------------------------------------------------
# Remove golden evaluation examples
# ------------------------------------------------------------

train_pool = threads[
    ~threads["customer_tweet_id"].isin(
        golden_ids
    )
].copy()


print(
    f"Total historical conversations : {len(threads)}"
)

print(
    f"Golden evaluation examples     : {len(golden)}"
)

print(
    f"Training pool                  : {len(train_pool)}"
)


# ============================================================
# WEAK LABELING
# ============================================================

labels = []
scores = []

for text in train_pool["customer_text"].fillna(""):

    intent, score = classify(text)

    labels.append(intent)
    scores.append(score)


train_pool["intent"] = labels
train_pool["rule_score"] = scores


# ------------------------------------------------------------
# Keep only confidently classified examples
# ------------------------------------------------------------

training = train_pool[
    train_pool["intent"].notna()
].copy()


# ------------------------------------------------------------
# Remove duplicate customer messages
# ------------------------------------------------------------

before_dedup = len(training)

training = training.drop_duplicates(
    subset=["customer_text"]
)

after_dedup = len(training)


# ------------------------------------------------------------
# Keep useful columns
# ------------------------------------------------------------

training = training[
    [
        "customer_tweet_id",
        "customer_text",
        "support_reply",
        "intent",
        "customer_time",
        "support_time",
    ]
]


# ============================================================
# SAVE
# ============================================================

training.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING DATA CREATED")
print("=" * 60)

print(
    f"Training examples: {len(training)}"
)

print(
    f"Duplicates removed: "
    f"{before_dedup - after_dedup}"
)

print("\nIntent distribution:")

print(
    training["intent"].value_counts()
)


print("\nGolden set excluded:")
print(
    len(golden_ids)
)


print("\nSaved to:")
print(
    OUTPUT_FILE
)

print("=" * 60)