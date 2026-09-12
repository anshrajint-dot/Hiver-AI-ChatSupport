from pathlib import Path
import re
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "data" / "processed" / "intent_model.joblib"
INDEX_FILE = BASE_DIR / "data" / "processed" / "resolution_index.csv"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# LOAD HISTORICAL RESOLUTION INDEX
# ============================================================

df = pd.read_csv(INDEX_FILE)

df["customer_clean"] = (
    df["customer_clean"]
    .fillna("")
    .astype(str)
)

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

df["intent"] = (
    df["intent"]
    .fillna("")
    .astype(str)
)


# ============================================================
# TF-IDF RETRIEVAL MODEL
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000,
    sublinear_tf=True,
)

matrix = vectorizer.fit_transform(
    df["customer_clean"]
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Normalize customer message before classification/retrieval.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_signals(text):
    """
    Detect broad device/topic signals used to improve
    historical retrieval.
    """

    text = str(text).lower()

    signals = set()

    # Apple devices
    if any(
        x in text
        for x in [
            "iphone",
            "ipad",
            "ios",
            "ipod",
            "mac",
            "macbook",
        ]
    ):
        signals.add("ios_device")

    # Apple Watch
    if any(
        x in text
        for x in [
            "watch",
            "apple watch",
            "watchos",
        ]
    ):
        signals.add("watch")

    # Battery
    if any(
        x in text
        for x in [
            "battery",
            "charging",
            "charge",
            "drain",
            "dies",
            "dying",
        ]
    ):
        signals.add("battery")

    # Cellular / network
    if any(
        x in text
        for x in [
            "network",
            "signal",
            "carrier",
            "sim",
            "no service",
            "searching",
            "cellular",
        ]
    ):
        signals.add("connectivity")

    # App Store
    if any(
        x in text
        for x in [
            "app store",
            "download app",
            "download apps",
            "apps",
            "application",
        ]
    ):
        signals.add("app_store")

    # Calls
    if any(
        x in text
        for x in [
            "call",
            "calling",
            "phone call",
            "cannot call",
            "can't call",
            "unable to call",
        ]
    ):
        signals.add("calls")

    # Messages
    if any(
        x in text
        for x in [
            "message",
            "messages",
            "imessage",
            "text",
            "texting",
        ]
    ):
        signals.add("messages")

    # Apple ID / account
    if any(
        x in text
        for x in [
            "apple id",
            "icloud account",
            "password",
            "sign in",
            "signin",
            "login",
            "log in",
        ]
    ):
        signals.add("account")

    return signals


# ============================================================
# SAFE REPLY GENERATION
# ============================================================

def generate_reply(intent, text):
    """
    Generate a safe support draft.

    Historical replies are used as grounding evidence,
    but are NOT copied directly to the customer.
    """

    if intent == "apps_app_store":
        return (
            "We're here to help with your App Store issue. "
            "Could you let us know what happens when you try to "
            "download or update an app?"
        )

    if intent == "calls":
        return (
            "We're here to help with your calling issue. "
            "Please let us know which iOS version your device is running "
            "and whether you're able to receive calls."
        )

    if intent == "battery_charging":
        return (
            "We're happy to help with the battery issue. "
            "Could you tell us which device and iOS version you're using "
            "and when you first noticed the battery or charging problem?"
        )

    if intent == "sim_cellular_network":
        return (
            "We're here to help with the network issue. "
            "Could you let us know which device you're using and whether "
            "you see 'No Service' or 'Searching'?"
        )

    if intent == "ios_update":
        return (
            "We're happy to help with the iOS update issue. "
            "Could you tell us which device and iOS version you're currently "
            "using and what happens when you try to update?"
        )

    if intent == "messages_imessage":
        return (
            "We're here to help with your messaging issue. "
            "Could you tell us what happens when you try to send or receive "
            "a message?"
        )

    if intent == "icloud_photos":
        return (
            "We're here to help with your iCloud or Photos issue. "
            "Could you tell us what is happening with your photos and "
            "which device you're using?"
        )

    if intent == "apple_id_account":
        return (
            "We're here to help with your Apple ID issue. "
            "Could you tell us whether you're having trouble signing in, "
            "with your password, or with your account settings?"
        )

    if intent == "purchases_billing":
        return (
            "We're here to help with your purchase or billing issue. "
            "Could you tell us what happened with the purchase or charge "
            "and whether you received any error message?"
        )

    if intent == "device_settings_features":
        return (
            "We're here to help with your device issue. "
            "Could you tell us which device you're using and what happens "
            "when you try the affected feature?"
        )

    if intent == "performance_crash":
        return (
            "We're here to help troubleshoot the issue. "
            "Could you tell us which device and iOS version you're using "
            "and what happens when the problem occurs?"
        )

    # other_unclear / fallback
    return (
        "We're happy to help. Could you provide a few more details "
        "about the issue you're experiencing?"
    )


# ============================================================
# HISTORICAL RETRIEVAL
# ============================================================

def retrieve(customer_message, predicted_intent, top_k=3):
    """
    Retrieve historically similar AppleSupport conversations.

    Retrieval is first filtered by predicted intent and then
    reranked using device/topic signals.
    """

    cleaned = clean_text(
        customer_message
    )

    query_vector = vectorizer.transform(
        [cleaned]
    )

    # --------------------------------------------------------
    # Intent-aware filtering
    # --------------------------------------------------------

    intent_df = df[
        df["intent"] == predicted_intent
    ].copy()

    if len(intent_df) == 0:
        candidate_indices = df.index.tolist()
    else:
        candidate_indices = intent_df.index.tolist()

    candidate_matrix = matrix[
        candidate_indices
    ]

    # --------------------------------------------------------
    # TF-IDF cosine similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_vector,
        candidate_matrix
    )[0]

    # --------------------------------------------------------
    # Query signals
    # --------------------------------------------------------

    query_signals = detect_signals(
        customer_message
    )

    results = []

    for local_position, similarity in enumerate(
        similarities
    ):

        original_index = candidate_indices[
            local_position
        ]

        row = df.loc[
            original_index
        ]

        historical_text = (
            row["customer_clean"]
            .lower()
        )

        historical_signals = detect_signals(
            historical_text
        )

        shared_signals = (
            query_signals.intersection(
                historical_signals
            )
        )

        score = float(similarity)

        # Reward shared device/topic signals
        score += (
            0.08 * len(shared_signals)
        )

        # Avoid obvious device mismatch
        if (
            "watch" in query_signals
            and "ios_device" in historical_signals
        ):
            score -= 0.10

        if (
            "ios_device" in query_signals
            and "watch" in historical_signals
        ):
            score -= 0.10

        results.append(
            {
                "customer_text": row[
                    "customer_text"
                ],
                "support_reply": row[
                    "support_reply"
                ],
                "similarity": round(
                    float(similarity),
                    4
                ),
                "rerank_score": round(
                    float(score),
                    4
                ),
                "shared_signals": sorted(
                    list(shared_signals)
                ),
            }
        )

    # Highest evidence first
    results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# EVIDENCE QUALITY
# ============================================================

def evidence_quality(
    results,
    confidence
):
    """
    Safety-first evidence check.

    Historical evidence must have sufficiently strong
    semantic similarity AND the classifier must have
    reasonable confidence.
    """

    if not results:
        return (
            False,
            "No historical evidence found."
        )

    top = results[0]

    similarity = top["similarity"]

    # Strict evidence requirement
    if (
        confidence >= 0.80
        and similarity >= 0.50
    ):
        return (
            True,
            "High-confidence intent with strong "
            "historical similarity."
        )

    return (
        False,
        "Historical evidence or intent confidence "
        "is not strong enough for safe automatic handling."
    )


# ============================================================
# SAFETY-FIRST ESCALATION POLICY
# ============================================================

def decide_action(
    confidence,
    evidence_ok,
    top_similarity
):
    """
    Decide whether the AI should handle the case
    or escalate it to a human.

    AUTO-HANDLE requires ALL THREE:
        confidence >= 0.80
        similarity >= 0.50
        evidence_ok == True

    Otherwise -> ESCALATE
    """

    # --------------------------------------------------------
    # Condition 1: classifier confidence
    # --------------------------------------------------------

    if confidence < 0.80:
        return (
            "escalate",
            "Intent confidence is below the safe "
            "auto-handle threshold (0.80)."
        )

    # --------------------------------------------------------
    # Condition 2: historical evidence
    # --------------------------------------------------------

    if top_similarity < 0.50:
        return (
            "escalate",
            "Historical similarity is below the safe "
            "evidence threshold (0.50)."
        )

    # --------------------------------------------------------
    # Condition 3: evidence quality
    # --------------------------------------------------------

    if not evidence_ok:
        return (
            "escalate",
            "Historical evidence is insufficient "
            "for reliable automated handling."
        )

    # --------------------------------------------------------
    # Safe to automate
    # --------------------------------------------------------

    return (
        "auto_handle",
        "Intent confidence, historical similarity, "
        "and evidence quality all meet the safe "
        "auto-handle thresholds."
    )


# ============================================================
# MAIN SUPPORT AGENT
# ============================================================

def support_agent(customer_message):

    # --------------------------------------------------------
    # Clean message
    # --------------------------------------------------------

    cleaned = clean_text(
        customer_message
    )

    # --------------------------------------------------------
    # Intent classification
    # --------------------------------------------------------

    prediction = model.predict(
        [cleaned]
    )[0]

    probabilities = model.predict_proba(
        [cleaned]
    )[0]

    confidence = float(
        max(probabilities)
    )

    # --------------------------------------------------------
    # Historical retrieval
    # --------------------------------------------------------

    evidence = retrieve(
        customer_message,
        prediction,
        top_k=3
    )

    # --------------------------------------------------------
    # Top historical similarity
    # --------------------------------------------------------

    if evidence:
        top_similarity = float(
            evidence[0]["similarity"]
        )
    else:
        top_similarity = 0.0

    # --------------------------------------------------------
    # Evidence quality
    # --------------------------------------------------------

    evidence_ok, evidence_reason = (
        evidence_quality(
            evidence,
            confidence
        )
    )

    # --------------------------------------------------------
    # Automation decision
    # --------------------------------------------------------

    action, action_reason = (
        decide_action(
            confidence,
            evidence_ok,
            top_similarity
        )
    )

    # --------------------------------------------------------
    # Generate safe reply
    # --------------------------------------------------------

    reply = generate_reply(
        prediction,
        customer_message
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "customer_message": customer_message,
        "intent": prediction,
        "confidence": round(
            confidence,
            4
        ),
        "top_similarity": round(
            top_similarity,
            4
        ),
        "action": action,
        "reason": action_reason,
        "reply": reply,
        "evidence_reason": evidence_reason,
        "evidence": evidence,
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_messages = [

        "I cannot download any apps from the App Store.",

        "My iPhone cannot make any calls.",

        "I have a problem with my Apple ID.",

        "My iPhone battery drains very quickly after the latest iOS update.",
    ]

    for message in test_messages:

        result = support_agent(
            message
        )

        print("\n")
        print("=" * 60)
        print("SUPPORT AGENT RESULT")
        print("=" * 60)

        print("\nCustomer:")
        print(
            result["customer_message"]
        )

        print("\nIntent:")
        print(
            result["intent"]
        )

        print("\nConfidence:")
        print(
            result["confidence"]
        )

        print("\nTop Historical Similarity:")
        print(
            result["top_similarity"]
        )

        print("\nAction:")
        print(
            result["action"]
        )

        print("\nReason:")
        print(
            result["reason"]
        )

        print("\nEvidence Reason:")
        print(
            result["evidence_reason"]
        )

        print("\nDraft Reply:")
        print(
            result["reply"]
        )

        print("\nHistorical Evidence:")
        print("-" * 60)

        for i, item in enumerate(
            result["evidence"],
            1
        ):

            print(
                f"\n{i}. "
                f"Similarity: {item['similarity']} "
                f"| Rerank: {item['rerank_score']}"
            )

            print(
                "Customer:",
                item["customer_text"]
            )

            print(
                "AppleSupport:",
                item["support_reply"]
            )

        print("\nJSON:")

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )