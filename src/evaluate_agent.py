from pathlib import Path
import sys
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

GOLDEN_FILE = BASE_DIR / "data" / "processed" / "golden_set_final_v2.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "agent_evaluation.jsonl"

sys.path.insert(0, str(BASE_DIR / "src"))

from support_agent import support_agent


df = pd.read_csv(GOLDEN_FILE)

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


print("=" * 60)
print("HIVER AGENT EVALUATION")
print("=" * 60)

print(f"\nGolden examples: {len(df)}")


results = []

for i, row in df.iterrows():

    customer_message = row["customer_text"]
    gold_intent = row["intent"]

    try:

        result = support_agent(customer_message)

        record = {
            "customer_message": customer_message,
            "gold_intent": gold_intent,

            "predicted_intent": result["intent"],
            "confidence": result["confidence"],

            "action": result["action"],
            "reason": result["reason"],

            "reply": result["reply"],

            "evidence_reason": result["evidence_reason"],
            "evidence": result["evidence"],

            "intent_correct": (
                result["intent"] == gold_intent
            )
        }

        results.append(record)

    except Exception as e:

        print(
            f"Error processing example {i}: {e}"
        )


# ----------------------------------------------------------
# SAVE JSONL
# ----------------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for record in results:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# ----------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------

result_df = pd.DataFrame(results)

intent_accuracy = (
    result_df["intent_correct"].mean()
)

auto_handle_rate = (
    result_df["action"] == "auto_handle"
).mean()

escalation_rate = (
    result_df["action"] == "escalate"
).mean()


print("\n" + "=" * 60)
print("AGENT SUMMARY")
print("=" * 60)

print(
    f"\nIntent accuracy : "
    f"{intent_accuracy:.4f}"
)

print(
    f"Auto-handle rate: "
    f"{auto_handle_rate:.4f}"
)

print(
    f"Escalation rate : "
    f"{escalation_rate:.4f}"
)


# ----------------------------------------------------------
# CORRECT vs INCORRECT AUTO-HANDLES
# ----------------------------------------------------------

auto_df = result_df[
    result_df["action"] == "auto_handle"
]

if len(auto_df) > 0:

    auto_intent_accuracy = (
        auto_df["intent_correct"].mean()
    )

    print(
        f"\nAuto-handle intent accuracy: "
        f"{auto_intent_accuracy:.4f}"
    )


# ----------------------------------------------------------
# ESCALATED CASES
# ----------------------------------------------------------

escalated_df = result_df[
    result_df["action"] == "escalate"
]

print(
    f"\nEscalated examples: "
    f"{len(escalated_df)}"
)


# ----------------------------------------------------------
# FALSE AUTO-HANDLES
# ----------------------------------------------------------

false_auto = result_df[
    (result_df["action"] == "auto_handle") &
    (result_df["intent_correct"] == False)
]

print(
    f"False auto-handles: "
    f"{len(false_auto)}"
)


# ----------------------------------------------------------
# SAVE CSV SUMMARY
# ----------------------------------------------------------

summary_file = (
    BASE_DIR /
    "data" /
    "processed" /
    "agent_evaluation_summary.csv"
)

result_df[
    [
        "customer_message",
        "gold_intent",
        "predicted_intent",
        "confidence",
        "action",
        "intent_correct",
        "reply",
        "reason",
        "evidence_reason"
    ]
].to_csv(
    summary_file,
    index=False,
    encoding="utf-8"
)


print("\n" + "=" * 60)
print("FILES CREATED")
print("=" * 60)

print(f"\nJSONL:")
print(OUTPUT_FILE)

print(f"\nCSV:")
print(summary_file)