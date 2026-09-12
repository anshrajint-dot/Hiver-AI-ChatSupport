from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "agent_evaluation.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "human_evaluation_30.csv"

# ============================================================
# CONFIG
# ============================================================

SAMPLE_SIZE = 30
RANDOM_STATE = 2027

# ============================================================
# LOAD AGENT EVALUATION
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_json(INPUT_FILE, lines=True)

print("=" * 60)
print("CREATING HUMAN EVALUATION SET")
print("=" * 60)

print(f"\nTotal agent evaluation examples: {len(df)}")

# ============================================================
# BASIC CLEANING
# ============================================================

required_columns = [
    "customer_message",
    "gold_intent",
    "predicted_intent",
    "confidence",
    "action",
    "reason",
    "reply",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in agent_evaluation.jsonl: {missing_columns}"
    )

df["gold_intent"] = (
    df["gold_intent"]
    .fillna("other_unclear")
    .astype(str)
    .str.strip()
)

# ============================================================
# STRATIFIED SAMPLING
# ============================================================

# Try to represent different intents proportionally.
sample_parts = []

for intent, group in df.groupby("gold_intent", group_keys=False):

    proportional_n = round(
        SAMPLE_SIZE * len(group) / len(df)
    )

    # At least one example from every represented intent
    n = max(1, proportional_n)

    n = min(n, len(group))

    sampled_group = group.sample(
        n=n,
        random_state=RANDOM_STATE
    )

    sample_parts.append(sampled_group)

sample = pd.concat(
    sample_parts,
    ignore_index=True
)

# ============================================================
# ENSURE EXACTLY 30 EXAMPLES
# ============================================================

if len(sample) > SAMPLE_SIZE:

    sample = sample.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

elif len(sample) < SAMPLE_SIZE:

    remaining = df[
        ~df.index.isin(sample.index)
    ]

    needed = SAMPLE_SIZE - len(sample)

    if len(remaining) >= needed:

        extra = remaining.sample(
            n=needed,
            random_state=RANDOM_STATE
        )

        sample = pd.concat(
            [sample, extra],
            ignore_index=True
        )

# ============================================================
# SHUFFLE FINAL ORDER
# ============================================================

sample = sample.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ============================================================
# CREATE HUMAN EVALUATION TABLE
# ============================================================

output = pd.DataFrame({

    "example_id": range(1, len(sample) + 1),

    # Input
    "customer_message": sample["customer_message"],

    # Ground truth / model information
    "gold_intent": sample["gold_intent"],
    "predicted_intent": sample["predicted_intent"],
    "confidence": sample["confidence"],

    # Agent decision
    "action": sample["action"],
    "agent_reason": sample["reason"],
    "agent_reply": sample["reply"],

    # ========================================================
    # HUMAN LABELS
    # ========================================================

    "human_groundedness": "",
    "human_helpfulness": "",
    "human_actionability": "",
    "human_safety": "",
    "human_evidence_relevance": "",

    "human_total": "",
    "human_overall": "",
    "human_notes": "",
})

# ============================================================
# SAVE
# ============================================================

output.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# SUMMARY
# ============================================================

print(f"\nExamples selected: {len(output)}")

print("\nIntent distribution:")
print(
    output["gold_intent"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print(f"\nHuman evaluation file:")
print(OUTPUT_FILE)

print("\nScoring columns to fill manually:")
print("  human_groundedness        : 0-2")
print("  human_helpfulness         : 0-2")
print("  human_actionability       : 0-2")
print("  human_safety              : 0-2")
print("  human_evidence_relevance  : 0-2")
print("  human_total               : 0-10")
print("  human_overall             : poor/fair/good/excellent")
print("  human_notes               : short explanation")