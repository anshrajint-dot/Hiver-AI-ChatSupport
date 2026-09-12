from pathlib import Path
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(BASE_DIR / ".env")


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "agent_evaluation.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "llm_judge_results.jsonl"
)


# ============================================================
# OPENAI CLIENT
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if not api_key or api_key == "YOUR_API_KEY":
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Add it to the .env file in the project root."
    )

client = OpenAI(api_key=api_key)


# ============================================================
# JUDGE PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an evaluator for an AI customer-support agent.

The agent supports Apple customers using historical
AppleSupport Twitter conversations as evidence.

Evaluate ONLY the quality of the generated support reply
and the relevance of the retrieved historical evidence.

Do not reward the agent simply because the intent is correct.

Use this scoring rubric:

1. Groundedness (0-2)
   0 = reply is unsupported or contradicts the evidence
   1 = partially grounded
   2 = clearly consistent with the historical evidence

2. Helpfulness (0-2)
   0 = not useful
   1 = somewhat useful
   2 = useful and appropriately addresses the issue

3. Actionability (0-2)
   0 = no useful next step
   1 = asks for information but is somewhat vague
   2 = gives or requests a clear next step

4. Safety (0-2)
   0 = unsafe, misleading, or inappropriate
   1 = mostly safe but could be improved
   2 = appropriately cautious and safe

5. Evidence relevance (0-2)
   0 = historical evidence is irrelevant
   1 = partially relevant
   2 = strongly relevant to the customer's issue

Return ONLY valid JSON with this structure:

{
  "groundedness": 0,
  "helpfulness": 0,
  "actionability": 0,
  "safety": 0,
  "evidence_relevance": 0,
  "total": 0,
  "overall": "poor|fair|good|excellent",
  "reason": "brief explanation"
}

The total must equal the five scores added together.
"""


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:
        line = line.strip()

        if line:
            records.append(json.loads(line))


# ============================================================
# SAMPLE
# ============================================================

# Fixed first 50 examples for reproducibility.
SAMPLE_SIZE = min(50, len(records))

records = records[:SAMPLE_SIZE]

print("=" * 60)
print("HIVER LLM-AS-JUDGE")
print("=" * 60)

print(f"\nExamples to judge: {len(records)}")


# ============================================================
# JUDGE FUNCTION
# ============================================================

def judge_record(record):

    evidence_text = []

    for i, evidence in enumerate(
        record.get("evidence", []),
        1
    ):

        evidence_text.append(
            f"""
Historical Example {i}:
Customer:
{evidence.get("customer_text", "")}

Historical AppleSupport Reply:
{evidence.get("support_reply", "")}

Similarity:
{evidence.get("similarity", 0)}

Rerank Score:
{evidence.get("rerank_score", 0)}
"""
        )

    evidence_block = "\n".join(evidence_text)

    user_prompt = f"""
Evaluate this AI support response.

Customer message:
{record.get("customer_message", "")}

Predicted intent:
{record.get("predicted_intent", "")}

Gold intent:
{record.get("gold_intent", "")}

Classifier confidence:
{record.get("confidence", 0)}

Agent action:
{record.get("action", "")}

Agent reason:
{record.get("reason", "")}

Generated reply:
{record.get("reply", "")}

Historical evidence:
{evidence_block}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    content = response.choices[0].message.content

    judgment = json.loads(content)

    # Validate required scores
    score_fields = [
        "groundedness",
        "helpfulness",
        "actionability",
        "safety",
        "evidence_relevance"
    ]

    for field in score_fields:

        if field not in judgment:
            raise ValueError(
                f"Missing score field: {field}"
            )

        score = int(judgment[field])

        if score < 0 or score > 2:
            raise ValueError(
                f"Invalid {field} score: {score}"
            )

        judgment[field] = score

    # Recalculate total
    judgment["total"] = sum(
        judgment[field]
        for field in score_fields
    )

    # Attach metadata
    judgment["customer_message"] = (
        record.get("customer_message", "")
    )

    judgment["predicted_intent"] = (
        record.get("predicted_intent", "")
    )

    judgment["gold_intent"] = (
        record.get("gold_intent", "")
    )

    judgment["action"] = (
        record.get("action", "")
    )

    judgment["intent_correct"] = (
        record.get("intent_correct", False)
    )

    return judgment


# ============================================================
# RUN JUDGE
# ============================================================

results = []

for index, record in enumerate(records, 1):

    print(
        f"\n[{index}/{len(records)}] Evaluating..."
    )

    try:

        result = judge_record(record)

        results.append(result)

        print(
            "Score:",
            result.get("total")
        )

        print(
            "Overall:",
            result.get("overall")
        )

    except Exception as e:

        print(
            "ERROR:",
            str(e)
        )

    # Small delay between requests
    time.sleep(0.2)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for result in results:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# SUMMARY
# ============================================================

if results:

    dimensions = [
        "groundedness",
        "helpfulness",
        "actionability",
        "safety",
        "evidence_relevance",
        "total",
    ]

    print("\n")
    print("=" * 60)
    print("LLM JUDGE SUMMARY")
    print("=" * 60)

    for dimension in dimensions:

        values = [
            float(
                result.get(
                    dimension,
                    0
                )
            )
            for result in results
        ]

        average = (
            sum(values)
            / len(values)
        )

        print(
            f"{dimension:20s}: "
            f"{average:.2f}"
        )

    print(
        f"\nJudged examples: "
        f"{len(results)}"
    )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_FILE}"
    )

else:

    print(
        "\nNo successful judge results."
    )