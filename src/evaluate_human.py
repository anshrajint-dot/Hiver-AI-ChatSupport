from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "human_evaluation_30.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "human_evaluation_summary.csv"

df = pd.read_csv(INPUT_FILE)

score_columns = [
    "human_groundedness",
    "human_helpfulness",
    "human_actionability",
    "human_safety",
    "human_evidence_relevance",
]

# Convert scores to numeric
for col in score_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Check missing scores
missing = df[score_columns].isna().sum()

if missing.sum() > 0:
    print("\nERROR: Some scores are missing:")
    print(missing[missing > 0])
    print("\nPlease complete all 5 scoring columns first.")
    exit()

# Calculate total
df["human_total"] = df[score_columns].sum(axis=1)

# Overall rating
def get_overall(score):
    if score <= 2:
        return "poor"
    elif score <= 5:
        return "fair"
    elif score <= 8:
        return "good"
    else:
        return "excellent"

df["human_overall"] = df["human_total"].apply(get_overall)

# Save updated detailed evaluation
df.to_csv(INPUT_FILE, index=False, encoding="utf-8-sig")

print("=" * 60)
print("HUMAN EVALUATION RESULTS")
print("=" * 60)

print(f"\nExamples evaluated: {len(df)}")

print("\nAverage scores:")
for col in score_columns:
    print(
        f"{col.replace('human_', '').replace('_', ' ').title():25s}: "
        f"{df[col].mean():.2f} / 2"
    )

print(f"\nAverage total score : {df['human_total'].mean():.2f} / 10")

print("\nOverall distribution:")
print(
    df["human_overall"]
    .value_counts()
    .reindex(
        ["poor", "fair", "good", "excellent"],
        fill_value=0
    )
    .to_string()
)

# Evidence relevance
evidence_good = (df["human_evidence_relevance"] >= 1).mean() * 100
evidence_strong = (df["human_evidence_relevance"] == 2).mean() * 100

print("\nEvidence relevance:")
print(f"Acceptable or better (>=1): {evidence_good:.1f}%")
print(f"Strong (2/2)              : {evidence_strong:.1f}%")

# Safety
safe = (df["human_safety"] >= 1).mean() * 100

print("\nSafety:")
print(f"Acceptable or better: {safe:.1f}%")

# Save summary
summary = pd.DataFrame({
    "metric": [
        "examples_evaluated",
        "avg_groundedness",
        "avg_helpfulness",
        "avg_actionability",
        "avg_safety",
        "avg_evidence_relevance",
        "avg_total",
        "evidence_relevance_acceptable_pct",
        "evidence_relevance_strong_pct",
        "safety_acceptable_pct",
    ],
    "value": [
        len(df),
        round(df["human_groundedness"].mean(), 2),
        round(df["human_helpfulness"].mean(), 2),
        round(df["human_actionability"].mean(), 2),
        round(df["human_safety"].mean(), 2),
        round(df["human_evidence_relevance"].mean(), 2),
        round(df["human_total"].mean(), 2),
        round(evidence_good, 1),
        round(evidence_strong, 1),
        round(safe, 1),
    ]
})

summary.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)
print(f"\nDetailed results:")
print(INPUT_FILE)
print(f"\nSummary:")
print(OUTPUT_FILE)