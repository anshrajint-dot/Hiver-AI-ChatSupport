import pandas as pd
from pathlib import Path

INPUT_FILE = "data/processed/applesupport_threads.csv"
OUTPUT_FILE = "data/processed/intent_sample.csv"

SAMPLE_SIZE = 500


def main():
    print("Loading conversations...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total conversations: {len(df):,}")

    sample_size = min(SAMPLE_SIZE, len(df))

    sample = df.sample(
        n=sample_size,
        random_state=42
    ).reset_index(drop=True)

    # Keep only useful columns for intent discovery
    sample = sample[
        [
            "customer_tweet_id",
            "customer_text",
            "support_reply"
        ]
    ]

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    sample.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===================================")
    print("Intent sample created!")
    print("===================================")

    print(f"Sample size: {len(sample):,}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nFirst 10 examples:\n")

    for i, row in sample.head(10).iterrows():
        print(f"\n--- Example {i + 1} ---")
        print("CUSTOMER:")
        print(row["customer_text"])
        print("\nAPPLE SUPPORT:")
        print(row["support_reply"])


if __name__ == "__main__":
    main()