import pandas as pd
import re
from pathlib import Path


INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/processed/applesupport.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total tweets: {len(df):,}")

    # AppleSupport conversations
    # Support account tweets contain AppleSupport in the author/mentions.
    mask = (
        df["author_id"].astype(str).str.lower().eq("applesupport")
        | df["text"].astype(str).str.contains(
            "@AppleSupport",
            case=False,
            na=False
        )
    )

    apple = df[mask].copy()

    print(f"AppleSupport tweets: {len(apple):,}")

    # Clean text
    apple["clean_text"] = apple["text"].apply(clean_text)

    # Convert date
    apple["created_at"] = pd.to_datetime(
        apple["created_at"],
        errors="coerce",
        utc=True
    )

    # Keep useful columns
    apple = apple[
        [
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "clean_text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ]
    ]

    # Remove empty tweets
    apple = apple[apple["clean_text"].str.len() > 0]

    # Sort chronologically
    apple = apple.sort_values("created_at")

    # Save
    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    apple.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nProcessing complete!")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Final rows: {len(apple):,}")


if __name__ == "__main__":
    main()