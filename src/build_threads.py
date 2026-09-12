import pandas as pd
from pathlib import Path


INPUT_FILE = "data/processed/applesupport.csv"
OUTPUT_FILE = "data/processed/applesupport_threads.csv"


def is_inbound(value):
    return str(value).strip().lower() in {
        "true", "1", "1.0"
    }


def is_outbound(value):
    return str(value).strip().lower() in {
        "false", "0", "0.0"
    }


def split_ids(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def main():

    print("Loading AppleSupport data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total AppleSupport tweets: {len(df):,}")

    # Normalize tweet IDs
    df["tweet_id"] = df["tweet_id"].astype(str).str.strip()

    # Create lookup
    tweet_map = df.set_index("tweet_id").to_dict("index")

    conversations = []

    print("Building customer -> AppleSupport conversations...")

    for _, row in df.iterrows():

        # We only want customer tweets
        if not is_inbound(row["inbound"]):
            continue

        customer_text = str(row["clean_text"]).strip()

        if not customer_text:
            continue

        # Find tweets that this customer tweet points to
        response_ids = split_ids(row["response_tweet_id"])

        for response_id in response_ids:

            if response_id not in tweet_map:
                continue

            reply = tweet_map[response_id]

            # Reply must be from AppleSupport
            if not is_outbound(reply["inbound"]):
                continue

            conversations.append({
                "customer_tweet_id": row["tweet_id"],
                "customer_text": customer_text,
                "support_tweet_id": response_id,
                "support_reply": str(reply["clean_text"]).strip(),
                "customer_time": row["created_at"],
                "support_time": reply["created_at"]
            })

    result = pd.DataFrame(conversations)

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===================================")
    print("Conversation building complete!")
    print("===================================")

    print(f"Conversations: {len(result):,}")

    print(f"Saved to: {OUTPUT_FILE}")

    if len(result) > 0:

        print("\nSample conversations:\n")

        print(
            result[
                [
                    "customer_text",
                    "support_reply"
                ]
            ].head(5).to_string(index=False)
        )


if __name__ == "__main__":
    main()