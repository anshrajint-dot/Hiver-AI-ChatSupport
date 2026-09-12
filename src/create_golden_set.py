import pandas as pd
from pathlib import Path

INPUT_FILE = "data/processed/applesupport_threads.csv"
OUTPUT_FILE = "data/processed/golden_set.csv"

GOLDEN_SIZE = 200


def main():
    print("Loading AppleSupport conversations...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total conversations: {len(df):,}")

    # Reproducible random sample
    golden = df.sample(
        n=GOLDEN_SIZE,
        random_state=2027
    ).reset_index(drop=True)

    golden["intent"] = ""
    golden["notes"] = ""

    golden = golden[
        [
            "customer_tweet_id",
            "customer_text",
            "support_reply",
            "intent",
            "notes"
        ]
    ]

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    golden.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===================================")
    print("Golden set created!")
    print("===================================")

    print(f"Golden examples: {len(golden)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nIntent labels to use:")

    intents = [
        "ios_update",
        "battery_charging",
        "performance_crash",
        "sim_cellular_network",
        "apps_app_store",
        "apple_id_account",
        "icloud_photos",
        "messages_imessage",
        "calls",
        "purchases_billing",
        "device_settings_features",
        "other_unclear"
    ]

    for i, intent in enumerate(intents, 1):
        print(f"{i}. {intent}")


if __name__ == "__main__":
    main()