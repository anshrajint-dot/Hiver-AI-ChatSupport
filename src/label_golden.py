import pandas as pd

INPUT_FILE = "data/processed/golden_set.csv"
OUTPUT_FILE = "data/processed/golden_set.csv"

INTENTS = {
    "1": "ios_update",
    "2": "battery_charging",
    "3": "performance_crash",
    "4": "sim_cellular_network",
    "5": "apps_app_store",
    "6": "apple_id_account",
    "7": "icloud_photos",
    "8": "messages_imessage",
    "9": "calls",
    "10": "purchases_billing",
    "11": "device_settings_features",
    "12": "other_unclear"
}


def show_intents():
    print("\nINTENTS:")
    for key, value in INTENTS.items():
        print(f"{key}. {value}")


def is_labeled(value):
    """
    Correctly detect whether an intent is actually labeled.
    NaN, empty string and 'nan' are treated as unlabeled.
    """
    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    if value == "" or value == "nan":
        return False

    return True


def main():

    print("Loading golden set...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total examples: {len(df)}")

    # ---------------------------------------------
    # Ensure required columns exist
    # ---------------------------------------------

    if "intent" not in df.columns:
        df["intent"] = ""

    if "notes" not in df.columns:
        df["notes"] = ""

    # ---------------------------------------------
    # Count already labeled examples
    # ---------------------------------------------

    labeled = 0

    for i in range(len(df)):
        if is_labeled(df.loc[i, "intent"]):
            labeled += 1

    print(f"Already labeled: {labeled}")
    print(f"Remaining: {len(df) - labeled}")

    # ---------------------------------------------
    # Label each example
    # ---------------------------------------------

    for i in range(len(df)):

        # Skip already labeled examples
        if is_labeled(df.loc[i, "intent"]):
            continue

        print("\n" + "=" * 70)
        print(f"Example {i + 1} / {len(df)}")
        print("=" * 70)

        print("\nCUSTOMER:")
        print(df.loc[i, "customer_text"])

        print("\nHISTORICAL APPLE SUPPORT REPLY:")
        print(df.loc[i, "support_reply"])

        show_intents()

        # -----------------------------------------
        # Ask for intent
        # -----------------------------------------

        while True:

            choice = input(
                "\nEnter intent number (1-12), or q to quit: "
            ).strip().lower()

            if choice == "q":

                # Save current progress
                df.to_csv(
                    OUTPUT_FILE,
                    index=False
                )

                print("\n===================================")
                print("Progress saved!")
                print("===================================")

                # Recalculate actual labels
                current_labeled = sum(
                    is_labeled(value)
                    for value in df["intent"]
                )

                print(
                    f"Labeled examples: "
                    f"{current_labeled}/{len(df)}"
                )

                print(f"Saved to: {OUTPUT_FILE}")

                return

            if choice in INTENTS:

                # Save intent
                df.loc[i, "intent"] = INTENTS[choice]

                # ---------------------------------
                # Optional note
                # ---------------------------------

                note = input(
                    "Optional note (press Enter to skip): "
                ).strip()

                df.loc[i, "notes"] = note

                # ---------------------------------
                # Save immediately
                # ---------------------------------

                df.to_csv(
                    OUTPUT_FILE,
                    index=False
                )

                current_labeled = sum(
                    is_labeled(value)
                    for value in df["intent"]
                )

                print(
                    f"\nSaved! Progress: "
                    f"{current_labeled}/{len(df)}"
                )

                break

            print(
                "Invalid choice. "
                "Enter a number from 1-12 or q."
            )

    # ---------------------------------------------
    # Final verification
    # ---------------------------------------------

    df = pd.read_csv(OUTPUT_FILE)

    final_labeled = sum(
        is_labeled(value)
        for value in df["intent"]
    )

    print("\n===================================")
    print("GOLDEN SET LABELING COMPLETE")
    print("===================================")

    print(
        f"Labeled examples: "
        f"{final_labeled}/{len(df)}"
    )

    print(f"Saved to: {OUTPUT_FILE}")

    print("\nIntent distribution:")

    print(
        df["intent"]
        .dropna()
        .value_counts()
    )


if __name__ == "__main__":
    main()