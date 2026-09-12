import pandas as pd
from pathlib import Path

FILE = Path("data/processed/golden_set.csv")

# CSV row number -> corrected intent
CORRECTIONS = {
    131: "device_settings_features",
    133: "other_unclear",
    137: "messages_imessage",
    138: "messages_imessage",
    140: "purchases_billing",
    143: "device_settings_features",
    144: "sim_cellular_network",
}

df = pd.read_csv(FILE)

for csv_row, new_intent in CORRECTIONS.items():
    index = csv_row - 2   # CSV header = row 1
    old_intent = df.loc[index, "intent"]

    print(
        f"Row {csv_row}: "
        f"{old_intent} -> {new_intent}"
    )

    df.loc[index, "intent"] = new_intent

df.to_csv(FILE, index=False)

print("\nCorrections applied successfully!")
print(f"Saved: {FILE}")