import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/golden_set_final.csv")
OUTPUT = Path("data/processed/golden_set_final_v2.csv")

df = pd.read_csv(INPUT)

# Manually reviewed corrections.
# These are semantic corrections, NOT keyword-based automatic labels.
CORRECTIONS = {

    # iCloud / Photos
    2093887: "icloud_photos",
    1467521: "icloud_photos",

    # Device / feature
    233933: "device_settings_features",
    797587: "device_settings_features",
    934477: "device_settings_features",
    1296682: "device_settings_features",
    1434363: "device_settings_features",
    1707734: "device_settings_features",
    554071: "device_settings_features",
    2817201: "device_settings_features",
    1692538: "sim_cellular_network",

    # Performance / crashes / glitches
    939219: "performance_crash",
    342860: "performance_crash",
    407077: "performance_crash",
    2284084: "battery_charging",
    2095389: "performance_crash",
    2039833: "performance_crash",
    778952: "performance_crash",
    2285946: "performance_crash",
    2179480: "performance_crash",
    1573107: "performance_crash",
    2505105: "performance_crash",
    1013789: "performance_crash",
    690402: "performance_crash",
    1920077: "performance_crash",
    2858333: "performance_crash",
    2284833: "performance_crash",
    2145640: "performance_crash",
    2424371: "performance_crash",
    2002392: "performance_crash",
    656279: "performance_crash",

    # Battery
    312188: "battery_charging",
    2924687: "battery_charging",
    1443126: "battery_charging",
    2004360: "battery_charging",
    1508053: "battery_charging",
    2834452: "battery_charging",
    944338: "battery_charging",
    1583045: "battery_charging",
    1857314: "battery_charging",
    2557328: "battery_charging",
    376031: "battery_charging",

    # Apps / App Store
    1053680: "apps_app_store",
    2082339: "apps_app_store",
    676798: "apps_app_store",
    747794: "apps_app_store",
    1658294: "apps_app_store",
    2894743: "apps_app_store",
    1194558: "apps_app_store",
    2440818: "apps_app_store",
    2690813: "apps_app_store",

    # Apple ID / account
    1588130: "apple_id_account",

    # Purchases / billing
    427835: "purchases_billing",
    36631: "purchases_billing",
    2014234: "purchases_billing",
    2841991: "purchases_billing",
    2123183: "purchases_billing",
    718096: "purchases_billing",

    # Messages
    1080466: "messages_imessage",
    2369420: "messages_imessage",
    613015: "messages_imessage",

    # Calls
    2518282: "calls",

    # iOS update
    407077: "performance_crash",
    751744: "ios_update",
    1788932: "battery_charging",
    2123904: "ios_update",

    # Insufficient context
    1995486: "other_unclear",
    585982: "other_unclear",
    1720797: "other_unclear",
    1143490: "other_unclear",
    2283106: "other_unclear",
    484495: "other_unclear",
    711656: "other_unclear",
    682100: "other_unclear",
    2725955: "other_unclear",
    1636519: "other_unclear",
    2633080: "other_unclear",
    2496361: "other_unclear",
}

changed = 0

for tweet_id, new_label in CORRECTIONS.items():
    mask = df["customer_tweet_id"] == tweet_id

    if mask.any():
        old_label = df.loc[mask, "intent"].iloc[0]

        if old_label != new_label:
            df.loc[mask, "intent"] = new_label
            changed += 1
            print(
                f"{tweet_id}: "
                f"{old_label} -> {new_label}"
            )

df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 60)
print("GOLDEN SET V2 CREATED")
print("=" * 60)
print("Total examples :", len(df))
print("Labels changed :", changed)
print("Saved to        :", OUTPUT)
print("\nNew distribution:")
print(df["intent"].value_counts())