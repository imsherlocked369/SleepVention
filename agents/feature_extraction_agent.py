# agents/feature_extraction_agent.py

def feature_extraction_agent(data, baseline):
    """
    Agent 2: Feature Extraction Agent

    Purpose:
    Extracts useful sleep features from wearable data.
    Also compares today's sleep values with the user's baseline.
    """

    features = {}

    features["date"] = data["date"]

    # Main sleep values
    features["sleep_duration_min"] = data["sleep_duration_min"]
    features["sleep_duration_hours"] = round(data["sleep_duration_min"] / 60, 2)

    features["time_in_bed_min"] = data["time_in_bed_min"]
    features["sleep_efficiency"] = data["sleep_efficiency"]

    features["waso_min"] = data["waso_min"]
    features["wake_count"] = data["wake_count"]

    features["resting_hr"] = data["resting_hr"]
    features["hrv"] = data["hrv"]
    features["movement_count"] = data["movement_count"]

    # Optional wearable-estimated sleep stages
    features["deep_sleep_min"] = data.get("deep_sleep_min")
    features["rem_sleep_min"] = data.get("rem_sleep_min")

    # Baseline comparisons
    features["sleep_duration_change"] = (
        data["sleep_duration_min"] - baseline["avg_sleep_duration_min"]
    )

    features["sleep_efficiency_change"] = (
        data["sleep_efficiency"] - baseline["avg_sleep_efficiency"]
    )

    features["resting_hr_change"] = (
        data["resting_hr"] - baseline["avg_resting_hr"]
    )

    features["hrv_change"] = (
        data["hrv"] - baseline["avg_hrv"]
    )

    features["movement_change"] = (
        data["movement_count"] - baseline["avg_movement_count"]
    )

    return features