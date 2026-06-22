# agents/data_quality_agent.py

def data_quality_agent(data):
    """
    Agent 1: Data Quality Agent

    Purpose:
    Checks whether the wearable sleep data is complete and realistic enough
    to generate a sleep insight.
    """

    issues = []

    required_fields = [
        "sleep_duration_min",
        "time_in_bed_min",
        "sleep_efficiency",
        "waso_min",
        "wake_count",
        "resting_hr",
        "hrv",
        "movement_count"
    ]

    for field in required_fields:
        if field not in data or data[field] is None:
            issues.append(f"{field} is missing")

    # Stop deeper checks if key fields are missing
    if issues:
        return {
            "quality": "Poor" if len(issues) > 2 else "Limited",
            "issues": issues
        }

    if data["sleep_duration_min"] < 180:
        issues.append("Sleep duration is too short; the record may be incomplete")

    if data["sleep_duration_min"] > data["time_in_bed_min"]:
        issues.append("Sleep duration cannot be greater than time in bed")

    if data["sleep_efficiency"] < 0 or data["sleep_efficiency"] > 100:
        issues.append("Sleep efficiency must be between 0 and 100")

    if data["resting_hr"] < 35 or data["resting_hr"] > 130:
        issues.append("Resting heart rate looks unrealistic")

    if data["hrv"] < 0:
        issues.append("HRV cannot be negative")

    if len(issues) == 0:
        quality = "Good"
    elif len(issues) <= 2:
        quality = "Limited"
    else:
        quality = "Poor"

    return {
        "quality": quality,
        "issues": issues
    }