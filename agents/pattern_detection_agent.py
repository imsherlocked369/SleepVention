# agents/pattern_detection_agent.py

def pattern_detection_agent(features):
    """
    Agent 3: Pattern Detection Agent

    Purpose:
    Detects sleep patterns from extracted features.
    This agent does not write the final report.
    It only identifies patterns and supporting evidence.
    """

    patterns = []
    evidence = []

    # 1. Sleep duration
    if features["sleep_duration_min"] < 360:
        patterns.append("short_sleep")
        evidence.append("Sleep duration was below 6 hours")

    if features["sleep_duration_change"] < -45:
        patterns.append("less_sleep_than_usual")
        evidence.append(
            f"Sleep was {abs(features['sleep_duration_change'])} minutes below baseline"
        )

    # 2. Sleep efficiency
    if features["sleep_efficiency"] < 80:
        patterns.append("low_sleep_efficiency")
        evidence.append(
            f"Sleep efficiency was low at {features['sleep_efficiency']}%"
        )
    elif features["sleep_efficiency"] < 85:
        patterns.append("slightly_low_sleep_efficiency")
        evidence.append(
            f"Sleep efficiency was slightly low at {features['sleep_efficiency']}%"
        )

    # 3. Wake after sleep onset
    if features["waso_min"] > 60:
        patterns.append("high_sleep_fragmentation")
        evidence.append(
            f"Wake after sleep onset was high at {features['waso_min']} minutes"
        )
    elif features["waso_min"] > 45:
        patterns.append("moderate_sleep_fragmentation")
        evidence.append(
            f"Wake after sleep onset was elevated at {features['waso_min']} minutes"
        )

    # 4. Wake count
    if features["wake_count"] >= 7:
        patterns.append("frequent_awakenings")
        evidence.append(
            f"Wake count was high at {features['wake_count']} awakenings"
        )

    # 5. Resting heart rate
    if features["resting_hr_change"] >= 5:
        patterns.append("elevated_resting_hr")
        evidence.append(
            f"Resting heart rate was {features['resting_hr_change']} bpm above baseline"
        )

    # 6. HRV
    if features["hrv_change"] <= -8:
        patterns.append("reduced_hrv")
        evidence.append(
            f"HRV was {abs(features['hrv_change'])} ms below baseline"
        )

    # 7. Movement
    if features["movement_change"] >= 10:
        patterns.append("increased_movement")
        evidence.append(
            f"Movement count was {features['movement_change']} above baseline"
        )

    # Severity scoring
    negative_patterns = [
        "short_sleep",
        "less_sleep_than_usual",
        "low_sleep_efficiency",
        "high_sleep_fragmentation",
        "moderate_sleep_fragmentation",
        "frequent_awakenings",
        "elevated_resting_hr",
        "reduced_hrv",
        "increased_movement"
    ]

    negative_count = 0

    for pattern in patterns:
        if pattern in negative_patterns:
            negative_count += 1

    if negative_count >= 5:
        severity = "High"
    elif negative_count >= 2:
        severity = "Moderate"
    else:
        severity = "Low"

    if len(patterns) == 0:
        patterns.append("no_major_issue_detected")
        evidence.append("No major negative sleep pattern was detected")

    return {
        "patterns": patterns,
        "evidence": evidence,
        "severity": severity
    }