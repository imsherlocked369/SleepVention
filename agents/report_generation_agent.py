# agents/report_generation_agent.py

def report_generation_agent(data_quality, features, pattern_result):
    """
    Agent 4: Report Generation Agent

    Purpose:
    Converts detected patterns into a user-friendly sleep insight.
    Later, this can be replaced or upgraded with an LLM.
    """

    report = ""

    report += f"Nightly Sleep Insight - {features['date']}\n"
    report += "-" * 50 + "\n\n"

    # Main finding
    report += "1. Main finding\n"

    if pattern_result["severity"] == "High":
        report += (
            "Your wearable data suggests a clearly disturbed sleep pattern last night.\n\n"
        )
    elif pattern_result["severity"] == "Moderate":
        report += (
            "Your wearable data suggests some signs of reduced sleep quality last night.\n\n"
        )
    else:
        report += (
            "Your wearable data does not show major sleep concerns last night.\n\n"
        )

    # Evidence
    report += "2. Evidence from wearable data\n"

    for item in pattern_result["evidence"]:
        report += f"- {item}\n"

    # Interpretation
    report += "\n3. Interpretation\n"

    if "high_sleep_fragmentation" in pattern_result["patterns"] or \
       "moderate_sleep_fragmentation" in pattern_result["patterns"]:
        report += (
            "The main issue appears to be sleep fragmentation. This means you spent "
            "more time awake after initially falling asleep.\n"
        )

    if "short_sleep" in pattern_result["patterns"] or \
       "less_sleep_than_usual" in pattern_result["patterns"]:
        report += (
            "Your total sleep duration was lower than expected, which may reduce recovery "
            "and daytime energy.\n"
        )

    if "elevated_resting_hr" in pattern_result["patterns"]:
        report += (
            "Your resting heart rate was higher than usual. This may suggest reduced recovery, "
            "stress, late exercise, illness, alcohol, or other lifestyle factors.\n"
        )

    if "reduced_hrv" in pattern_result["patterns"]:
        report += (
            "Your HRV was lower than usual. This may indicate lower recovery or higher body stress, "
            "but it should be interpreted as a trend rather than a medical conclusion.\n"
        )

    if "increased_movement" in pattern_result["patterns"]:
        report += (
            "Increased movement may suggest restlessness or disturbed sleep continuity.\n"
        )

    if "no_major_issue_detected" in pattern_result["patterns"]:
        report += (
            "The available wearable signals suggest a relatively stable sleep night.\n"
        )

    # Recommendation
    report += "\n4. Recommendation\n"
    report += (
        "For tonight, try keeping a consistent bedtime, avoiding late caffeine, reducing screen time "
        "before bed, and keeping a calm pre-sleep routine.\n"
    )

    # Wearable uncertainty
    report += "\n5. Uncertainty note\n"
    report += (
        "Wearable sleep data should be interpreted as an estimate. Sleep-stage values such as deep sleep "
        "and REM sleep should be treated as approximate trends rather than exact measurements.\n"
    )

    # Safety
    report += "\n6. Data quality and safety note\n"
    report += f"Data quality: {data_quality['quality']}.\n"

    if data_quality["issues"]:
        report += "Data issues found:\n"
        for issue in data_quality["issues"]:
            report += f"- {issue}\n"

    report += (
        "\nThis is a wellness insight based on wearable data. "
        "It is not a medical diagnosis.\n"
    )

    return report