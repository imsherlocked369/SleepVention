# main.py

from sleep_data import sleep_data, baseline_data

from agents.data_quality_agent import data_quality_agent
from agents.feature_extraction_agent import feature_extraction_agent
from agents.pattern_detection_agent import pattern_detection_agent
from agents.report_generation_agent import report_generation_agent


def main():
    print("Running SleepInsight Agent...\n")

    # Agent 1: Check data quality
    data_quality = data_quality_agent(sleep_data)

    # Agent 2: Extract features and compare with baseline
    features = feature_extraction_agent(sleep_data, baseline_data)

    # Agent 3: Detect sleep patterns
    pattern_result = pattern_detection_agent(features)

    # Agent 4: Generate final sleep report
    final_report = report_generation_agent(
        data_quality=data_quality,
        features=features,
        pattern_result=pattern_result
    )

    print(final_report)


if __name__ == "__main__":
    main()