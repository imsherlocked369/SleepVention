# SleepInsightAgent

SleepInsightAgent is an initial AI-agent-based prototype for interpreting wearable sleep data.

The system takes wearable sleep metrics such as sleep duration, sleep efficiency, wake after sleep onset, resting heart rate, HRV, and movement count, then generates a personalized, non-diagnostic sleep insight.

## Current Agents

1. Data Quality Agent  
   Checks whether the wearable sleep data is complete and realistic.

2. Feature Extraction Agent  
   Extracts useful sleep features and compares them with baseline values.

3. Pattern Detection Agent  
   Detects sleep patterns such as short sleep, fragmentation, elevated resting heart rate, reduced HRV, and increased movement.

4. Report Generation Agent  
   Converts the detected patterns into a user-friendly sleep report.

## Project Structure

```text
SleepInsightAgent/
│
├── main.py
├── sleep_data.py
│
└── agents/
    ├── __init__.py
    ├── data_quality_agent.py
    ├── feature_extraction_agent.py
    ├── pattern_detection_agent.py
    └── report_generation_agent.py