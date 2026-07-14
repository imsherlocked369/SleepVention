# SleepVention

SleepVention is an AI-agent-based prototype for interpreting wearable sleep data with machine learning-powered pattern detection.

The system takes wearable sleep metrics such as sleep duration, sleep efficiency, wake after sleep onset (WASO), resting heart rate, HRV, and movement count, then generates a personalized, non-diagnostic sleep insight by identifying meaningful sleep patterns and severity levels.

## Architecture Overview

The system uses a **4-agent pipeline** where each agent has a specific responsibility:

```
Raw Sleep Data
      ↓
[Agent 1: Data Quality] → Validates completeness and realism
      ↓
[Agent 2: Feature Extraction] → Extracts features and compares with baseline
      ↓
[Agent 3: Pattern Detection] → Identifies and labels sleep patterns (ML-enhanced)
      ↓
[Agent 4: Report Generation] → Converts patterns into user-friendly narrative
      ↓
Personalized Sleep Insight
```

## Agents

### 1. Data Quality Agent
**Purpose:** Validates wearable data quality before analysis.

**Responsibilities:**
- Checks for missing required fields
- Verifies realistic value ranges (e.g., RHR 35-130 bpm, sleep efficiency 0-100%)
- Flags incomplete or suspicious data

**Output:** Quality verdict (`Good`, `Limited`, or `Poor`) + list of issues

---

### 2. Feature Extraction Agent
**Purpose:** Converts raw metrics into meaningful features for pattern analysis.

**Responsibilities:**
- Extracts key sleep metrics (duration, efficiency, fragmentation, sleep stages)
- Compares current night against user's baseline (rolling average)
- Calculates deltas: `sleep_duration_change`, `resting_hr_change`, `hrv_change`, etc.

**Output:** Structured feature dict with absolute values + baseline comparisons

---

### 3. Pattern Detection Agent (ML-Enhanced)
**Purpose:** Identifies sleep patterns using a hybrid rule-based + ML approach.

**Key Innovation:** Uses **Isolation Forest** anomaly detection to catch multivariate patterns that rule-based systems miss.

#### How It Works:

**Hybrid Detection Strategy:**
1. **Rule-based layer** (explicit thresholds for obvious extremes):
   - Severe short sleep: `< 300 minutes`
   - Critically low efficiency: `< 75%`
   
2. **ML-based layer** (anomaly scoring):
   - Trains on user's historical sleep data (20+ nights recommended)
   - Learns what "normal" looks like for that individual
   - Detects subtle multivariate deviations (e.g., "slightly elevated HR + slightly reduced deep sleep + borderline efficiency = unusual night")
   - Returns anomaly score (-1 to 1, where -1 = extreme anomaly)

3. **Pattern Labeling**:
   - Even when ML detects anomalies, Agent 3 explicitly names which patterns are present
   - Maintains interpretability for downstream reporting
   - Ensures Agent 4 can explain *why* the night was concerning

#### Detected Patterns:
- `short_sleep`: Sleep duration < 6 hours
- `less_sleep_than_usual`: Sleep 45+ minutes below baseline
- `low_sleep_efficiency`: Efficiency < 80%
- `slightly_low_sleep_efficiency`: Efficiency < 85%
- `high_sleep_fragmentation`: WASO > 60 min
- `moderate_sleep_fragmentation`: WASO > 45 min
- `frequent_awakenings`: Wake count ≥ 7
- `elevated_resting_hr`: RHR ≥ 5 bpm above baseline
- `reduced_hrv`: HRV ≥ 8 ms below baseline
- `increased_movement`: Movement count ≥ 10 above baseline
- `no_major_issue_detected`: No concerning patterns

#### Severity Scoring:
- **High**: 5+ negative patterns OR extreme anomaly score (< -0.6)
- **Moderate**: 2-4 patterns OR moderate anomaly score (< -0.3)
- **Low**: ≤ 1 pattern AND normal anomaly score

**Output:** 
```python
{
    "patterns": ["short_sleep", "moderate_sleep_fragmentation"],
    "evidence": ["Sleep duration was below 6 hours", "WASO was elevated at 52 minutes"],
    "severity": "Moderate",
    "ml_anomaly_score": -0.42  # For transparency and debugging
}
```

---

### 4. Report Generation Agent
**Purpose:** Converts detected patterns into a user-friendly, actionable sleep insight.

**Responsibilities:**
- Builds narrative based on detected patterns
- Provides evidence from wearable metrics
- Interprets patterns in plain language
- Offers context-aware sleep recommendations
- Includes uncertainty disclaimers

**Output:** Formatted sleep report with findings, interpretation, and recommendations

---

## Machine Learning Approach

### Why Hybrid (Rules + ML)?

| Aspect | Pure Rules | Pure ML | Hybrid ✅ |
|--------|-----------|---------|----------|
| **User Adaptation** | ❌ One-size-fits-all thresholds | ✅ Learns individual patterns | ✅ Both |
| **Explainability** | ✅ Transparent | ❌ Black box | ✅ Transparent |
| **Multivariate Patterns** | ❌ Limited | ✅ Excellent | ✅ Excellent |
| **Data Requirements** | ✅ None | ❌ Needs labels | ✅ Unsupervised |
| **False Positives** | ⚠️ High | ⚠️ Depends on tuning | ✅ Reduced |

### Algorithm: Isolation Forest

**Why Isolation Forest?**
- **Unsupervised**: No labeled training data needed (you have 40 nights, not labeled)
- **Multivariate**: Detects patterns across multiple dimensions simultaneously
- **Adaptive**: Learns user's personal sleep distribution
- **Efficient**: Fast inference for real-time reports
- **Robust**: Works well with small datasets (20-40 nights)

**How It Works:**
- Isolation Forest learns the "normal" sleep pattern by isolating anomalies
- For new sleep data, it calculates how "unusual" the combination of metrics is
- Returns a score: -1 (extremely anomalous) to +1 (extremely normal)

### Training the Model

```python
# One-time setup on historical sleep data
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load historical sleep data
df = pd.read_csv('sleep.csv')

# Features for ML
features = [
    'minutesAsleep', 'minutesAwake', 'efficiency',
    'deep_minutes', 'light_minutes', 'rem_minutes', 'wake_count'
]

X = df[features]

# Train Isolation Forest (contamination = expected % of anomalies)
clf = IsolationForest(contamination=0.15, random_state=42)
clf.fit(X)

# Save for use in pattern detection agent
joblib.dump(clf, 'models/sleep_anomaly_detector.pkl')
```

### Using the Model in Pattern Detection

The trained model is loaded and used during runtime to calculate anomaly scores, which inform severity classification without replacing the rule-based pattern labels.

---

## Project Structure

```
SleepVention/
│
├── README.md
├── requirements.txt
├── main.py
├── sleep_data.py
│
├── agents/
│   ├── __init__.py
│   ├── data_quality_agent.py
│   ├── feature_extraction_agent.py
│   ├── pattern_detection_agent.py (ML-enhanced)
│   └── report_generation_agent.py
│
├── models/
│   └── sleep_anomaly_detector.pkl (trained Isolation Forest)
│
└── data/
    └── sleep.csv (historical sleep records for training)
```

---

## Key Design Principles

1. **Interpretability First**: Every pattern is explicitly labeled and explained. Users and medical professionals can understand why a night was flagged.

2. **Hybrid Approach**: Rules handle obvious extremes; ML detects subtle multivariate patterns. Best of both worlds.

3. **User-Centric**: Baselines and ML models adapt to individual sleep profiles, not universal thresholds.

4. **Non-Diagnostic**: All outputs include disclaimers that wearable data is an estimate and should not replace professional medical advice.

5. **Agent Separation of Concerns**: Each agent has a single, clear responsibility in the pipeline.

---

## Usage

```python
from sleep_data import sleep_data, baseline_data
from agents.data_quality_agent import data_quality_agent
from agents.feature_extraction_agent import feature_extraction_agent
from agents.pattern_detection_agent import pattern_detection_agent
from agents.report_generation_agent import report_generation_agent

# Run the pipeline
data_quality = data_quality_agent(sleep_data)
features = feature_extraction_agent(sleep_data, baseline_data)
pattern_result = pattern_detection_agent(features)
final_report = report_generation_agent(data_quality, features, pattern_result)

print(final_report)
```

---

## Future Enhancements

- **LLM Integration**: Replace template-based report generation with an LLM for more natural, contextual narratives
- **Multi-User Support**: Extend to track patterns across multiple users
- **Longitudinal Analysis**: Identify trends over weeks/months (e.g., "sleep quality declining over time")
- **Integration with Wearables APIs**: Real-time data ingestion from Fitbit, Oura, Apple Watch, etc.
- **Intervention Recommendations**: Suggest lifestyle changes based on detected patterns
- **Sleep Stage Deep Dive**: Enhanced analysis of deep sleep and REM patterns for recovery optimization
