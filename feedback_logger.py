import pandas as pd
import os
import json
from datetime import datetime

LOG_FILE = "triage_feedback_log.csv"
HISTORY_FILE = "assessment_history.json"
INCREMENTAL_DATA_FILE = "incremental_training_data.csv"


# ─────────────────────────────────────────
# Core Prediction Logger
# ─────────────────────────────────────────
def log_prediction(case_data):
    """Log initial triage prediction to CSV with symptoms included"""

    case_data["timestamp"] = datetime.now().isoformat()
    case_data["actual_outcome"] = None

    # 🔥 Convert symptoms list to JSON string (important!)
    if isinstance(case_data.get("symptoms"), list):
        case_data["symptoms"] = json.dumps(case_data["symptoms"])

    df = pd.DataFrame([case_data])

    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)


def update_outcome(case_id, actual_outcome):
    """Update real hospital outcome later"""
    if not os.path.exists(LOG_FILE):
        return False
    df = pd.read_csv(LOG_FILE)
    if case_id not in df["case_id"].values:
        return False
    df.loc[df["case_id"] == case_id, "actual_outcome"] = actual_outcome
    df.to_csv(LOG_FILE, index=False)
    return True


# ─────────────────────────────────────────
# Assessment History (for UI)
# ─────────────────────────────────────────
def save_to_history(assessment_record):
    """
    Save a full assessment to the history JSON file.
    """
    history = load_history()
    assessment_record["timestamp"] = datetime.now().isoformat()
    history.insert(0, assessment_record)
    history = history[:100]  # keep max 100
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_history():
    """Load all assessment history"""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def delete_history_record(case_id):
    """Delete a single record by case_id"""
    history = load_history()
    history = [h for h in history if h.get("case_id") != case_id]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    return True


def clear_all_history():
    """Wipe the entire history"""
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)
    return True


# ─────────────────────────────────────────
# Incremental Training Data Collector
# ─────────────────────────────────────────
def save_incremental_sample(symptoms_list, predicted_disease):
    """
    Save a new symptom->disease sample for incremental training.
    """
    record = {"Disease": predicted_disease, "timestamp": datetime.now().isoformat()}
    for i, sym in enumerate(symptoms_list[:17]):
        record[f"Symptom_{i+1}"] = sym
    for i in range(len(symptoms_list), 17):
        record[f"Symptom_{i+1}"] = None

    df = pd.DataFrame([record])
    if not os.path.exists(INCREMENTAL_DATA_FILE):
        df.to_csv(INCREMENTAL_DATA_FILE, index=False)
    else:
        df.to_csv(INCREMENTAL_DATA_FILE, mode="a", header=False, index=False)


def get_incremental_sample_count():
    """Return number of new samples collected"""
    if not os.path.exists(INCREMENTAL_DATA_FILE):
        return 0
    try:
        df = pd.read_csv(INCREMENTAL_DATA_FILE)
        return len(df)
    except Exception:
        return 0