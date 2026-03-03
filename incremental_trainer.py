# incremental_trainer.py
"""
Incremental model retraining.
Called after every N new assessments to keep the model improving.
"""

import pandas as pd
import numpy as np
import os
import random
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DATASET = "d1.csv"
INCREMENTAL_DATA_FILE = "incremental_training_data.csv"
MODEL_FILE = "disease_model.pkl"
ENCODER_FILE = "symptom_encoder.pkl"
ACCURACY_LOG = "model_accuracy_log.json"

# Retrain after every N new samples
RETRAIN_THRESHOLD = 5


def should_retrain(new_sample_count):
    """Check if we have enough new samples to retrain"""
    return new_sample_count > 0 and new_sample_count % RETRAIN_THRESHOLD == 0


def load_combined_dataset():
    """Combine base dataset + incremental data"""
    if not os.path.exists(BASE_DATASET):
        return None

    base_df = pd.read_csv(BASE_DATASET)
    base_df["Disease"] = base_df["Disease"].str.strip()

    if os.path.exists(INCREMENTAL_DATA_FILE):
        try:
            inc_df = pd.read_csv(INCREMENTAL_DATA_FILE)
            inc_df["Disease"] = inc_df["Disease"].str.strip()
            # Drop timestamp column if present
            inc_df = inc_df.drop(columns=["timestamp"], errors="ignore")
            combined = pd.concat([base_df, inc_df], ignore_index=True)
            print(f"[Trainer] Base: {len(base_df)} rows + Incremental: {len(inc_df)} rows = {len(combined)} total")
            return combined
        except Exception as e:
            print(f"[Trainer] Could not load incremental data: {e}")
            return base_df

    return base_df


def augment_data(df, augment_factor=4):
    """Generate partial-symptom augmentation"""
    symptom_columns = [col for col in df.columns if "Symptom" in col]
    df["Symptoms_List"] = df[symptom_columns].values.tolist()
    df["Symptoms_List"] = df["Symptoms_List"].apply(
        lambda x: [str(i).strip() for i in x if pd.notna(i) and str(i).strip() != "nan"]
    )

    augmented_rows = []
    for _, row in df.iterrows():
        symptoms = row["Symptoms_List"]
        disease = row["Disease"]
        augmented_rows.append((symptoms, disease))
        for _ in range(augment_factor):
            if len(symptoms) > 1:
                subset = random.sample(symptoms, random.randint(1, len(symptoms)))
            else:
                subset = symptoms
            augmented_rows.append((subset, disease))

    return pd.DataFrame(augmented_rows, columns=["Symptoms_List", "Disease"])


def retrain_model():
    """
    Full retrain on base + incremental data.
    Returns dict with accuracy info.
    """
    print("[Trainer] Starting incremental retrain...")

    df = load_combined_dataset()
    if df is None:
        return {"success": False, "error": "Base dataset not found"}

    aug_df = augment_data(df)

    mlb = MutiLabelBinarizer()
    X = mlb.fit_transform(aug_df["Symptoms_List"])
    y = aug_df["Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=3000, C=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    # Save updated model
    joblib.dump(model, MODEL_FILE)
    joblib.dump(mlb, ENCODER_FILE)

    # Log accuracy history
    import json
    log = []
    if os.path.exists(ACCURACY_LOG):
        try:
            with open(ACCURACY_LOG) as f:
                log = json.load(f)
        except Exception:
            log = []

    from datetime import datetime
    log.append({
        "timestamp": datetime.now().isoformat(),
        "accuracy": accuracy,
        "total_samples": len(aug_df),
        "incremental_samples": len(pd.read_csv(INCREMENTAL_DATA_FILE)) if os.path.exists(INCREMENTAL_DATA_FILE) else 0
    })

    with open(ACCURACY_LOG, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[Trainer] Retrain complete. New accuracy: {accuracy}%")
    return {
        "success": True,
        "accuracy": accuracy,
        "total_samples": len(aug_df)
    }


def get_accuracy_history():
    """Return accuracy log"""
    import json
    if not os.path.exists(ACCURACY_LOG):
        return []
    try:
        with open(ACCURACY_LOG) as f:
            return json.load(f)
    except Exception:
        return []


# Fix typo in function
MutiLabelBinarizer = MultiLabelBinarizer