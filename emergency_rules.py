# emergency_rules.py

CRITICAL_SYMPTOMS = [
    "chest_pain",
    "breathlessness",
    "unconsciousness",
    "severe_bleeding",
    "seizures"
]

STROKE_SIGNS = [
    "unconsciousness",
    "slurred_speech",
    "facial_droop"
]

SEPSIS_SIGNS = [
    "high_fever",
    "shivering",
    "vomiting",
    "low_blood_pressure"
]


def check_emergency(
    data,
    predicted_disease=None,
    expected_risk_score=0,
    temporal_score=0
):
    """
    Multi-layer emergency detection logic
    """

    # ----------------------------
    # 1️⃣ Direct Critical Symptoms
    # ----------------------------
    for symptom in CRITICAL_SYMPTOMS:
        if data.get(symptom, 0) == 1:
            return True

    # ----------------------------
    # 2️⃣ Cardiac Pattern
    # ----------------------------
    if (
        data.get("chest_pain", 0) == 1 and
        data.get("breathlessness", 0) == 1
    ):
        return True

    # ----------------------------
    # 3️⃣ Stroke Pattern
    # ----------------------------
    stroke_count = sum(data.get(symptom, 0) for symptom in STROKE_SIGNS)
    if stroke_count >= 2:
        return True

    # ----------------------------
    # 4️⃣ Sepsis / Infection Shock Pattern
    # ----------------------------
    sepsis_count = sum(data.get(symptom, 0) for symptom in SEPSIS_SIGNS)
    if sepsis_count >= 3:
        return True

    # ----------------------------
    # 5️⃣ High Risk Disease Override
    # ----------------------------
    HIGH_RISK_DISEASES = [
        "Heart Attack",
        "Stroke",
        "Pulmonary Embolism"
    ]

    if predicted_disease in HIGH_RISK_DISEASES and expected_risk_score > 0.75:
        return True

    # ----------------------------
    # 6️⃣ Rapid Worsening Escalation
    # ----------------------------
    if temporal_score > 0.8:
        return True

    # ----------------------------
    # 7️⃣ Elderly Escalation
    # ----------------------------
    age = data.get("age", 30)
    if age > 70 and expected_risk_score > 0.6:
        return True

    return False