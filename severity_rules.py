# severity_rules.py

def calculate_severity_score(data):
    score = 0

    # Critical symptoms
    if data.get("chest_pain", 0) == 1:
        score += 3
    if data.get("breathlessness", 0) == 1:
        score += 3
    if data.get("unconsciousness", 0) == 1:
        score += 4

    # High severity
    if data.get("high_fever", 0) == 1:
        score += 2
    if data.get("persistent_vomiting", 0) == 1 or data.get("vomiting", 0) == 1:
        score += 2
    if data.get("chills", 0) == 1:
        score += 1
    if data.get("shivering", 0) == 1:
        score += 1

    # Acidity / GERD related
    if data.get("acidity", 0) == 1:
        score += 1
    if data.get("stomach_pain", 0) == 1 and data.get("acidity", 0) == 1:
        score += 1  # combined acidity + stomach pain = higher concern

    return score


def score_to_risk(score):
    if score >= 6:
        return 2  # High
    elif score >= 3:
        return 1  # Moderate
    else:
        return 0  # Low