# risk_mapping.py

# ---------------------------------------------------
# 1️⃣ Original Risk Category Mapping
# ---------------------------------------------------

DISEASE_RISK_MAP = {

    # LOW
    "Fungal infection": "Low",
    "Allergy": "Low",
    "GERD": "Low",
    "Chronic cholestasis": "Low",
    "Drug Reaction": "Low",
    "Peptic ulcer diseae": "Low",
    "Gastroenteritis": "Low",
    "Migraine": "Low",
    "Cervical spondylosis": "Low",
    "Dimorphic hemmorhoids(piles)": "Low",
    "Varicose veins": "Low",
    "Hypothyroidism": "Low",
    "Hyperthyroidism": "Low",
    "Osteoarthristis": "Low",
    "Arthritis": "Low",
    "(vertigo) Paroymsal  Positional Vertigo": "Low",
    "Acne": "Low",
    "Urinary tract infection": "Low",
    "Psoriasis": "Low",
    "Impetigo": "Low",
    "Common Cold": "Low",

    # MODERATE
    "Diabetes ": "Moderate",
    "Hypertension ": "Moderate",
    "Hypoglycemia": "Moderate",
    "Jaundice": "Moderate",
    "Malaria": "Moderate",
    "Chicken pox": "Moderate",
    "Dengue": "Moderate",
    "Typhoid": "Moderate",
    "hepatitis A": "Moderate",
    "Hepatitis E": "Moderate",
    "Alcoholic hepatitis": "Moderate",
    "Bronchial Asthma": "Moderate",

    # HIGH
    "AIDS": "High",
    "Paralysis (brain hemorrhage)": "High",
    "Heart attack": "High",
    "Tuberculosis": "High",
    "Pneumonia": "High",
    "Hepatitis B": "High",
    "Hepatitis C": "High",
    "Hepatitis D": "High"
}


# ---------------------------------------------------
# 2️⃣ Convert Risk Category to Impact Weight
# ---------------------------------------------------

RISK_IMPACT_WEIGHTS = {
    "Low": 0.3,
    "Moderate": 0.6,
    "High": 1.0
}


# ---------------------------------------------------
# 3️⃣ Probability-Aware Risk Calculation
# ---------------------------------------------------

def compute_expected_risk(predicted_disease, probability):
    """
    Expected Risk = Model Probability × Disease Impact Weight
    """

    risk_category = DISEASE_RISK_MAP.get(predicted_disease, "Low")

    impact_weight = RISK_IMPACT_WEIGHTS.get(risk_category, 0.3)

    expected_risk_score = probability * impact_weight

    return expected_risk_score