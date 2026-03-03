from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import uuid
import threading
from datetime import datetime

# Import your custom modules
from emergency_rules import check_emergency
from hospital_locator import get_nearest_hospitals
from risk_mapping import compute_expected_risk
from severity_rules import calculate_severity_score, score_to_risk
from temporal_features import compute_temporal_score
from feedback_logger import (
    log_prediction, update_outcome,
    save_to_history, load_history,
    delete_history_record, clear_all_history,
    save_incremental_sample, get_incremental_sample_count
)
from explainability import initialize_explainer, explain_case
from incremental_trainer import retrain_model, get_accuracy_history, should_retrain

app = Flask(__name__)

# ─────────────────────────────────────────
# Load model & encoder
# ─────────────────────────────────────────
model = joblib.load("disease_model.pkl")
mlb   = joblib.load("symptom_encoder.pkl")

background = np.zeros((1, len(mlb.classes_)))
initialize_explainer(model, background)

RISK_TO_NUM = {"Low": 0, "Moderate": 1, "High": 2}
NUM_TO_RISK  = {0: "Low", 1: "Moderate", 2: "High"}
CONFIDENCE_THRESHOLD = 40


# ─────────────────────────────────────────
# Helper: background retrain
# ─────────────────────────────────────────
def _background_retrain():
    try:
        result = retrain_model()
        if result.get("success"):
            global model, mlb
            model = joblib.load("disease_model.pkl")
            mlb   = joblib.load("symptom_encoder.pkl")
            background = np.zeros((1, len(mlb.classes_)))
            initialize_explainer(model, background)
            print(f"[App] Model hot-reloaded. Accuracy: {result['accuracy']}%")
    except Exception as e:
        print(f"[App] Retrain failed: {e}")


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    case_id = str(uuid.uuid4())

    # Active symptoms
    input_symptoms = [s for s in mlb.classes_ if data.get(s, 0) == 1]
    X_input = mlb.transform([input_symptoms])

    # Disease prediction
    predicted_disease = model.predict(X_input)[0]
    probabilities     = model.predict_proba(X_input)[0]
    max_prob          = float(np.max(probabilities))
    confidence        = round(max_prob * 100, 2)

    expected_risk_score = compute_expected_risk(predicted_disease, max_prob)

    time_since_onset = data.get("time_since_onset", 1)
    worsening_flag   = data.get("worsening_flag", 0)
    temporal_score   = compute_temporal_score(time_since_onset, worsening_flag)

    # ── Save incremental sample ──
    if input_symptoms:
        save_incremental_sample(input_symptoms, predicted_disease)
        sample_count = get_incremental_sample_count()
        if should_retrain(sample_count):
            t = threading.Thread(target=_background_retrain, daemon=True)
            t.start()

    # ── Emergency override ──
    is_emergency = check_emergency(
        data,
        predicted_disease=predicted_disease,
        expected_risk_score=expected_risk_score,
        temporal_score=temporal_score
    )

    if is_emergency:
        hospitals = get_nearest_hospitals(data.get("latitude"), data.get("longitude"), emergency=True)
        try:
            top_features = explain_case(X_input, mlb.classes_)
        except Exception:
            top_features = []

        result = {
            "case_id": case_id,
            "predicted_disease": predicted_disease,
            "risk": "High",
            "confidence": 100,
            "emergency": True,
            "temporal_score": round(temporal_score, 2),
            "expected_risk_score": round(expected_risk_score, 2),
            "top_contributing_features": top_features,
            "nearest_hospitals": hospitals
        }
        log_prediction({**result, "age": data.get("age")})
        # Auto-save history
        _save_history(data, result, input_symptoms)
        return jsonify(result)

    # ── Normal risk computation ──
    severity_score = calculate_severity_score(data)
    severity_risk  = score_to_risk(severity_score)

    disease_risk = 2 if expected_risk_score > 0.7 else (1 if expected_risk_score > 0.4 else 0)
    if confidence < CONFIDENCE_THRESHOLD:
        disease_risk = 0

    combined_score = 0.4 * severity_risk + 0.4 * disease_risk + 0.2 * temporal_score
    final_risk = 2 if combined_score >= 1.5 else (1 if combined_score >= 0.7 else 0)

    age = data.get("age", 30)
    if age > 60 and final_risk < 2:
        final_risk += 1

    hospitals = []
    # Show hospitals for Moderate and High risk
    if final_risk >= 1:
        hospitals = get_nearest_hospitals(
            data.get("latitude"),
            data.get("longitude"),
            emergency=(final_risk == 2)  # emergency=True only for High
        )

    try:
        top_features = explain_case(X_input, mlb.classes_)
    except Exception:
        top_features = []

    result = {
        "case_id": case_id,
        "predicted_disease": predicted_disease,
        "risk": NUM_TO_RISK[final_risk],
        "confidence": confidence,
        "emergency": False,
        "temporal_score": round(temporal_score, 2),
        "expected_risk_score": round(expected_risk_score, 2),
        "top_contributing_features": top_features,
        "nearest_hospitals": hospitals
    }
    
    log_prediction({
        "case_id": case_id,
        "symptoms": input_symptoms,
        "predicted_disease": predicted_disease,
        "risk": NUM_TO_RISK[final_risk],
        "confidence": confidence,
        "age": age,
        "time_since_onset": time_since_onset,
        "worsening_flag": worsening_flag
    })
    
    # Auto-save history
    _save_history(data, result, input_symptoms)
    return jsonify(result)


# ─────────────────────────────────────────
# History Helper
# ─────────────────────────────────────────
def _save_history(data, result, input_symptoms):
    """Build and save a rich history record with timestamp"""
    save_to_history({
        "case_id":            result["case_id"],
        "predicted_disease":  result["predicted_disease"],
        "risk":               result["risk"],
        "confidence":         result["confidence"],
        "emergency":          result["emergency"],
        "expected_risk_score":result["expected_risk_score"],
        "temporal_score":     result["temporal_score"],
        "severity":           data.get("severity_self", 5),
        "symptoms":           input_symptoms,
        "age":                data.get("age"),
        "gender":             data.get("gender", ""),
        "duration":           data.get("duration", ""),
        "trend":              data.get("trend", ""),
        # ─── FIX: Add timestamp for frontend display ───
        "timestamp":          datetime.now().isoformat()
    })


# ─────────────────────────────────────────
# History Endpoints
# ─────────────────────────────────────────
@app.route("/history", methods=["GET", "POST"])
def handle_history():
    # ─── FIX: Handle POST request to save history explicitly ───
    if request.method == "POST":
        data = request.json
        # Ensure case_id exists
        if not data.get("case_id"):
            data["case_id"] = str(uuid.uuid4())
        
        # Add timestamp if missing
        if not data.get("timestamp"):
            data["timestamp"] = datetime.now().isoformat()
            
        save_to_history(data)
        return jsonify({"status": "success", "case_id": data["case_id"]}), 201
    
    # GET request returns history
    return jsonify(load_history())


@app.route("/history/<case_id>", methods=["DELETE"])
def delete_history(case_id):
    delete_history_record(case_id)
    return jsonify({"message": "Deleted"})


@app.route("/history/clear", methods=["POST"])
def clear_history():
    clear_all_history()
    return jsonify({"message": "History cleared"})


# ─────────────────────────────────────────
# Model Stats Endpoint
# ─────────────────────────────────────────
@app.route("/model_stats", methods=["GET"])
def model_stats():
    acc_history = get_accuracy_history()
    sample_count = get_incremental_sample_count()
    return jsonify({
        "incremental_samples": sample_count,
        "next_retrain_in": max(0, 5 - (sample_count % 5)),
        "accuracy_history": acc_history[-10:],   # last 10 retrains
        "latest_accuracy": acc_history[-1]["accuracy"] if acc_history else None
    })


# ─────────────────────────────────────────
# Other existing endpoints
# ─────────────────────────────────────────
@app.route("/update_outcome", methods=["POST"])
def update_case_outcome():
    data = request.json
    case_id = data.get("case_id")
    actual_outcome = data.get("actual_outcome")
    if not case_id or not actual_outcome:
        return jsonify({"error": "Missing case_id or actual_outcome"}), 400
    success = update_outcome(case_id, actual_outcome)
    if success:
        return jsonify({"message": "Outcome updated successfully"})
    return jsonify({"error": "Case not found"}), 404


@app.route("/symptoms")
def get_symptoms():
    return jsonify(list(mlb.classes_))


if __name__ == "__main__":
    app.run(debug=True)