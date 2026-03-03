# explainability.py

import shap
import numpy as np

explainer = None


def initialize_explainer(model, background_data=None):
    """
    Initialize SHAP explainer for Logistic Regression
    """

    global explainer

    try:
        # Use LinearExplainer for LogisticRegression
        explainer = shap.LinearExplainer(
            model,
            background_data,
            feature_perturbation="interventional"
        )
    except Exception:
        explainer = None


def explain_case(X_input, feature_names):
    """
    Return top contributing features
    """

    if explainer is None:
        return []

    try:
        shap_values = explainer.shap_values(X_input)

        # For multi-class, take first class
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        values = shap_values[0]

        feature_contributions = list(zip(feature_names, values))

        feature_contributions = sorted(
            feature_contributions,
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return [
            {"feature": f, "impact": round(float(v), 4)}
            for f, v in feature_contributions[:5]
        ]

    except Exception:
        return []