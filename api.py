from flask import Flask, request, jsonify
import pandas as pd
import joblib
import shap

app = Flask(__name__)

# Load model and feature list
model = joblib.load("invoice_anomaly_model.pkl")
features = joblib.load("invoice_model_features.pkl")

# Background dataset for SHAP
background = pd.DataFrame([[0]*len(features)], columns=features)
explainer = shap.Explainer(model.decision_function, background)

@app.route("/")
def home():
    return jsonify({
        "message": "Invoice Anomaly API Running",
        "endpoint": "/predict/invoice",
        "method": "POST"
    })

@app.route("/health")
def health():
    return jsonify({"status": "API is running"})

@app.route("/predict/invoice", methods=["POST"])
def predict_invoice():
    try:
        data = request.get_json()

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Feature engineering
        if 'usage_units' in df.columns and 'usage_rate' in df.columns:
            df['expected_charge'] = df['usage_units'] * df['usage_rate']
        else:
            df['expected_charge'] = 0

        if 'debit' in df.columns:
            df['charge_diff'] = df['debit'] - df['expected_charge']
        else:
            df['charge_diff'] = 0

        # Align features
        X = df.reindex(columns=features, fill_value=0)

        # Model prediction
        pred = model.predict(X)[0]
        score = model.decision_function(X)[0]

        # SHAP explanation
        shap_values = explainer(X)
        shap_dict = dict(zip(features, shap_values.values[0]))

        # Top reason
        top_reason = max(shap_dict, key=lambda k: abs(shap_dict[k]))

        # Reason description
        reason_map = {
            "debit": "Debit amount deviates from expected charge",
            "expected_charge": "Expected charge calculation unusual",
            "usage_units": "Usage units abnormal",
            "usage_rate": "Usage rate abnormal",
            "charge_diff": "Difference between debit and expected charge high",
            "duration_days": "Billing duration unusual",
            "proration_remaining_days": "Proration days unusual"
        }

        reason_description = reason_map.get(top_reason, "Unknown reason")

        # Response
        response = {
            "anomaly": int(1 if pred == -1 else 0),
            "anomaly_score": float(score),
            "top_reason": top_reason,
            "reason_description": reason_description,
            "explanation": {
                k: float(v) for k, v in shap_dict.items()
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
