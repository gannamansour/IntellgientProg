"""
ml_model/predict.py
Load the saved model and predict on new patient data.
"""

import pandas as pd
import joblib

MODEL_PATH = "ml_model/decision_tree_model.pkl"


def predict(patient: dict) -> dict:
    """
    Predict heart disease risk from a patient dict.
    Values should be NORMALISED (0-1) for numerical features,
    as the model was trained on cleaned_data.csv.
    """
    model_data = joblib.load(MODEL_PATH)
    
    # Handle both old format (just model) and new format (dict with model + features)
    if isinstance(model_data, dict):
        model = model_data["model"]
        selected_features = model_data["selected_features"]
    else:
        # Backward compatibility with old model format
        model = model_data
        selected_features = list(patient.keys())
    
    # Filter patient data to selected features only
    patient_filtered = {k: patient[k] for k in selected_features if k in patient}
    
    df_in  = pd.DataFrame([patient_filtered])
    pred   = model.predict(df_in)[0]
    prob   = model.predict_proba(df_in)[0][pred]
    label  = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
    
    return {
        "prediction": int(pred), 
        "label": label, 
        "confidence": round(prob, 3),
        "features_used": selected_features
    }


if __name__ == "__main__":
    # Example: pass a row from cleaned_data (normalised values)
    sample = {
        "age": 0.72, "sex": 1, "cp": 0, "trestbps": 0.58,
        "chol": 0.44, "fbs": 1, "restecg": 0, "thalach": 0.31,
        "exang": 1, "oldpeak": 0.52, "slope": 0, "ca": 2, "thal": 3,
    }
    result = predict(sample)
    print(result)
