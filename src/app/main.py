import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field   
import uvicorn

# Adding project root for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

app = FastAPI(title="Diabetes API - Feature Engineering Fixed")
artifacts = {}
CLASSIFICATION_THRESHOLD = 0.40

# 2. Define the request schema 
class DiabetesInput(BaseModel):
    """
    Data Transfer Object (DTO) for diabetes prediction.
    Using 'Field' for strict schema validation.
    """
    # 'ge=0' ensures Greater than or Equal to zero
    Pregnancies: int = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=1, le=120)

@app.on_event("startup")
def load_production_artifacts():
    try:
        artifacts["model"] = joblib.load("artifacts/model.pkl")
        artifacts["scaler"] = joblib.load("artifacts/scaler.pkl")
        artifacts["columns"] = joblib.load("artifacts/columns.pkl")
        print(f"✅ Production System Online. Threshold: {CLASSIFICATION_THRESHOLD}")
    except Exception as e:
        print(f"❌ Initialization Error: {e}")


@app.post("/predict")
def predict(data: DiabetesInput):
    try:
        # 1. Input Parsing
        input_df = pd.DataFrame([data.model_dump()])
        
        # 2. Preprocessing
        X_clean = preprocess_data(input_df, is_training=False)

        # 3. Feature Engineering
        feat_result = build_features(X_clean, is_training=False)
        X_features = feat_result[0] if isinstance(feat_result, tuple) else feat_result

        # 4. Alignment
        if isinstance(X_features, np.ndarray):
            X_features = pd.DataFrame(X_features, columns=artifacts["columns"])
        
        X_aligned = X_features.reindex(columns=artifacts["columns"], fill_value=0)

        # --- THE FIX: REMOVE REDUNDANT SCALING ---
        # Since DEBUG showed '2.015', your functions are ALREADY scaling the data.
        # We will use X_aligned directly for prediction.
        
        # We skip artifacts["scaler"].transform and go straight to the model
        X_final_input = X_aligned.values # Convert to array for the model
        
        # 5. Model Inference
        probability = artifacts["model"].predict_proba(X_final_input)[0][1]
        prediction = 1 if probability >= CLASSIFICATION_THRESHOLD else 0

        # --- DEBUG LOG FOR CONFIRMATION ---
        print(f"DEBUG: Prob from model: {probability}")
        # ----------------------------------

        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "status": "Success"
        }

    except Exception as e:
        print(f"❌ Inference Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
    
