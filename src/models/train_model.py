import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import mlflow # Added for monitoring

ARTIFACTS_DIR = "artifacts"

def train_model(X, y):
    """
    Model training with SMOTE for class imbalance. 
    Matches the Notebook experiment results.
    """
    # Keep stratify=y to ensure test set represents original distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Balance training data only (Never SMOTE the test set!)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    # Random Forest tuned as per notebook
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42
    )
    
    model.fit(X_res, y_res)

    # Eval
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) # Extracted for MLflow
    
    # --- MLFLOW LOGGING ---
    # Log parameters and metrics without affecting training flow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", acc)
    # ----------------------

    print(f"--- Training Complete ---\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # Persistence
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.pkl"))
