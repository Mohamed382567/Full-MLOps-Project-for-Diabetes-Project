# run_pipeline.py
import pandas as pd
import os
import shutil
import sys
import mlflow
# --- 1. إعدادات MLflow للتتبع عن بُعد ---
# يتم سحب هذه المتغيرات من GitHub Secrets عند تشغيل الـ Pipeline
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✅ Remote tracking enabled: {tracking_uri}")
else:
    print("ℹ️ Local tracking enabled (mlruns folder).")

# Add src to the system path to allow importing modules (CRITICAL for MLOps structure)
# This allows the script to find src.data.load_data, etc.
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all necessary functions from your modules
from data.load_data import load_data
from data.validate_data import validate_data
from data.preprocess import preprocess_data
from features.build_features import build_features
from models.train_model import train_model

# --- Configuration ---
# ⚠️ CRITICAL: Set the correct path to your data file
# Assuming 'diabetes.csv' is inside a folder named 'data'
DATA_PATH = 'data/diabetes.csv' 
ARTIFACTS_DIR = "artifacts"

def clean_old_artifacts():
    """Deletes old model files (artifacts) to ensure a clean slate before training."""
    if os.path.exists(ARTIFACTS_DIR):
        print(f"🗑️ Deleting old artifacts folder at {ARTIFACTS_DIR}...")
        shutil.rmtree(ARTIFACTS_DIR)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("✅ Artifacts directory is ready for new files.")

def main():
    print("-" * 50)
    print("      MLOps Pipeline Execution Started")
    print("-" * 50)
    
    # --- 1. Clean and Load ---
    clean_old_artifacts()
    
    try:
        # Load the raw dataset using the defined path
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        print("🛑 ACTION REQUIRED: Please ensure 'diabetes.csv' is inside the 'data' folder and run again.")
        return
    except Exception as e:
        print(f"🛑 CRITICAL ERROR during data loading: {e}")
        return

    # --- 2. Validation ---
    validate_data(df)
    
    # --- 3. Split Data Manually (CRITICAL STEP) ---
    # Separate the target variable (y) from features (X) BEFORE any processing.
    print("✂️ Separating target variable (Outcome) from features...")
    y = df['Outcome']
    X_raw = df.drop(columns=['Outcome']) 

    # --- 4. Preprocessing (Imputation) ---
    # X_raw is passed, ensuring 'Outcome' is NOT seen by the Imputer.
    X_imputed = preprocess_data(X_raw, is_training=True)

    # --- 5. Feature Engineering & Scaling ---
    # X_imputed is passed, ensuring 'Outcome' is NOT seen by the Scaler/Feature Engineering.
    # build_features will automatically handle the scaling and save the scaler/column names.
    X_scaled, _ = build_features(X_imputed, is_training=True)

    # --- 6. Train Model ---
    # The clean, scaled features (X_scaled) and the correct target (y) are used for final training.
    train_model(X_scaled, y)

    print("\n" + "=" * 50)
    print("🚀 PIPELINE FINISHED SUCCESSFULLY!")
    print(f"New artifacts saved in the '{ARTIFACTS_DIR}/' directory.")
    print("= ACTION REQUIRED: Upload all new artifacts and updated code to GitHub.")
    print("=" * 50)

if __name__ == "__main__":
    main()