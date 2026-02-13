import pandas as pd
import os
import shutil
import sys
import mlflow

# --- MLFLOW TRACKING SETUP ---
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    # Set experiment name for better organization in DagsHub
    mlflow.set_experiment("Diabetes_Inference_Monitoring")
    print(f"✅ Remote tracking enabled: {tracking_uri}")
else:
    print("ℹ️ Local tracking enabled (mlruns folder).")

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.load_data import load_data
from data.validate_data import validate_data
from data.preprocess import preprocess_data
from features.build_features import build_features
from models.train_model import train_model

DATA_PATH = 'data/diabetes.csv' 
ARTIFACTS_DIR = "artifacts"

def clean_old_artifacts():
    if os.path.exists(ARTIFACTS_DIR):
        print(f"🗑️ Deleting old artifacts folder at {ARTIFACTS_DIR}...")
        shutil.rmtree(ARTIFACTS_DIR)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("✅ Artifacts directory is ready for new files.")

def main():
    # Start MLflow run to capture the pipeline execution
    with mlflow.start_run(run_name="Full_MLOps_Cycle"):
        print("-" * 50)
        print("      MLOps Pipeline Execution Started")
        print("-" * 50)
        
        clean_old_artifacts()
        
        try:
            df = load_data(DATA_PATH)
            # Log dataset size as a parameter
            mlflow.log_param("total_samples", df.shape[0])
        except Exception as e:
            print(f"🛑 Error: {e}")
            return

        validate_data(df)
        
        print("✂️ Separating target variable (Outcome) from features...")
        y = df['Outcome']
        X_raw = df.drop(columns=['Outcome']) 

        X_imputed = preprocess_data(X_raw, is_training=True)
        X_scaled, _ = build_features(X_imputed, is_training=True)

        # train_model now logs metrics to the active MLflow run
        train_model(X_scaled, y)

        print("\n" + "=" * 50)
        print("🚀 PIPELINE FINISHED SUCCESSFULLY!")
        print(f"New artifacts saved and logged to MLflow.")
        print("=" * 50)

if __name__ == "__main__":
    main()
