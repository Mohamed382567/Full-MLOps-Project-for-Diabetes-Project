# src/features/build_features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Global Configuration
EPSILON = 1e-6
GLUCOSE_CRITICAL_CUTOFF = 126
ARTIFACTS_DIR = "artifacts"

def classify_bmi(bmi: float) -> str:
    """Classifies BMI into standard WHO categories."""
    if bmi < 18.5: return 'Underweight'
    elif 18.5 <= bmi < 25: return 'Normal'
    elif 25 <= bmi < 30: return 'Overweight'
    elif 30 <= bmi < 35: return 'Obese_Class_I'
    elif 35 <= bmi < 40: return 'Obese_Class_II'
    else: return 'Obese_Class_III'

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies EXACT feature extraction logic: Log, Sqrt, Ratios, One-Hot.
    """
    print("⚙️ Applying Advanced Feature Engineering...")
    df_eng = df.copy()

    # --- CRITICAL FIX: PREVENT DATA LEAKAGE ---
    # Explicitly remove 'Outcome' if it exists in the dataframe.
    # This prevents the target variable from being treated as a feature.
    if 'Outcome' in df_eng.columns:
        print("⚠️ Info: Dropping 'Outcome' column from features to prevent data leakage.")
        df_eng = df_eng.drop(columns=['Outcome'])
    # ------------------------------------------

    # --- 1. Log and Sqrt Transforms ---
    # Adding Epsilon to avoid log(0) errors just in case
    df_eng['Log_DPF'] = np.log(df_eng['DiabetesPedigreeFunction'] + EPSILON)
    df_eng['Log_Age'] = np.log1p(df_eng['Age']) 
    df_eng['Sqrt_Insulin'] = np.sqrt(df_eng['Insulin'].clip(lower=0)) # Clip negative values safety
    df_eng['Sqrt_Pregnancies'] = np.sqrt(df_eng['Pregnancies'].clip(lower=0))

    # --- 2. Ratios & Interactions ---
    df_eng['Glucose_to_Insulin_Ratio'] = df_eng['Glucose'] / (df_eng['Insulin'] + EPSILON)
    df_eng['Age_BMI_Interaction'] = df_eng['Age'] * df_eng['BMI']
    df_eng['BP_Age_Index'] = df_eng['BloodPressure'] / (df_eng['Age'] + EPSILON)
    df_eng['Skin_BMI_Ratio'] = df_eng['SkinThickness'] / (df_eng['BMI'] + EPSILON)

    # --- 3. Critical Flags & Categorization ---
    df_eng['Is_Glucose_Critical'] = (df_eng['Glucose'] >= GLUCOSE_CRITICAL_CUTOFF).astype(int)
    df_eng['BMI_Category'] = df_eng['BMI'].apply(classify_bmi)

    # --- 4. One-Hot Encoding ---
    df_eng = pd.get_dummies(df_eng, columns=['BMI_Category'], drop_first=True)

    # --- 5. Drop Columns ---
    if 'DiabetesPedigreeFunction' in df_eng.columns:
        df_eng = df_eng.drop('DiabetesPedigreeFunction', axis=1)
    
    # Clean up any infinites created by division
    df_eng.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df_eng

def build_features(df: pd.DataFrame, is_training=True):
    """
    Orchestrates Feature Engineering and Scaling.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    columns_path = os.path.join(ARTIFACTS_DIR, "columns.pkl")
    
    # Separate Target if it exists (for Training)
    if 'Outcome' in df.columns:
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
    else:
        X = df
        y = None

    # Apply Feature Engineering
    # Note: Even if 'Outcome' slipped into X above, feature_engineering() now handles it safely.
    X_engineered = feature_engineering(X)

    if is_training:
        print("⚖️ Scaling features (Training Mode)...")
        
        # Save the exact column list structure (Critical for One-Hot encoding alignment)
        joblib.dump(list(X_engineered.columns), columns_path)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_engineered)
        joblib.dump(scaler, scaler_path)
        
        return X_scaled, y
    else:
        # Inference Mode
        scaler = joblib.load(scaler_path)
        saved_cols = joblib.load(columns_path)
        
        # Align columns: Add missing columns (with 0) and remove extra ones
        X_engineered = X_engineered.reindex(columns=saved_cols, fill_value=0)
        
        X_scaled = scaler.transform(X_engineered)
        
        return X_scaled, y