import pandas as pd
import numpy as np
import joblib
import os
# Force enabling iterative imputer (still experimental in sklearn 1.3)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Consts based on notebook findings
MISSING_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
ARTIFACTS_DIR = "artifacts"
MIN_PHYSIOLOGICAL_INSULIN = 1.0

def create_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """Track which values were originally missing before imputation"""
    X_copy = X.copy()
    for col in MISSING_COLS:
        if col in X_copy.columns:
            X_copy[f'Is_{col}_Missing'] = (X_copy[col] == 0).astype(int)
    return X_copy

def preprocess_data(X: pd.DataFrame, is_training: bool = True):
    """
    Refactored to match Notebook: MICE Imputation + Insulin Clamping.
    Using IterativeImputer instead of KNN for better cross-feature estimation.
    """
    # Defensive copy to avoid SettingWithCopy warnings
    X = X.copy()
    
    # 1. Target leak prevention
    if 'Outcome' in X.columns:
        X = X.drop(columns=['Outcome'])

    # 2. Setup missingness
    X = create_missing_indicators(X)
    for col in MISSING_COLS:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
    
    imputer_path = os.path.join(ARTIFACTS_DIR, 'mice_imputer.pkl')

    # 3. MICE Logic
    if is_training:
        # Match notebook params: 10 iters, fixed seed
        imputer = IterativeImputer(max_iter=10, random_state=42)
        X_imputed_array = imputer.fit_transform(X)
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        joblib.dump(imputer, imputer_path)
    else:
        if not os.path.exists(imputer_path):
            raise RuntimeError("Missing imputer artifact. Run training pipeline first.")
        imputer = joblib.load(imputer_path)
        X_imputed_array = imputer.transform(X)

    # 4. Post-imputation cleanup
    X_imputed_df = pd.DataFrame(X_imputed_array, columns=X.columns)

    # 5. Domain Logic: Insulin can't be 0 or negative in living patients
    if 'Insulin' in X_imputed_df.columns:
        X_imputed_df['Insulin'] = X_imputed_df['Insulin'].clip(lower=MIN_PHYSIOLOGICAL_INSULIN)

    return X_imputed_df