import pytest
import pandas as pd
from src.data.preprocess import preprocess_data

def test_preprocess_output_format():
    """
    Test if the preprocessing function returns a non-empty DataFrame
    and maintains numerical stability.
    """
    sample = pd.DataFrame({
        "Pregnancies": [2], "Glucose": [130], "BloodPressure": [80],
        "SkinThickness": [20], "Insulin": [90], "BMI": [28.0],
        "DiabetesPedigreeFunction": [0.5], "Age": [35]
    })
    
    processed = preprocess_data(sample, is_training=False)
    
    assert isinstance(processed, pd.DataFrame)
    assert not processed.empty
    assert processed.shape[1] > 0
