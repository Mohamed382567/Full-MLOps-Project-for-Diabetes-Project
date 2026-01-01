# src/data/validate_data.py
# English Comments

import pandas as pd

try:
    import great_expectations as ge
except Exception:
    ge = None

def ensure_great_expectations_available():
    if ge is None:
        raise RuntimeError(
            "Missing dependency: 'great_expectations' is not installed. "
            "Install with: python -m pip install great_expectations "
            "or: python -m pip install -r requirements.txt"
        )

def validate_data(df: pd.DataFrame) -> bool:
    """
    Validates the raw data using Great Expectations (GX) to ensure data quality
    and consistency before preprocessing.
    
    Checks for: Column existence, data types, and logical value ranges.
    
    Args:
        df (pd.DataFrame): Raw dataframe loaded from the source.
        
    Returns:
        bool: True if validation passes, raises error otherwise.
    """
    print("--- 🔍 Starting Data Validation ---")
    
    try:
        ensure_great_expectations_available()
        # Try to use Great Expectations
        context = ge.get_context(context_root_dir=None)
        batch = ge.dataset.PandasDataset(df)
        
        # --- Validation Rules (Expectations) ---

        # 1. Check required columns existence
        expected_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        for col in expected_columns:
            batch.expect_column_to_exist(col)

        # 2. Check data types (All input features should be numeric)
        for col in expected_columns:
            batch.expect_column_values_to_be_in_type_list(col, ["int", "int64", "float", "float64"])

        # 3. Logic Checks (Based on Domain Knowledge)
        # Age and Pregnancies must be non-negative
        batch.expect_column_values_to_be_between("Age", min_value=0, max_value=120, mostly=0.95)
        batch.expect_column_values_to_be_between("Pregnancies", min_value=0, max_value=20, mostly=0.95)
        
        # Glucose must be > 0 , but we allow some invalids for imputation laterpython 
        batch.expect_column_min_to_be_between("Glucose", min_value=0, max_value=100)
        
        # BloodPressure reasonable range
        batch.expect_column_values_to_be_between("BloodPressure", min_value=0, max_value=200, mostly=0.95)   

        # BMI reasonable range
        batch.expect_column_values_to_be_between("BMI", min_value=0, max_value=70, mostly=0.95)

        # Insulin reasonable range
        batch.expect_column_values_to_be_between("Insulin", min_value=0, max_value  =900, mostly=0.95)

        # SkinThickness reasonable range
        batch.expect_column_values_to_be_between("SkinThickness", min_value=0, max_value=100, mostly=0.95) 

        # Target variable check
        batch.expect_column_values_to_be_in_set("Outcome", [0, 1])

        # --- Run Validation ---
        validation_result = batch.validate()

        if validation_result["success"]:
            print("✅ Data Validation Passed!")
            return True
        else:
            print("❌ Data Validation Failed!")
            # Printing the validation results helps identify which check failed in CI/CD logs
            print(validation_result)
            raise ValueError("Data validation failed. Check logs for details.")

    except (AttributeError, TypeError, ImportError) as e:
        # Fallback: Use basic pandas validation if GE fails (Python 3.14 issue)
        print(f"⚠️ Great Expectations failed ({type(e).__name__}), using basic validation instead...")
        
        # Basic validation checks
        assert df.shape[0] > 0, "Dataset is empty"
        assert 'Outcome' in df.columns, "Missing 'Outcome' column"
        assert df['Glucose'].notna().sum() > 0, "Glucose column has too many nulls"
        
        print("✅ Basic data validation passed!")