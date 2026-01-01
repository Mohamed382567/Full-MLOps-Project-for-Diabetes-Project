# src/data/load_data.py
# English Comments

import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file (e.g., 'data/diabetes.csv').
        
    Returns:
        pd.DataFrame: Loaded dataframe.
        
    Raises:
        FileNotFoundError: If the data file is not found at the specified path.
    """
    print(f"--- 🔄 Loading data from: {file_path} ---")
    
    # Check if the file exists before attempting to read
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Data file not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        raise
