import pandas as pd
import os

def load_scrub_file(filepath: str) -> pd.DataFrame:
    """
    Loads a local Excel file (the list to be scrubbed) and standardizes
    all column headers to be lowercase and stripped of whitespace.
    """
    expanded_path = os.path.expanduser(filepath)
    if not os.path.exists(expanded_path):
        raise FileNotFoundError(f"Input file not found at: {expanded_path}")
    
    print(f"Loading scrub file: {expanded_path}")
    df = pd.read_excel(expanded_path)
    
    # Standardize column headers
    df.columns = df.columns.str.lower().str.strip()
    
    print(f"-> Successfully loaded {len(df):,} records from scrub file.")
    return df

def save_to_excel(df: pd.DataFrame, filepath: str):
    """Saves a DataFrame to an Excel file, creating directories if needed."""
    expanded_path = os.path.expanduser(filepath)
    os.makedirs(os.path.dirname(expanded_path), exist_ok=True)
    
    df.to_excel(expanded_path, index=False)
    print(f"✅ Output saved to: {expanded_path}")