import os
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus


def build_conn_str(db_config: dict) -> str:
    """
    Build a safe SQLAlchemy connection string for mssql+pyodbc.
    Uses URL encoding to handle special characters in credentials.
    """
    required = ["server", "database", "username", "password", "driver"]
    missing = [k for k in required if not db_config.get(k)]
    if missing:
        raise ValueError(f"Missing required database config keys: {', '.join(missing)}")

    odbc_str = (
        f"DRIVER={{{db_config['driver']}}};"
        f"SERVER={db_config['server']};"
        f"DATABASE={db_config['database']};"
        f"UID={db_config['username']};"
        f"PWD={db_config['password']};"
        "TrustServerCertificate=yes;"   # prevents SSL errors in corporate networks
    )

    encoded = quote_plus(odbc_str)
    return f"mssql+pyodbc:///?odbc_connect={encoded}"


def load_from_sql(query: str, db_config: dict) -> pd.DataFrame:
    """Connect to SQL Server and return query results as a DataFrame."""
    conn_str = build_conn_str(db_config)
    print("Creating SQLAlchemy engine...")
    engine = create_engine(conn_str)

    try:
        with engine.connect() as conn:
            print("Executing SQL query...")
            df = pd.read_sql(text(query), conn)
        print(f"✅ Successfully pulled {len(df):,} records from the database.")
        return df
    except Exception as ex:
        print("❌ Database connection or query failed.")
        raise RuntimeError(f"SQL Error: {ex}") from ex


def load_scrub_file(filepath: str) -> pd.DataFrame:
    # Step 1: Expand the '~' user home directory shortcut.
    expanded_path = os.path.expanduser(filepath)
    
    normalized_path = os.path.normpath(expanded_path)
    
    # Step 2: Check if the clean, normalized path exists.
    if not os.path.exists(normalized_path):
        raise FileNotFoundError(f"Input file not found at: {normalized_path}")
        
    print(f"Loading scrub file: {normalized_path}")
    
    # Step 3: Read the Excel file using the clean path.
    df = pd.read_excel(normalized_path)
    
    print(f"-> Successfully loaded {len(df):,} records from the Excel file.")
    return df


def save_to_excel(df: pd.DataFrame, filepath: str):
    """Saves a DataFrame to Excel, creating directories if needed."""
    expanded_path = os.path.expanduser(filepath)
    os.makedirs(os.path.dirname(expanded_path), exist_ok=True)

    df.to_excel(expanded_path, index=False)
    print(f"✅ Output saved to: {expanded_path}")
