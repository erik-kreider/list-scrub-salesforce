import pandas as pd
import numpy as np
import re

# A list of common company suffixes to remove.
COMPANY_SUFFIXES = [
    ' co inc', ' co.', ' corp', ' co inc', ' pllc', ' llc', ' llp', ' ltd',
    ' inc.', ' inc', ' corp.', ' lp', ' lc', ' lc.', ' pa', ' dpm', ' m.d.',
    ' md', ' pc', ' co', ' asso', ' md pa', ' pa m.d.', ' pa md', ' od'
]

# A list of common words to ignore for matching keys (if needed elsewhere)
STOP_WORDS = ['the', 'a', 'an', 'and', 'of', 'for', 'in', 'on', 'at', 'with']


# Keywords used to identify and separate suite/apartment info from street addresses.
SUITE_KEYWORDS = [
    ' ste ', ' apt.', ' ste.', ' appt', ' no.', ' unit ', 'apartment', ' apt',
    ' suite', 'number', '#',
]

# --- Normalization Functions ---
def remove_stopwords(series: pd.Series) -> pd.Series:
    """Removes a predefined list of stop words from a pandas Series of strings."""
    pattern = r'\b(' + '|'.join(STOP_WORDS) + r')\b'
    return series.str.replace(pattern, '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

def normalize_company(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """An advanced normalization function for company names."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    s = df[column].astype(str).str.lower()
    s = s.str.replace(r'\s*\((fka|aka)[^)]*\)', '', regex=True)
    s = s.str.replace(r'\s+-\s+.*$', '', regex=True)
    for suffix in COMPANY_SUFFIXES:
        s = s.str.replace(r'\b' + re.escape(suffix) + r'\b', '', regex=True)
    s = s.str.replace(r'[^\w\s]', '', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    df['NormalizedCompany'] = s.str.replace(' ', '')
    return df

def normalize_website(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    A robust function to normalize website URLs. It removes prefixes, paths,
    and explicitly converts common junk/null values to an empty string.
    """
    if column not in df.columns:
        df['NormalizedWebsite'] = ''
        return df

    s = df[column].astype(str).str.lower().str.strip()
    
    # --- THE DEFINITIVE FIX ---
    # Define a pattern for all junk values, including 'nan', 'none', etc.
    junk_pattern = r'^(nan|none|null|n/a|na)$'
    # Replace junk values with an empty string FIRST.
    s = s.str.replace(junk_pattern, '', regex=True)

    # Now, normalize the actual URLs.
    s = s.str.replace(r'^(https?://)?(www\.)?', '', regex=True)
    s = s.str.split('/').str[0]
    
    df['NormalizedWebsite'] = s.str.strip()
    return df

def _split_street_suite(address_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Helper function to split an address series into street and suite components."""
    address_series = address_series.astype(str).str.lower()
    
    # Create a regex pattern from the suite keywords: (ste |apt.|#|...)
    pattern = '|'.join(re.escape(key) for key in SUITE_KEYWORDS)
    
    # Split the address by the first occurrence of any suite keyword
    split_data = address_series.str.split(f'({pattern})', n=1, expand=True)
    
    street = split_data[0].str.strip()
    
    # The suite is everything after the first keyword
    suite = split_data.iloc[:, 1:].fillna('').astype(str).agg(''.join, axis=1).str.strip()
    suite = suite.replace(r'^[\W_]+', '', regex=True) # Remove leading separators
    
    return street, suite

def normalize_street_and_suite(df: pd.DataFrame, street_col: str) -> pd.DataFrame:
    """
    Normalizes street addresses and separates suite information into its own column.
    It handles both Shipping and Billing addresses if they exist.

    Args:
        df: The DataFrame containing the address data.
        street_col: The base name of the street column (e.g., 'ShippingStreet').

    Returns:
        The DataFrame with added normalized street and suite columns.
    """
    if street_col not in df.columns:
         raise ValueError(f"Column '{street_col}' not found in DataFrame.")
    
    # --- Normalize Shipping Address ---
    norm_ship_street, norm_ship_suite = _split_street_suite(df[street_col])
    
    # Clean up punctuation and extra spaces from the street part
    df['NormShipStreet'] = norm_ship_street.str.replace(r'[\W_]+', '', regex=True)
    
    # Clean up the suite part to contain only the number/letter
    df['NormShipSuite'] = norm_ship_suite.str.replace(r'[\W_]+', '', regex=True)
    df.loc[df['NormShipSuite'] == '', 'NormShipSuite'] = np.nan # Use NaN for empty suites

    # --- Normalize Billing Address (if it exists) ---
    billing_street_col = 'BillingStreet'
    if billing_street_col in df.columns:
        norm_bill_street, norm_bill_suite = _split_street_suite(df[billing_street_col])
        df['NormBillStreet'] = norm_bill_street.str.replace(r'[\W_]+', '', regex=True)
        df['NormBillSuite'] = norm_bill_suite.str.replace(r'[\W_]+', '', regex=True)
        df.loc[df['NormBillSuite'] == '', 'NormBillSuite'] = np.nan
    else:
        # If no billing address, copy from shipping for consistent columns
        df['NormBillStreet'] = df['NormShipStreet']
        df['NormBillSuite'] = df['NormShipSuite']
        
    return df

def normalize_postal(df: pd.DataFrame, postal_col: str) -> pd.DataFrame:
    """
    Normalizes postal codes to a zero-padded 5-digit string.
    Handles both Shipping and Billing postal codes.

    Args:
        df: The DataFrame containing the postal code data.
        postal_col: The base name of the postal code column (e.g., 'ShippingPostalCode').

    Returns:
        The DataFrame with added normalized postal columns.
    """
    if postal_col not in df.columns:
        raise ValueError(f"Column '{postal_col}' not found in DataFrame.")

    def format_postal(code):
        if not isinstance(code, (str, int, float)):
            return np.nan
        # Convert to string and take the part before any hyphen
        code_str = str(code).split('-')[0].strip()
        # Keep only digits
        digits = ''.join(filter(str.isdigit, code_str))
        if len(digits) >= 5:
            return digits[:5].zfill(5)
        return np.nan

    # --- Normalize Shipping Postal ---
    df['NormShipPostal'] = df[postal_col].apply(format_postal)

    # --- Normalize Billing Postal ---
    billing_postal_col = 'BillingPostalCode'
    if billing_postal_col in df.columns:
        df['NormBillPostal'] = df[billing_postal_col].apply(format_postal)
    else:
        df['NormBillPostal'] = df['NormShipPostal']
        
    return df

def normalize_phone(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalizes phone numbers to a consistent 11-digit format (country code + number).
    Assumes US numbers and prepends '1' if the number is 10 digits long.

    Args:
        df: The DataFrame containing the phone number data.
        column: The name of the column with phone numbers.

    Returns:
        The DataFrame with an added 'NormalizedPhone' column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    def format_phone(phone_str):
        if pd.isna(phone_str):
            return np.nan
        
        # Extract all digits from the string
        digits = ''.join(filter(str.isdigit, str(phone_str)))
        
        if not digits:
            return np.nan
            
        # Prepend '1' for 10-digit numbers (common US format)
        if len(digits) == 10:
            digits = '1' + digits
        
        # Return the first 11 digits, or nan if it's not a valid length
        return digits[:11] if len(digits) >= 11 else np.nan

    df['NormalizedPhone'] = df[column].apply(format_phone)
    return df

def normalize_street_number(df: pd.DataFrame, street_col: str) -> pd.DataFrame:
    """
    Extracts the leading numerical digits from a street address to be used as a key.

    Args:
        df: The DataFrame containing the address data.
        street_col: The name of the street column (e.g., 'ShippingStreet').

    Returns:
        The DataFrame with an added 'StreetNumber' column.
    """
    if street_col not in df.columns:
        df['StreetNumber'] = ''
        return df

    # Extract the first sequence of digits found at the start of the string
    df['StreetNumber'] = df[street_col].astype(str).str.extract(r'^(\d+)', expand=False).fillna('')
    return df