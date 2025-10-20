import pandas as pd
import numpy as np
import re

def normalize_company(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """
    Advanced company name normalization.
    Removes suffixes AND trailing location info (e.g., ' - San Jose, CA').
    """
    if source_col not in df.columns:
        df['normalizedcompany'] = ''
        return df

    s = df[source_col].astype(str).str.lower()
    # Remove trailing location data like ' - city, state'
    s = s.str.replace(r'\s+-\s+.*$', '', regex=True)
    
    s = s.str.replace(r'[^\w\s]', '', regex=True) # Remove punctuation
    
    suffixes = ['llc', 'inc', 'corp', 'ltd', 'lp', 'co']
    pattern = r'\b(' + '|'.join(suffixes) + r')\b'
    s = s.str.replace(pattern, '', regex=True)
    
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    df['normalizedcompany'] = s.str.replace(' ', '')
    return df

def normalize_street(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """Safely normalizes street addresses. Creates 'normalizedstreet' column."""
    if source_col not in df.columns:
        df['normalizedstreet'] = ''
        return df

    s = df[source_col].astype(str).str.lower()
    s = s.str.split(r'\s(?:#|apt|suite|ste)\s?\w*', n=1, expand=True)[0]
    s = s.str.replace(r'[^\w\s]', '', regex=True)
    s = s.str.replace(r'\s+', '', regex=True).str.strip()
    df['normalizedstreet'] = s
    return df

def normalize_postal(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """Safely normalizes postal codes to 5 digits. Creates 'normalizedpostal'."""
    if source_col not in df.columns:
        df['normalizedpostal'] = ''
        return df

    def format_postal(code):
        if pd.isna(code): return ''
        digits = ''.join(filter(str.isdigit, str(code)))
        return digits[:5].zfill(5) if len(digits) >= 5 else ''

    df['normalizedpostal'] = df[source_col].apply(format_postal)
    return df

def normalize_website(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """Safely normalizes website URLs. Creates 'normalizedwebsite' column."""
    if source_col not in df.columns:
        df['normalizedwebsite'] = ''
        return df

    s = df[source_col].astype(str).str.lower().str.strip()
    junk_pattern = r'^(nan|none|null|n/a|na|-)$'
    s = s.str.replace(junk_pattern, '', regex=True)
    s = s.str.replace(r'^(https?://)?(www\.)?', '', regex=True)
    s = s.str.split('/').str[0]
    s = s.str.split('?').str[0]
    df['normalizedwebsite'] = s.str.strip()
    return df

def normalize_phone(df: pd.DataFrame, source_col: str) -> pd.DataFrame:
    """
    Normalizes phone numbers to a consistent digit-only format.
    Handles formats like (555) 555-5555 and 5555555555.
    """
    if source_col not in df.columns:
        df['normalizedphone'] = ''
        return df

    # Extract all digits from the string
    df['normalizedphone'] = df[source_col].astype(str).str.extractall('(\d)').unstack().fillna('').agg(''.join, axis=1)
    return df

def normalize_text_field(df: pd.DataFrame, source_col: str, dest_col: str) -> pd.DataFrame:
    """
    A generic function for simple text fields like city, state, country, LOB.
    """
    if source_col not in df.columns:
        df[dest_col] = ''
        return df
    
    df[dest_col] = df[source_col].astype(str).str.lower().str.strip()
    return df