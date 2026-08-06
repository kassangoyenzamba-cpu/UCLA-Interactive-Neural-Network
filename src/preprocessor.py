"""
Pure pandas preprocessing logic: target binarization, dtype casting and
one hot encoding. No Streamlit imports here, so this module can be unit
tested with pytest in isolation and reused outside the app (for example
in a batch scoring script).
"""

import pandas as pd

from config import TARGET_RAW, CATEGORICAL_COLS


def prep_data(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Mirrors the notebook: binarize the target, drop Serial_No, and cast
    University_Rating and Research to object dtype so they are treated as
    categorical rather than numeric during encoding."""
    df = df.copy()
    df[TARGET_RAW] = (df[TARGET_RAW] >= threshold).astype(int)
    if "Serial_No" in df.columns:
        df = df.drop("Serial_No", axis=1)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("object")
    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """One hot encode the categorical columns."""
    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    return pd.get_dummies(df, columns=cat_present, dtype=int)


def encode_single_input(raw_input: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """One hot encode a single row raw input and align it to the trained
    feature column order, filling any missing dummy columns with 0."""
    raw_input = raw_input.copy()
    for col in CATEGORICAL_COLS:
        if col in raw_input.columns:
            raw_input[col] = raw_input[col].astype("object")
    encoded = pd.get_dummies(raw_input, columns=CATEGORICAL_COLS, dtype=int)
    return encoded.reindex(columns=feature_columns, fill_value=0)
