"""
Unit tests for src/preprocessor.py, pure functions, no Streamlit needed.
Run with:  pytest tests/
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import prep_data, encode_data, encode_single_input
from config import TARGET_RAW, CATEGORICAL_COLS


@pytest.fixture
def raw_sample():
    return pd.DataFrame({
        "Serial_No": [1, 2, 3, 4],
        "GRE_Score": [337, 316, 300, 290],
        "TOEFL_Score": [118, 104, 100, 95],
        "University_Rating": [4, 3, 2, 1],
        "SOP": [4.5, 3.0, 2.5, 2.0],
        "LOR": [4.5, 3.5, 3.0, 2.0],
        "CGPA": [9.65, 8.0, 7.5, 7.0],
        "Research": [1, 0, 0, 0],
        TARGET_RAW: [0.92, 0.72, 0.6, 0.5],
    })


def test_prep_data_binarizes_target_at_threshold(raw_sample):
    prepped = prep_data(raw_sample, threshold=0.8)
    assert list(prepped[TARGET_RAW]) == [1, 0, 0, 0]


def test_prep_data_binarizes_with_different_threshold(raw_sample):
    prepped = prep_data(raw_sample, threshold=0.6)
    assert list(prepped[TARGET_RAW]) == [1, 1, 1, 0]


def test_prep_data_drops_serial_no(raw_sample):
    prepped = prep_data(raw_sample, threshold=0.8)
    assert "Serial_No" not in prepped.columns


def test_prep_data_casts_categoricals_to_object(raw_sample):
    prepped = prep_data(raw_sample, threshold=0.8)
    for col in CATEGORICAL_COLS:
        assert prepped[col].dtype == object


def test_encode_data_creates_dummy_columns(raw_sample):
    prepped = prep_data(raw_sample, threshold=0.8)
    encoded = encode_data(prepped)
    for col in CATEGORICAL_COLS:
        assert any(c.startswith(f"{col}_") for c in encoded.columns)
    assert not any(c in encoded.columns for c in CATEGORICAL_COLS)


def test_encode_data_keeps_target_binary(raw_sample):
    prepped = prep_data(raw_sample, threshold=0.8)
    encoded = encode_data(prepped)
    assert set(encoded[TARGET_RAW].unique()).issubset({0, 1})


def test_encode_single_input_aligns_to_feature_columns(raw_sample):
    prepped = prep_data(raw_sample, threshold=0.8)
    encoded = encode_data(prepped)
    feature_columns = [c for c in encoded.columns if c != TARGET_RAW]

    new_applicant = pd.DataFrame([{
        "GRE_Score": 320, "TOEFL_Score": 110, "University_Rating": 3,
        "SOP": 4.0, "LOR": 4.0, "CGPA": 8.8, "Research": 1,
    }])

    result = encode_single_input(new_applicant, feature_columns)
    assert list(result.columns) == feature_columns
    assert result.shape[0] == 1
    assert result.isna().sum().sum() == 0


def test_encode_single_input_handles_unseen_category(raw_sample):
    """A University_Rating value not present in training data should not
    crash, it should just produce all zero dummy columns for that field."""
    prepped = prep_data(raw_sample, threshold=0.8)
    encoded = encode_data(prepped)
    feature_columns = [c for c in encoded.columns if c != TARGET_RAW]

    new_applicant = pd.DataFrame([{
        "GRE_Score": 300, "TOEFL_Score": 100, "University_Rating": 5,  # 5 not in training sample
        "SOP": 3.0, "LOR": 3.0, "CGPA": 7.5, "Research": 0,
    }])

    result = encode_single_input(new_applicant, feature_columns)
    rating_cols = [c for c in feature_columns if c.startswith("University_Rating_")]
    assert result[rating_cols].sum(axis=1).iloc[0] == 0
