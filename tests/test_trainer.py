"""
Unit tests for src/trainer.py and src/evaluator.py.
Run with:  pytest tests/
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.trainer import split_data, scale_features, parse_hidden_layers, train_mlp
from src.evaluator import evaluate, meets_accuracy_target
from config import TARGET_RAW


@pytest.fixture
def encoded_sample():
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame({
        "GRE_Score": rng.integers(290, 340, n),
        "TOEFL_Score": rng.integers(90, 120, n),
        "CGPA": rng.uniform(6.5, 9.9, n),
        "Research_1": rng.integers(0, 2, n),
        TARGET_RAW: rng.integers(0, 2, n),
    })


def test_split_data_respects_test_size(encoded_sample):
    x_train, x_test, y_train, y_test = split_data(encoded_sample, test_size=0.2, random_state=1, stratify_target=False)
    assert len(x_test) == 24
    assert len(x_train) == 96


def test_split_data_excludes_target_from_features(encoded_sample):
    x_train, x_test, y_train, y_test = split_data(encoded_sample, test_size=0.2, random_state=1, stratify_target=False)
    assert TARGET_RAW not in x_train.columns
    assert TARGET_RAW not in x_test.columns


def test_scale_features_fits_only_on_train(encoded_sample):
    x_train, x_test, y_train, y_test = split_data(encoded_sample, test_size=0.2, random_state=1, stratify_target=False)
    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)
    assert x_train_scaled.min() >= 0
    assert x_train_scaled.max() <= 1
    assert scaler is not None


def test_scale_features_test_can_exceed_train_range(encoded_sample):
    """Since the scaler is fit on train only, values in test can legitimately
    fall slightly outside [0, 1] if test contains more extreme values."""
    x_train, x_test, y_train, y_test = split_data(encoded_sample, test_size=0.2, random_state=1, stratify_target=False)
    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)
    # this should not raise, and scaler.data_min_/data_max_ should reflect train only
    assert scaler.data_min_.shape[0] == x_train.shape[1]


def test_parse_hidden_layers_parses_comma_separated_string():
    assert parse_hidden_layers("3,4") == (3, 4)
    assert parse_hidden_layers("10") == (10,)
    assert parse_hidden_layers(" 5 , 6 , 7 ") == (5, 6, 7)


def test_parse_hidden_layers_falls_back_on_empty_string():
    assert parse_hidden_layers("") == (3,)


def test_parse_hidden_layers_falls_back_on_invalid_text():
    assert parse_hidden_layers("abc,def") == (3,)


def test_train_mlp_returns_fitted_model_with_matching_architecture(encoded_sample):
    x_train, x_test, y_train, y_test = split_data(encoded_sample, test_size=0.2, random_state=1, stratify_target=False)
    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)
    model = train_mlp(x_train_scaled, y_train, hidden_layer_sizes=(4, 2), max_iter=50, random_state=1)
    assert model.hidden_layer_sizes == (4, 2)
    preds = model.predict(x_test_scaled)
    assert len(preds) == len(x_test)
    assert set(preds).issubset({0, 1})


def test_evaluate_returns_expected_keys(encoded_sample):
    x_train, x_test, y_train, y_test = split_data(encoded_sample, test_size=0.2, random_state=1, stratify_target=False)
    x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)
    model = train_mlp(x_train_scaled, y_train, max_iter=50, random_state=1)
    preds = model.predict(x_test_scaled)
    result = evaluate(y_test, preds)
    assert set(result.keys()) == {"accuracy", "confusion_matrix", "report"}
    assert 0 <= result["accuracy"] <= 1


def test_meets_accuracy_target():
    assert meets_accuracy_target(0.95) is True
    assert meets_accuracy_target(0.90) is True
    assert meets_accuracy_target(0.89) is False
    assert meets_accuracy_target(0.75, target=0.70) is True
