"""
Model training logic: split, scale, fit the MLPClassifier. Pure
scikit-learn, no Streamlit.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

from config import TARGET_RAW


def split_data(encoded_df: pd.DataFrame, test_size: float, random_state: int, stratify_target: bool = True):
    x = encoded_df.drop(columns=[TARGET_RAW])
    y = encoded_df[TARGET_RAW]
    strat = y if stratify_target else None
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=strat)


def scale_features(x_train: pd.DataFrame, x_test: pd.DataFrame):
    """Fit MinMaxScaler on train only, avoiding data leakage into test."""
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def parse_hidden_layers(text: str) -> tuple:
    """Parse a comma separated string like '3,4' into a tuple (3, 4).
    Falls back to a single layer of 3 neurons if the text is empty or
    cannot be parsed."""
    try:
        parts = [int(p.strip()) for p in text.split(",") if p.strip()]
        return tuple(parts) if parts else (3,)
    except ValueError:
        return (3,)


def train_mlp(x_train_scaled, y_train, hidden_layer_sizes=(3, 4), activation="relu",
              solver="adam", batch_size=50, max_iter=200, learning_rate_init=0.001,
              alpha=0.0001, random_state=123) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        batch_size=batch_size,
        max_iter=max_iter,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        random_state=random_state,
    )
    model.fit(x_train_scaled, y_train)
    return model
