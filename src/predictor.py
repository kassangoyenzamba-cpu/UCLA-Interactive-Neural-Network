"""
Inference and model persistence. Bundles the model with its scaler,
feature order, and the admission threshold used, so a saved .pkl is
self contained and safe to reload.
"""

import io
import pickle

import pandas as pd
import streamlit as st

from src.preprocessor import encode_single_input


def make_bundle(model, scaler, feature_columns: list, threshold: float) -> dict:
    return {"model": model, "scaler": scaler, "feature_columns": feature_columns, "threshold": threshold}


def serialize_bundle(bundle: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    pickle.dump(bundle, buffer)
    buffer.seek(0)
    return buffer


@st.cache_resource(show_spinner=False)
def load_bundle_from_bytes(file_bytes: bytes) -> dict:
    """Deserialize an uploaded .pkl exactly once per unique file content."""
    return pickle.loads(file_bytes)


def predict_from_raw(bundle: dict, raw_input: pd.DataFrame):
    """Encode and scale a raw single row input, then predict the label and
    the probability of admission."""
    encoded = encode_single_input(raw_input, bundle["feature_columns"])
    scaled = bundle["scaler"].transform(encoded)
    model = bundle["model"]
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1]
    return pred, proba


def predict_from_encoded(bundle: dict, input_df: pd.DataFrame):
    """Predict from an already encoded single row DataFrame, used when the
    dataset came in pre-processed and there is no raw field form."""
    feature_columns = bundle["feature_columns"]
    input_df = input_df[feature_columns]
    scaled = bundle["scaler"].transform(input_df) if bundle["scaler"] is not None else input_df
    model = bundle["model"]
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1] if hasattr(model, "predict_proba") else None
    return pred, proba
