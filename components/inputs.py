"""
Reusable Streamlit input widgets. UI only, delegates all logic to src/.
"""

import pandas as pd
import streamlit as st

from src.data_loader import load_csv


def sidebar_data_source(filename: str, key_prefix: str = "data"):
    """Radio plus uploader for choosing a CSV source. Returns a DataFrame or None."""
    src = st.sidebar.radio("Data source", [f"Use {filename} in folder", "Upload CSV"], key=f"{key_prefix}_src")
    if src == f"Use {filename} in folder":
        try:
            return load_csv(filename)
        except FileNotFoundError:
            st.sidebar.warning(f"`{filename}` not found next to this script. Upload it instead.")
            return None
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv", key=f"{key_prefix}_upload")
    return load_csv(uploaded) if uploaded is not None else None


def sidebar_split_settings(default_test_size: float, default_random_state: int):
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, default_test_size, 0.05)
    stratify_target = st.sidebar.checkbox("Stratify by target", value=True)
    random_state = st.sidebar.number_input("Random state", value=default_random_state, step=1)
    return test_size, stratify_target, int(random_state)


def sidebar_mlp_settings(default_hidden_layers: tuple, default_batch_size: int, default_max_iter: int,
                          default_learning_rate: float, default_alpha: float):
    hidden_layers_text = st.sidebar.text_input(
        "Hidden layer sizes (comma-separated)", value=",".join(str(n) for n in default_hidden_layers)
    )
    activation = st.sidebar.selectbox("Activation function", ["relu", "tanh", "logistic", "identity"], index=0)
    solver = st.sidebar.selectbox("Solver", ["adam", "sgd", "lbfgs"], index=0)
    batch_size = st.sidebar.slider("Batch size", 8, 200, default_batch_size, 2)
    max_iter = st.sidebar.slider("Max iterations", 50, 1000, default_max_iter, 50)
    learning_rate_init = st.sidebar.select_slider(
        "Learning rate (init)", options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], value=default_learning_rate
    )
    alpha = st.sidebar.select_slider("Alpha (L2 penalty)", options=[0.00001, 0.0001, 0.001, 0.01, 0.1], value=default_alpha)
    return hidden_layers_text, activation, solver, batch_size, max_iter, learning_rate_init, alpha


def applicant_form():
    """Renders the raw field applicant input form. Returns a one row DataFrame."""
    c1, c2, c3 = st.columns(3)
    with c1:
        gre = st.number_input("GRE Score (out of 340)", min_value=260, max_value=340, value=316)
        toefl = st.number_input("TOEFL Score (out of 120)", min_value=80, max_value=120, value=107)
        uni_rating = st.selectbox("University Rating (bachelor's, out of 5)", [1, 2, 3, 4, 5], index=2)
    with c2:
        sop = st.slider("SOP Strength (out of 5)", 1.0, 5.0, 3.5, 0.5)
        lor = st.slider("LOR Strength (out of 5)", 1.0, 5.0, 3.5, 0.5)
        cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=8.6, step=0.01)
    with c3:
        research = st.selectbox("Research Experience", [0, 1], format_func=lambda v: "Yes" if v == 1 else "No")

    return pd.DataFrame([{
        "GRE_Score": gre, "TOEFL_Score": toefl, "University_Rating": uni_rating,
        "SOP": sop, "LOR": lor, "CGPA": cgpa, "Research": research,
    }])


def generic_encoded_form(feature_columns: list, reference_df: pd.DataFrame, key_prefix: str = "load"):
    """Fallback form built from encoded feature columns directly."""
    input_vals = {}
    cols = st.columns(3)
    for i, col in enumerate(feature_columns):
        with cols[i % 3]:
            default_val = float(reference_df[col].median()) if col in reference_df.columns else 0.0
            input_vals[col] = st.number_input(col, value=default_val, key=f"{key_prefix}_{col}")
    return pd.DataFrame([input_vals])[feature_columns]
