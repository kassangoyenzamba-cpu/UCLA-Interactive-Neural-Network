"""
UCLA Interactive Neural Network Dashboard (modular)
============================================================================
Entry point: layout, navigation, sidebar. All data and ML logic lives in
src/, all reusable UI pieces live in components/.

Run with:
    streamlit run app.py

Place `Admission.csv` in this folder, or upload it from the sidebar.
"""

import pandas as pd
import streamlit as st

import config
from src.preprocessor import prep_data, encode_data
from src.trainer import split_data, scale_features, parse_hidden_layers, train_mlp
from src.evaluator import evaluate, meets_accuracy_target
from src.predictor import make_bundle, serialize_bundle, load_bundle_from_bytes, predict_from_raw, predict_from_encoded
from components.inputs import sidebar_data_source, sidebar_split_settings, sidebar_mlp_settings, applicant_form, generic_encoded_form
from components.charts import (
    confusion_matrix_chart, correlation_heatmap, scatter_by_outcome, distribution_chart,
    boxplot_by_outcome, before_after_scaling, loss_curve_chart,
)

# --------------------------------------------------------------------------
st.set_page_config(page_title="UCLA Interactive Neural Network: Prediction Dashboard", page_icon="🎓", layout="wide")

for key, default in {
    "raw_df": None, "prepped_df": None, "encoded_df": None,
    "x_train": None, "x_test": None, "y_train": None, "y_test": None,
    "x_train_scaled": None, "x_test_scaled": None, "scaler": None,
    "mlp_model": None, "train_acc": None, "test_acc": None, "cm": None, "report": None,
    "trained": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------------------------------------------------------
# Sidebar, data loading
# --------------------------------------------------------------------------
st.sidebar.title("UCLA: Interactive Neural Network")
st.sidebar.markdown("### 1. Load data")

df = sidebar_data_source(config.CSV_PATH, "ucla")
if df is not None:
    st.session_state.raw_df = df

if st.session_state.raw_df is None:
    st.title(" UCLA: Interactive Neural Network Dashboard")
    st.info("Load `Admission.csv` (or upload your own CSV) from the sidebar to get started.")
    st.stop()

raw_df = st.session_state.raw_df
TARGET = config.TARGET_RAW

if TARGET not in raw_df.columns:
    st.error(f"Dataset must contain an '{TARGET}' column (0 to 1 admission probability).")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar, target threshold, split, architecture
# --------------------------------------------------------------------------
st.sidebar.markdown("### 2. Target binarization")
threshold = st.sidebar.slider("Admit_Chance threshold (>= is Admitted)", 0.5, 0.95, config.DEFAULT_THRESHOLD, 0.05)

st.session_state.prepped_df = prep_data(raw_df, threshold)
st.session_state.encoded_df = encode_data(st.session_state.prepped_df)
encoded_df = st.session_state.encoded_df

st.sidebar.markdown("### 3. Train/test split")
test_size, stratify_target, random_state = sidebar_split_settings(config.DEFAULT_TEST_SIZE, config.DEFAULT_RANDOM_STATE)

st.sidebar.markdown("### 4. Neural network architecture")
hidden_layers_text, activation, solver, batch_size, max_iter, learning_rate_init, alpha = sidebar_mlp_settings(
    config.DEFAULT_HIDDEN_LAYERS, config.DEFAULT_BATCH_SIZE, config.DEFAULT_MAX_ITER,
    config.DEFAULT_LEARNING_RATE_INIT, config.DEFAULT_ALPHA,
)

train_clicked = st.sidebar.button("🚀 Train / Retrain neural network", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
st.title(" UCLA Interactive Neural Network: Prediction")
st.caption(
    "Reproduces `UCLA_Neural_Networks_Solution.ipynb`: target binarization, EDA, encoding, "
    "MinMax scaling fit on train only, MLPClassifier neural network, evaluation, prediction "
    "(goal: accuracy at least 90%)."
)

tab_overview, tab_eda, tab_prep, tab_train, tab_predict, tab_model = st.tabs(
    [" Data Overview", " EDA", " Preprocessing", "🧠 Train & Evaluate Neural Network",
     " Predict Admission", "💾 Save / Load Model"]
)

# ==========================================================================
with tab_overview:
    st.subheader("Dataset preview (raw)")
    st.dataframe(raw_df.head(), use_container_width=True)
    st.write(f"**Shape:** {raw_df.shape[0]} rows by {raw_df.shape[1]} columns")

    st.subheader("Column info")
    info_df = pd.DataFrame({
        "dtype": raw_df.dtypes.astype(str),
        "missing": raw_df.isna().sum(),
        "unique": raw_df.nunique(),
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Summary statistics")
    st.dataframe(raw_df.describe().T, use_container_width=True)

    st.markdown(
        f"""
**Quick observations** (mirroring the notebook):
- {raw_df.shape[0]} observations, {raw_df.shape[1]} columns, all numeric, no missing values.
- Average GRE score: **{raw_df['GRE_Score'].mean():.0f}** out of 340. Average TOEFL score: **{raw_df['TOEFL_Score'].mean():.0f}** out of 120.
- Average CGPA: **{raw_df['CGPA'].mean():.2f}** out of 10.
- **{(raw_df['Research'].mean()*100):.0f}%** of students have research experience.
- At a threshold of **{threshold:.2f}**, **{(st.session_state.prepped_df[TARGET].mean()*100):.0f}%** of students are classified as Admitted.
        """
    )

# ==========================================================================
with tab_eda:
    st.subheader("Correlation matrix")
    correlation_heatmap(raw_df.drop(columns=["Serial_No"], errors="ignore"))

    st.subheader("GRE Score vs. TOEFL Score, colored by admission outcome")
    plot_df = st.session_state.prepped_df.copy()
    plot_df["Admit_Chance_Label"] = plot_df[TARGET].map({1: "Admitted", 0: "Not Admitted"})
    scatter_by_outcome(plot_df, "GRE_Score", "TOEFL_Score", "Admit_Chance_Label")
    st.caption("As in the notebook: GRE and TOEFL scores are strongly correlated, and admitted "
                "students tend to cluster at higher GRE (around 320 or more) and TOEFL (around 105 or more) scores.")

    st.subheader("Feature distribution explorer")
    num_cols = ["GRE_Score", "TOEFL_Score", "SOP", "LOR", "CGPA"]
    num_col = st.selectbox("Choose a feature", num_cols)
    distribution_chart(raw_df[num_col])

    st.subheader(f"{num_col} vs. admission outcome (boxplot)")
    boxplot_by_outcome(plot_df, num_col, "Admit_Chance_Label")

# ==========================================================================
with tab_prep:
    st.subheader("Step 1: Binarize the target")
    st.markdown(
        f"`Admit_Chance` is converted to 1 (Admitted) if the original value is at least "
        f"**{threshold:.2f}**, else 0. Adjust the threshold in the sidebar."
    )
    st.dataframe(st.session_state.prepped_df[[TARGET]].value_counts().rename("count").to_frame(),
                 use_container_width=True)

    st.subheader("Step 2: Treat categorical variables")
    st.markdown(
        "`University_Rating` and `Research` are numerically encoded but represent **categories**, "
        "not continuous quantities, so they are cast to `object` dtype and one hot encoded, "
        "just like the notebook."
    )
    st.dataframe(st.session_state.prepped_df.head(), use_container_width=True)

    st.subheader("Step 3: One hot encoded (model ready) data")
    st.dataframe(encoded_df.head(), use_container_width=True)
    st.caption(f"Encoded shape: {encoded_df.shape[0]} rows by {encoded_df.shape[1]} columns")

    st.subheader("Step 4: Scaling preview (fit on train only, to avoid data leakage)")
    x_preview = encoded_df.drop(columns=[TARGET])
    y_preview = encoded_df[TARGET]
    xtr_p, xte_p, ytr_p, yte_p = split_data(encoded_df, test_size, random_state, stratify_target)
    xtr_scaled_p, xte_scaled_p, preview_scaler = scale_features(xtr_p, xte_p)
    xtr_scaled_p_df = pd.DataFrame(xtr_scaled_p, columns=xtr_p.columns)

    before_after_scaling(xtr_p["GRE_Score"], xtr_scaled_p_df["GRE_Score"], "GRE_Score")

    csv_buf = encoded_df.to_csv(index=False).encode()
    st.download_button("Download encoded dataset", data=csv_buf, file_name="Processed_Admission_Dataset.csv", mime="text/csv")

# ==========================================================================
with tab_train:
    st.subheader("Train the MLPClassifier neural network")
    st.markdown(
        "Split, then **MinMaxScaler** fit on train only, then **MLPClassifier**, then accuracy, "
        "confusion matrix, loss curve (goal: accuracy at least **90%**)."
    )

    hidden_layers = parse_hidden_layers(hidden_layers_text)
    st.caption(f"Hidden layer sizes parsed as: {hidden_layers}")

    if train_clicked:
        x_train, x_test, y_train, y_test = split_data(encoded_df, test_size, random_state, stratify_target)
        x_train_scaled, x_test_scaled, scaler = scale_features(x_train, x_test)

        mlp = train_mlp(
            x_train_scaled, y_train, hidden_layer_sizes=hidden_layers, activation=activation,
            solver=solver, batch_size=batch_size, max_iter=max_iter,
            learning_rate_init=learning_rate_init, alpha=alpha, random_state=random_state,
        )

        ypred_train = mlp.predict(x_train_scaled)
        ypred_test = mlp.predict(x_test_scaled)

        st.session_state.x_train, st.session_state.x_test = x_train, x_test
        st.session_state.y_train, st.session_state.y_test = y_train, y_test
        st.session_state.x_train_scaled, st.session_state.x_test_scaled = x_train_scaled, x_test_scaled
        st.session_state.scaler = scaler
        st.session_state.mlp_model = mlp

        train_eval = evaluate(y_train, ypred_train)
        test_eval = evaluate(y_test, ypred_test)
        st.session_state.train_acc = train_eval["accuracy"]
        st.session_state.test_acc = test_eval["accuracy"]
        st.session_state.cm = test_eval["confusion_matrix"]
        st.session_state.report = test_eval["report"]
        st.session_state.trained = True
        st.success("Neural network trained successfully.")

    if not st.session_state.trained:
        st.info("Configure the architecture in the sidebar and click **Train / Retrain neural network**.")
    else:
        st.markdown(
            f"Train size: **{st.session_state.x_train.shape[0]}** rows. "
            f"Test size: **{st.session_state.x_test.shape[0]}** rows. "
            f"Architecture: **{st.session_state.mlp_model.hidden_layer_sizes}**. "
            f"Activation: **{st.session_state.mlp_model.activation}**."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Train Accuracy", f"{st.session_state.train_acc:.2%}")
        c2.metric("Test Accuracy", f"{st.session_state.test_acc:.2%}")
        c3.metric("Iterations run", f"{st.session_state.mlp_model.n_iter_}")

        if meets_accuracy_target(st.session_state.test_acc):
            st.success(f"Meets the 90% accuracy target ({st.session_state.test_acc:.2%}).")
        else:
            st.warning(f"Below the 90% accuracy target ({st.session_state.test_acc:.2%}). "
                        "Try tuning the architecture, activation, or learning rate.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion matrix (test set)**")
            confusion_matrix_chart(st.session_state.cm)
        with c2:
            st.markdown("**Loss curve**")
            loss_curve_chart(st.session_state.mlp_model.loss_curve_)

        with st.expander("Full classification report"):
            st.dataframe(pd.DataFrame(st.session_state.report).T, use_container_width=True)

# ==========================================================================
with tab_predict:
    st.subheader("Predict a student's admission outcome")

    if not st.session_state.trained:
        st.info("Train the neural network in the **Train & Evaluate Neural Network** tab first.")
    else:
        bundle = make_bundle(
            st.session_state.mlp_model, st.session_state.scaler,
            list(st.session_state.x_train.columns), threshold,
        )
        st.markdown("Enter applicant details:")
        raw_input = applicant_form()

        if st.button("🔮 Predict admission", type="primary"):
            pred, proba = predict_from_raw(bundle, raw_input)
            label = "Likely Admitted" if pred == 1 else "Likely Not Admitted"
            st.metric("Prediction", label, f"Predicted probability of admission: {proba:.1%}")

# ==========================================================================
with tab_model:
    st.subheader("Save trained model (and scaler) to disk")
    if not st.session_state.trained:
        st.info("Train the neural network first in the **Train & Evaluate Neural Network** tab.")
    else:
        bundle = make_bundle(
            st.session_state.mlp_model, st.session_state.scaler,
            list(st.session_state.x_train.columns), threshold,
        )
        buffer = serialize_bundle(bundle)

        st.download_button(
            label=f"Download model bundle as {config.MODEL_FILENAME}",
            data=buffer, file_name=config.MODEL_FILENAME, mime="application/octet-stream",
        )
        st.caption(
            "Bundles the MLPClassifier together with its scaler, feature column order, and the "
            "admission threshold used, so it can be reloaded for prediction without retraining."
        )

    st.markdown("---")
    st.subheader("Load a previously saved model bundle and predict")
    uploaded_model = st.file_uploader("Upload a pickled model bundle (.pkl)", type=["pkl", "pickle"], key="model_uploader")

    if uploaded_model is not None:
        loaded_bundle = load_bundle_from_bytes(uploaded_model.getvalue())
        feature_cols = loaded_bundle["feature_columns"]
        st.success("Model bundle loaded. Enter feature values to test a prediction.")

        input_df = generic_encoded_form(feature_cols, st.session_state.x_train if st.session_state.trained else encoded_df)
        if st.button("🔮 Predict with loaded model"):
            pred, proba = predict_from_encoded(loaded_bundle, input_df)
            label = "Likely Admitted" if pred == 1 else "Likely Not Admitted"
            if proba is not None:
                st.metric("Prediction (loaded model)", label, f"Predicted probability: {proba:.1%}")
            else:
                st.metric("Prediction (loaded model)", label)

st.sidebar.markdown("---")
st.sidebar.caption("Built from UCLA_Neural_Networks_Solution.ipynb. Modular architecture. scikit-learn.")
