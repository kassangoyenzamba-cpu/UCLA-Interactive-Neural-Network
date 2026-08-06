# UCLA Admission Neural Network: Dashboard Live Application

Click the badge or the text link below to open the deployment in a new tab:
<a href="https://atcqsavp3zea84cu22m88t.streamlit.app/" target="_blank">
  <img src="https://streamlit.io" alt="Streamlit App">
</a>

<a href="https://atcqsavp3zea84cu22m88t.streamlit.app/" target="_blank">👉 Open Interactive ML Dashboard</a>


## Folder structure

```
ucla_admission_app/
├── app.py                    # entry point, layout, tabs, sidebar only
├── config.py                 # column names, defaults, file paths
├── requirements.txt
│
├── src/                      # pure Python, no Streamlit imports, unit testable
│   ├── data_loader.py          # load_csv() with @st.cache_data
│   ├── preprocessor.py         # prep_data(), encode_data(), encode_single_input()
│   ├── trainer.py              # split_data(), scale_features(), parse_hidden_layers(), train_mlp()
│   ├── evaluator.py            # evaluate(), meets_accuracy_target()
│   └── predictor.py            # bundle save and load, predict_from_raw(), predict_from_encoded()
│
├── components/                 # Streamlit dependent, reusable UI pieces
│   ├── inputs.py                 # sidebar controls, applicant form
│   └── charts.py                  # confusion matrix, correlation heatmap, loss curve, etc.
│
├── tests/
│   ├── test_preprocessor.py      # 8 tests, target binarization, encoding, single row alignment
│   └── test_trainer.py            # 10 tests, split, scaling, hidden layer parsing, training, evaluation
│
└── Admission.csv
```

`app.py` never contains model logic, it only wires `src/` functions to
`components/` widgets and lays out tabs. `src/` never imports
`streamlit` (except for the two caching decorators), so every core
function can be tested with plain pytest and reused outside the
dashboard, for example in a batch scoring script.

## Setup

1. Put the whole `ucla_admission_app/` folder in VS Code, it already
   includes `Admission.csv`.
2. Optionally create and activate a virtual environment.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```
5. Run the tests:
   ```
   pytest tests/ -v
   ```
   All 18 tests should pass.

## What it does (same as the notebook and single file version)
- **Data Overview**: preview, shape, dtypes, missing values, summary stats, quick observations
- **EDA**: correlation heatmap, GRE vs. TOEFL scatter colored by admission outcome, feature
  distributions, boxplots vs. admission outcome
- **Preprocessing**: adjustable Admit_Chance threshold (default 0.8) to binarize the target,
  categorical casting and one hot encoding of University_Rating and Research, before and after
  MinMax scaling visualization (scaler fit on train only, avoiding data leakage), and a download
  button for the encoded dataset
- **Train & Evaluate Neural Network**: full MLPClassifier configuration in the sidebar, hidden
  layer sizes for any number of layers, activation function, solver, batch size, max iterations,
  learning rate, and L2 penalty, with train and test accuracy, confusion matrix, loss curve, and
  full classification report (goal: accuracy at least 90%)
- **Predict Admission**: applicant form (GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research)
  that encodes and scales inputs automatically and returns an admission prediction with probability
- **Save / Load Model**: `.pkl` bundle (model, scaler, feature order, threshold used); reloading
  uses `@st.cache_resource` so an uploaded model is deserialized once, not on every rerun

## Notes on the refactor
- Verified identical results to the original single file app on the same data and seed: train
  accuracy 92.25%, test accuracy 88% with the default architecture `hidden_layer_sizes=(3, 4)`,
  matching the notebook.
- `encode_single_input()` is tested against an unseen category value (a University_Rating not
  present in the training sample) to confirm it produces an all zero dummy row instead of
  crashing, which matters for a live prediction form where a user could in principle pick a
  combination the model never saw.
- `parse_hidden_layers()` has its own tests for the empty string and malformed text cases, since
  this comes straight from a free text sidebar input and needs to fail safely rather than crash
  the whole app.
