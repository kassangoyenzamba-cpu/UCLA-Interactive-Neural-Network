"""
Global configuration: column names, defaults, file paths.
No Streamlit code here so it can be imported by any module or test.
"""

TARGET_RAW = "Admit_Chance"
CATEGORICAL_COLS = ["University_Rating", "Research"]

DEFAULT_THRESHOLD = 0.8
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 123

DEFAULT_HIDDEN_LAYERS = (3, 4)
DEFAULT_ACTIVATION = "relu"
DEFAULT_SOLVER = "adam"
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_ITER = 200
DEFAULT_LEARNING_RATE_INIT = 0.001
DEFAULT_ALPHA = 0.0001

CSV_PATH = "Admission.csv"
MODEL_DIR = "models"
MODEL_FILENAME = "ucla_admission_model.pkl"
