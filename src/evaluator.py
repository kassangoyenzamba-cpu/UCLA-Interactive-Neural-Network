"""
Evaluation metric helpers. Pure scikit-learn, no Streamlit.
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, output_dict=True),
    }


def meets_accuracy_target(accuracy: float, target: float = 0.90) -> bool:
    return accuracy >= target
