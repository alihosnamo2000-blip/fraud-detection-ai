
# src/utils/metrics.py

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
    Returns:
        dict: Dictionary of computed metrics.
    """
    metrics = {
        'Precision': precision_score(y_true, y_pred, zero_division=0),

        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'Confusion Matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics

def compute_auc(y_true, y_scores):
    """
    Compute ROC-AUC score.
    Args:
        y_true (array-like): True labels.
        y_scores (array-like): Predicted probabilities or scores.
    Returns:
        float: AUC score.
    """
    return roc_auc_score(y_true, y_scores)
