import numpy as np


def accuracy_score(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """correct / N"""
    N = y_true.shape[0]
    acc = (y_true == y_hat).sum() / N
    return acc


def precision_score(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """tp / (tp + fp)"""
    TP = ((y_hat == 1) & (y_true == 1)).sum()
    FP = ((y_hat == 1) & (y_true == 0)).sum()
    return TP / (TP + FP)


def recall_score(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """tp / (tp + fn)"""
    TP = ((y_hat == 1) & (y_true == 1)).sum()
    FN = ((y_hat == 0) & (y_true == 1)).sum()
    return TP / (TP + FN)


def f1_score(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """F1 = 2 * (precision * recall) / (precision + recall)"""
    precision = precision_score(y_true, y_hat)
    recall = recall_score(y_true, y_hat)
    return 2 * (precision * recall) / (precision + recall)


def fbeta_score(y_true: np.ndarray, y_hat: np.ndarray, beta: float = 0.5) -> float:
    """Fbeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)"""
    precision = precision_score(y_true, y_hat)
    recall = recall_score(y_true, y_hat)
    beta2 = beta**2
    return ((1 + beta2) * precision * recall) / (beta2 * precision + recall)
