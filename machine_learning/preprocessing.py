import numpy as np


def standard_scaler(arr: np.ndarray) -> np.ndarray:
    """
    Standard Scaler. This method moves the mean of data to 0 and scales the data to unit variance.

    Info:
    -----
    Actually Standard Scaler is the Z score. When we see the z value, we know, that this value was standardized with
    this method. To standardize the value in unit variance, we should subtract the mean value from each array value
    to remove the mean of data (or just move the mean of data to 0) and then divide each value by std value to
    scale it in unit variance.

    Formula:
    ---------
    Z = (ai - M) / S,
        where:
            Z: scaled array
            ai: each value in array
            M: the mean of array
            S: the standard deviation value

    Args:
        arr: data array

    Returns:
        Normalized array using Z values method
    """
    m = np.mean(arr)
    s = np.std(arr)
    return (arr - m) / s


def min_max_scaler(arr: np.ndarray) -> np.ndarray:
    min_value = np.min(arr)
    max_value = np.max(arr)
    std = (arr - min_value) / (max_value - min_value)
    return std * (max_value - min_value) + min_value
