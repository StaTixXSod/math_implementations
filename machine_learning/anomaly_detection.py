from typing import Union, Literal
from machine_learning.preprocessing import standard_scaler
from statistical_functions.functions import get_quantile_info
import math
import numpy as np


def zscore_detection(arr: np.ndarray, thresh: int = 3) -> np.ndarray:
    """
    Z Score is a statistical method for anomaly detection. If the z value is outside 3 sigma (std),
    then the index of that value appended into the returning list
    Args:
        arr: (np.ndarray) the array
        thresh:

    Returns:
        the list of indices with detected anomaly
    """
    z = standard_scaler(arr)
    outlier_indices = np.where((z > thresh) | (z < -thresh))
    return outlier_indices[0]


def chauvenet_function(ai: float, arr: list) -> float:
    """Transforms current value with z method and returns erfc function output"""
    m = np.mean(arr)
    s = np.std(arr)
    z = np.abs(ai - m) / s
    return math.erfc(z)


def chauvenet_detection(arr):
    """
    Anomaly detection using Chauvenet's criterion.

    Info:
    -----
    This method performed iteratively first from maximum value to minimum
    and then, when the value from array, sorted from maximum to minimum is not an outlier, we
    perform the same detection method from minimum to maximum.

    Formula:
    --------
    The value is removed from the array if the following condition is True:

    - `erfc(abs(z)) < (1/2n)`
    where:
        n: the number of values in array \n
        z: the standard scaled value \n
        abs: absolute value function \n
        erfc: complementary error function at x \n

    - `erfc(x) = (2/√pi) * x->inf∫ (e^-t^2dt)`
    where:
        ∫: integral from x to infinite
        √: square root
    Args:
        arr: array data

    Returns:
        the same list without anomalies
    """
    # From maximum value to min
    for ai in sorted(arr, reverse=True):
        n = len(arr)
        criterion = 1 / (2 * n)
        prob = chauvenet_function(ai, arr)

        if prob < criterion:
            arr.remove(ai)
        else:  # break if not outlier
            break

    # Then from minimum to maximum
    for ai in sorted(arr):
        n = len(arr)
        criterion = 1 / (2 * n)
        prob = chauvenet_function(ai, arr)

        if prob < criterion:
            arr.remove(ai)

    return np.array(arr)


def iqr_outlier_detection(arr: Union[list, np.ndarray]) -> np.ndarray:
    """
    Outlier detection using Interquartile Range (IQR).

    Info:
    -----
    This method is used for outlier detection. Instead of mean and std it
    calculates the median, Q25 and Q75 values. If the value of array is lower or greater calculated border,
    the value consider as outlier.

    To calculate the borders, use formula:\n
    - `IQR - Q75 - Q25`\n
    - `lower = Q25 - 1.5 * IQR` \n
    - `upper = Q75 + 1.5 * IQR` \n
        where:
            IQR: interquartile range \n
            lower: lower border. The values lower that border are outliers \n
            upper: upper border. The values greater that border are outliers \n

    Args:
        arr: vector of data

    Returns:
        the filtered array without outliers
    """
    arr = np.array(arr)

    q25, median, q75 = get_quantile_info(arr)
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr

    return arr[(arr > lower) & (arr < upper)]


a = [-6, 0, 1, 2, 4, 5, 5, 6, 7, 100]
print(iqr_outlier_detection(a))
