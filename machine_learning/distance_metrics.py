from statistical_functions.functions import *
from typing import Literal, Union
import numpy as np
import pandas as pd

distance_types = Union[list, np.ndarray, pd.Series, pd.DataFrame]


def euclidean_distance(p: distance_types, q: distance_types) -> int:
    """
    Calculate the Euclidean metric between 2 points

    Steps:
    ------
    1. Calculate the difference between points
    2. Calculate the square of difference to get rid of negative signs
    3. Sum it up
    4. Calculate the square root to normalize values

    Args:
        p: the coordinates of the first point
        q: the coordinates of the second point

    Returns: the Euclidean distance between 2 points
    """
    p = np.array(p)
    q = np.array(q)

    difference = p - q
    squared_difference = squared(difference)
    summed_difference = np.sum(squared_difference)
    d = np.sqrt(summed_difference)

    return d


def manhattan_distance(p: distance_types, q: distance_types) -> float:
    """
    Calculate the Manhattan metric between 2 points.

    Formula:\n
    - `Σ| p - q |`\n
        where:
            p: coordinates of the first point \n
            d: coordinates of the second point

    Args:
        p: the coordinates of the first point
        q: the coordinates of the second point

    Returns: the Manhattan distance between 2 points
    """
    p = np.array(p)
    q = np.array(q)

    d = np.sum(np.abs(p - q))
    return d


def max_distance(p: distance_types, q: distance_types) -> int:
    """
    Calculate the Max-metric between 2 points
    Args:
        p: the coordinates of the first point
        q: the coordinates of the second point

    Returns: the Max-metric distance
    """
    p = np.array(p)
    q = np.array(q)

    d = np.max(np.abs(p - q))
    return d


def linear_combination(restore_feature, distance):
    num = np.sum(restore_feature / distance)
    den = np.sum(1 / distance)
    return num / den


def restore_missing_data(data: pd.DataFrame,
                         feature2restore: str,
                         method: Literal["euclidean", "manhattan", "max"] = "euclidean") -> pd.Series:
    """
    Restore missing data using distance metric approach.
    Args:
        data: (pandas.DataFrame) the data with missing values
        feature2restore: choose the feature with missing values
        method: choose the distance function to restore the values

    Returns: the series of restored feature with filled values

    """
    # 1. Define metric function
    methods = ["euclidean", "manhattan", "max"]
    if method not in methods:
        raise ValueError(f"The method should be from list: {methods}")
    if method == "manhattan":
        metric = manhattan_distance
    elif method == "max":
        metric = max_distance
    else:
        metric = euclidean_distance

    # 2. Calculate the distances between rows
    features = data.drop(feature2restore, axis=1)  # Drop column that we need to restore
    drop_indices = data[feature2restore].isna()  # Get indices with NA values
    rest_rows: pd.DataFrame = features.loc[~drop_indices, :]
    rows2restore: pd.DataFrame = features.loc[drop_indices, :]
    distances = rest_rows.apply(lambda x: metric(x, rows2restore), axis=1)

    # 3. Calculate the missing values
    restore_feature = data[feature2restore]  # The column that we need to restore
    restored = linear_combination(restore_feature, distances)
    return data[feature2restore].fillna(restored)
