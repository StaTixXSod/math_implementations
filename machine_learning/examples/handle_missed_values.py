from machine_learning.distance_metrics import *
import pandas as pd

df = pd.DataFrame(
    {'P1': [3, 5, 4, 5], 'P2': [4, 5, 3, 4], 'P3': [5, 5, 3, 3], 'P4': [3, 4, 2, 3], 'P': [4, 3, 5, np.NaN]},
    index=['A1', 'A2', 'A3', 'A'])

print(df)
euc = restore_missing_data(df, 'P', method="euclidean")
man = restore_missing_data(df, 'P', method="manhattan")
mx = restore_missing_data(df, 'P', method="max")

print(pd.DataFrame(
    data=[euc, man, mx],
    index=["Euclidean", "Manhattan", "Max"]
))

"""
    P1  P2  P3  P4    P
A1   3   4   5   3  4.0
A2   5   5   5   4  3.0
A3   4   3   3   2  5.0
A    5   4   3   3  NaN
            A1   A2   A3         A
Euclidean  4.0  3.0  5.0  4.126275
Manhattan  4.0  3.0  5.0  4.100000
Max        4.0  3.0  5.0  4.250000
"""