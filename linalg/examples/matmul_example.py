import os
import sys

sys.path.append(os.getcwd())
from linalg.functions import *
import numpy as np

np.random.seed(42)

a = np.random.random((3, 2))
b = np.random.random((3, 2))

print("[INFO] Numpy version")
print(a.T @ b)
print()
print(a @ b.T)

print("\n[INFO] My version")
print_matrix(matmul(transpose(a), b), use_round=8)
print_matrix(matmul(a, transpose(b)), use_round=8)

"""
[INFO] Numpy version
[[0.46497875 0.99404657]
 [0.41829459 1.39868033]]

[[0.8452407  0.89831642 0.92981689]
 [0.56106055 0.8639062  0.59571249]
 [0.14418086 0.20424059 0.15451219]]

[INFO] My version
0.46497875	0.99404657
0.41829459	1.39868033

0.8452407	0.89831642	0.92981689
0.56106055	0.8639062	0.59571249
0.14418086	0.20424059	0.15451219
"""
