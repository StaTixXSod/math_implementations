import os
import sys
sys.path.append(os.getcwd())

from linear_algebra.functions import *
import numpy as np
np.random.seed(42)

a = np.random.randint(1, 15, size=(3, 3))

print("[INFO] Numpy version")
print(np.linalg.inv(a))

print("\n[INFO] My version")
a = list(a)
print_matrix(inverse(a), use_round=8)

"""
[INFO] Numpy version
[[-0.04910714  0.22767857 -0.23214286]
 [-0.20089286  0.02232143  0.23214286]
 [ 0.16517857 -0.12946429  0.05357143]]

[INFO] My version
-0.04910714	0.22767857	-0.23214286
-0.20089286	0.02232143	0.23214286
0.16517857	-0.12946429	0.05357143
"""
