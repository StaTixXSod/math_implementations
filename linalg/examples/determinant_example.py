import os
import sys
sys.path.append(os.getcwd())

from linalg.functions import *
import numpy as np
np.random.seed(42)

a = np.random.randint(1, 15, size=(5, 5))

print("[INFO] Numpy version")
print(np.linalg.det(a))

print("\n[INFO] My version")
print(det(a))

"""
[INFO] Numpy version
-27279.999999999967

[INFO] My version
-27280
"""
