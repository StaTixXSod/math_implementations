import os
import sys

sys.path.append(os.getcwd())
from linalg.functions import matrix_rank
import numpy as np

np.random.seed(42)

a = np.random.random((6, 5))

print("[INFO] Numpy: ", np.linalg.matrix_rank(a))
print("[INFO] Hand version: ", matrix_rank(a))

for _ in range(10):
    i = np.random.randint(3, 6, size=1)[0]
    j = np.random.randint(3, 6, size=1)[0]

    a = np.random.random((i, j))

    my_rank = matrix_rank(a)
    numpy_rank = np.linalg.matrix_rank(a)

    assert my_rank == numpy_rank
