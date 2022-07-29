import os
import sys

sys.path.append(os.getcwd())
from linalg.functions import matrix_rank
import numpy as np
np.random.seed(42)

a = np.random.random((6, 5))

print(np.linalg.matrix_rank(a))
print(matrix_rank(a))
