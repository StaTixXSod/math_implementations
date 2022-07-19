from student_test import *
import numpy as np

v1 = np.random.binomial(1000, 0.01, (100))
v2 = np.random.binomial(100, 0.1, (100))

v1 = list(v1)
v2 = list(v2)

print(one_sample_ttest(v1, v2))