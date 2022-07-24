from student_test import *
# from scipy.stats import ttest_rel, ttest_ind
import numpy as np
np.random.seed(43)

v1 = np.random.standard_t(100, 1000)
v2 = np.random.standard_t(100, 1000)

x = [4, 5, 2, 3, 1]
y = [2, 1, 4, 3, 5]
# x = [43, 21, 25, 42, 57, 59]
# y = [99, 65, 79, 75, 87, 81]

print(correlation(x, y))
print(np.corrcoef(x, y))
