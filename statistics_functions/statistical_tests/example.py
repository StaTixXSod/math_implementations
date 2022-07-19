from student_test import *
import numpy as np
# from scipy import stats 

# np.random.seed(42)

# v1 = np.random.binomial(1000, 0.01, 100)
# v2 = np.random.binomial(100, 0.1, 100)

# rvs = stats.norm.rvs(loc=5, scale=10, size=(50, 2), random_state=42)

# v1 = list(v1)
# v2 = list(v2)

# print(one_sample_ttest(v1, v2))
# print(stats.ttest_ind(v1, v2))

# print(t_value(10.8, 10, 2, 25))
print(paired_ttest_simp(89.9, 80.7, 11.3, 11.7, 20, 20))