from student_test import *
import numpy as np
np.random.seed(43)

v1 = np.random.standard_t(100, 1000)
# rvs = stats.norm.rvs(loc=5, scale=10, size=(50, 2), random_state=42)
v1 = list(v1)

# print(one_sample_ttest(v1, v2))
# print(stats.ttest_ind(v1, v2))

# print(t_value(10.8, 10, 2, 25))
# print(paired_ttest_simp(89.9, 80.7, 11.3, 11.7, 20, 20))

# qqplot(v1)

# print(np.median(v1))
# print(median(v1))
print(var(v1))
print(np.var(v1))