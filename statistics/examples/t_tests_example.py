import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
np.random.seed(42)

from statistics.statistical_tests.student_test import *
from scipy.stats import ttest_rel, ttest_ind, ttest_1samp

v1 = np.random.standard_t(100, 1000)
v2 = np.random.standard_t(100, 1000)

sc_stat_samp, sc_pvalue_samp = ttest_1samp(v1, np.mean(v2))
my_stat_samp, my_pvalue_samp = one_sample_ttest(v1, np.mean(v2))

print("\n[INFO] One sample t test")
print(pd.DataFrame(
    {"Statistic": [sc_stat_samp, my_stat_samp],
    "Pvalue": [sc_pvalue_samp, my_pvalue_samp]}
    , index=["scipy", "my version"]
))

sc_stat_ind, sc_pvalue_ind = ttest_ind(v1, v2)
my_stat_ind, my_pvalue_ind = two_sample_ttest(v1, v2)

print("\n[INFO] Two sample t test for independent variables")
print(pd.DataFrame(
    {"Statistic": [sc_stat_ind, my_stat_ind],
    "Pvalue": [sc_pvalue_ind, my_pvalue_ind]}
    , index=["scipy", "my version"]
))

print("\n[INFO] Two sample t test for dependent variables")
sc_stat_rel, sc_pvalue_rel = ttest_rel(v1, v2)
my_stat_rel, my_pvalue_rel = paired_ttest(v1, v2)

print(pd.DataFrame(
    {"Statistic": [sc_stat_rel, my_stat_rel],
    "Pvalue": [sc_pvalue_rel, my_pvalue_rel]}
    , index=["scipy", "my version"]
))

"""
[INFO] One sample t test
            Statistic    Pvalue
scipy        0.554174  0.579583
my version   0.554174  0.579583

[INFO] Two sample t test for independent variables
            Statistic    Pvalue
scipy        0.388072  0.698004
my version   0.388072  0.698004

[INFO] Two sample t test for dependent variables
            Statistic    Pvalue
scipy        0.384084  0.700998
my version   0.384084  0.700998
"""