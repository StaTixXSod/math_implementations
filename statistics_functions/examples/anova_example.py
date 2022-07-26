import os
import sys
sys.path.append(os.getcwd())

import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

import pandas as pd
from statistics_functions.statistical_tests.anova import one_way_anova, two_way_anova
from scipy.stats import f_oneway

# l1 = [3, 1, 2]
# l2 = [5, 3, 4]
# l3 = [7, 6, 5]

# my_f_stat, my_pvalue = one_way_anova(l1, l2, l3)
# sc_f_stat, sc_pvalue = f_oneway(l1, l2, l3)

# print("\n[INFO] One way ANOVA test")
# print(pd.DataFrame(
#     {"Statistic": [sc_f_stat, my_f_stat],
#     "Pvalue": [sc_pvalue, my_pvalue]}
#     , index=["scipy", "my version"]
# ))

"""
[INFO] One way ANOVA test
            Statistic  Pvalue
scipy            12.0   0.008
my version       12.0   0.008
"""

data = pd.read_csv("data/atherosclerosis.csv")
print(two_way_anova(data, features=["age", "dose"], target='expr'))





# model = ols('expr ~ C(age) + C(dose) + C(age):C(dose)', data=data).fit()
# # model = ols('expr ~ C(age) + C(dose)', data=data).fit()
# print(sm.stats.anova_lm(model, typ=2))
