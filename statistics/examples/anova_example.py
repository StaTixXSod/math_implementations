import os
import sys
sys.path.append(os.getcwd())

import statsmodels.api as sm
from statsmodels.formula.api import ols

import pandas as pd
from statistics.statistical_tests.anova import one_way_anova, two_way_anova
from scipy.stats import f_oneway

l1 = [3, 1, 2]
l2 = [5, 3, 4]
l3 = [7, 6, 5]

my_f_stat, my_pvalue = one_way_anova(l1, l2, l3)
sc_f_stat, sc_pvalue = f_oneway(l1, l2, l3)

print("\n[INFO] One way ANOVA test")
print(pd.DataFrame(
    {"Statistic": [sc_f_stat, my_f_stat],
    "Pvalue": [sc_pvalue, my_pvalue]}
    , index=["scipy", "my version"]
))


print("\n[INFO] Two way ANOVA test")
data = pd.read_csv("data/atherosclerosis.csv")
statistic = two_way_anova(data, features=["age", "dose"], target='expr')
print(statistic.to_string(index=False))
print()

model = ols('expr ~ C(age) + C(dose) + C(age):C(dose)', data=data).fit()
print(sm.stats.anova_lm(model, typ=2))

"""
[INFO] One way ANOVA test
            Statistic  Pvalue
scipy            12.0   0.008
my version       12.0   0.008

[INFO] Two way ANOVA test
 feature       sum_sq df         F    Pvalue
     age   197.452754  1  7.449841  0.008313
    dose    16.912241  1  0.638094  0.427552
age:dose     0.927077  1  0.034978  0.852272
residual  1590.257424 60      None      None

                     sum_sq    df         F    PR(>F)
C(age)           197.452754   1.0  7.449841  0.008313
C(dose)           16.912241   1.0  0.638094  0.427552
C(age):C(dose)     0.927077   1.0  0.034978  0.852272
Residual        1590.257424  60.0       NaN       NaN
"""