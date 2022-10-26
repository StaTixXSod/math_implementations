from scipy.stats import chisquare
from statistics_functions.statistical_tests import pearson_chi_square as chi
import numpy as np


observed = np.random.randint(50, size=5)
print(chi.chi2_test(observed))
print(chisquare(observed))

"""
(37.4375, 1.463624085985581e-07)
Power_divergenceResult(statistic=37.4375, pvalue=1.463624085985581e-07)
"""