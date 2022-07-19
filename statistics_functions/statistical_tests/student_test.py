import os
import sys

sys.path.append(os.getcwd())
from statistics_functions.functions import *

def one_sample_ttest(v1, v2):
    """Return ttest for population data
    and sample data

    Args:
        population (list): population data
        sample (list): sample data
    """
    pop_mean = mean(v1)
    sample_mean = mean(v2)
    sample_size = len(v2)**0.5
    sample_std = std(v2)

    t = (sample_mean - pop_mean) / (sample_std / sample_size)
    return t

def two_sample_ttest(v1, v2):
    pass

def paired_ttest(v1, v2):
    pass
