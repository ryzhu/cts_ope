import sys
print(sys.path)
import numpy as np
with dview.sync_imports():
    from numpy import exp, where, mean, minimum
from hiv import HIVTreatment as model


import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import sem
from scipy import stats

