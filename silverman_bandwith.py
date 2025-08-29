# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 13:16:18 2025

@author: fayem
"""
import numpy as np
from scipy.stats import iqr

def silverman_bandwidth(data):
    n = len(data)
    std_dev = np.std(data, ddof=1)
    iqr_val = iqr(data)
    scale = min(std_dev, iqr_val / 1.34)
    h_silverman = 0.9 * scale * n ** (-1/5)
    return h_silverman

# Exemple d'utilisation
np.random.seed(0)
data = np.random.normal(0, 1, 100)
h_silverman = silverman_bandwidth(data)
print(f"Bande passante de Silverman : h = {h_silverman:.3f}")




