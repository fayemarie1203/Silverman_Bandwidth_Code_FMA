# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 17:32:51 2025

@author: fayem
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from scipy.stats import iqr

# 1. Simulation d'un échantillon de taille n issu de N(0,1)
np.random.seed(42)
n = 100
data = np.random.normal(loc=0, scale=1, size=n)

# 2. Calcul de h selon la règle de Silverman
sigma_hat = np.std(data, ddof=1)
iqr_value = iqr(data)
h_silverman = 0.9 * min(sigma_hat, iqr_value / 1.34) * n ** (-1/5)
print(f"h_Silverman = {h_silverman:.3f}")

# 3. Estimation de densité à noyau avec noyau gaussien
kde = gaussian_kde(data, bw_method=h_silverman / data.std(ddof=1))
x_vals = np.linspace(-4, 4, 500)
kde_vals = kde(x_vals)

# 4. Courbe de la vraie densité
true_density = norm.pdf(x_vals)

# 5. Affichage
plt.figure(figsize=(8, 5))
plt.plot(x_vals, kde_vals, label="Densité estimée (Silverman)", color="blue")
plt.plot(x_vals, true_density, label="Densité vraie $\mathcal{N}(0,1)$", linestyle="--", color="black")
plt.title("Estimation de densité avec $h = h_{Silverman}$")
plt.xlabel("x")
plt.ylabel("Densité")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("silverman_gaussian.png")
plt.show()
