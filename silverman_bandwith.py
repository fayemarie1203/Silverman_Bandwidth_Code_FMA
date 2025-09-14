import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1) Données : Laplace(0, b) avec variance 1  =>  b = 1/sqrt(2)
# -------------------------------
n = 100
b = 1 / np.sqrt(2.0)  # échelle Laplace pour Var=1
rng = np.random.default_rng(42)
x = rng.laplace(loc=0.0, scale=b, size=n)

# -------------------------------
# 2) Règle de Silverman
#    h = 0.9 * min(sigma_hat, IQR/1.34) * n^{-1/5}
# -------------------------------
sigma_hat = np.std(x, ddof=1)
q1, q3 = np.quantile(x, [0.25, 0.75])
iqr = q3 - q1
h_sil = 0.9 * min(sigma_hat, iqr / 1.34) * n ** (-1/5)

print(f"Silverman bandwidth h = {h_sil:.4f}")

# -------------------------------
# 3) Estimateur à noyau gaussien
#    \hat p_h(t) = (1/(n h)) * sum_i phi((t - x_i)/h), phi = N(0,1)
# -------------------------------
def gaussian_kernel(u):
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

# grille pour évaluation
x_grid = np.linspace(-5, 5, 1000)

# KDE "maison"
diff = (x_grid[:, None] - x[None, :]) / h_sil  # shape (len(grid), n)
k_vals = gaussian_kernel(diff)
p_hat = k_vals.mean(axis=1) / h_sil  # (1/(n h)) * sum phi

# -------------------------------
# 4) Densité vraie Laplace(0,b) pour comparaison
#    f(t) = (1/(2b)) * exp(-|t|/b)
# -------------------------------
f_true = np.exp(-np.abs(x_grid) / b) / (2 * b)

# -------------------------------
# 5) Figure
# -------------------------------
# -------------------------------

plt.figure(figsize=(7.2, 4.8))
plt.hist(x, bins=20, density=True, alpha=0.25, label="Histogramme (dens.)")
plt.plot(x_grid, p_hat, lw=2, label=r"Estimation noyau (gaussien, $h_{\rm Sil}$)")
plt.plot(x_grid, f_true, lw=2, ls="--", label="Densité Laplace vraie")
plt.xlabel("x"); plt.ylabel("Densité")
plt.title("KDE gaussien avec bande passante de Silverman (échantillon Laplace)")
plt.legend()
plt.tight_layout()
plt.savefig("estimation-dens-hsil.png", dpi=200)  # image enregistrée pour LaTeX
plt.show()   # <-- afficher à l'écran
# plt.close()



