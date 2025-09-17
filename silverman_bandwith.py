
import numpy as np
import matplotlib.pyplot as plt

# Reproducibilité
rng = np.random.default_rng(7)

# --- Données : échantillon simulé (Laplace(0,1)) ---
n = 100
X = rng.laplace(loc=0.0, scale=1.0, size=n)

# --- Règle de Silverman ---
def iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

sigma_hat = np.std(X, ddof=1)
iqr_hat   = iqr(X)
h_sil     = 0.9 * min(sigma_hat, iqr_hat / 1.34) * (n ** (-1/5))

# --- Noyaux ---
def K_gauss(u):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * u * u)

def K_uniform(u):
    return 0.5 * ((np.abs(u) <= 1).astype(float))

def K_tri(u):
    a = 1.0 - np.abs(u)
    a[a < 0] = 0.0
    return a

def K_epan(u):
    mask = (np.abs(u) <= 1)
    res = np.zeros_like(u, dtype=float)
    res[mask] = 0.75 * (1 - u[mask]**2)
    return res

def K_biweight(u):
    mask = (np.abs(u) <= 1)
    res = np.zeros_like(u, dtype=float)
    res[mask] = (15.0/16.0) * (1 - u[mask]**2)**2
    return res

def K_silverman(u):
    # Noyau de Silverman (ordre 4), oscillant
    return 0.5 * np.exp(-np.abs(u)/np.sqrt(2)) * np.sin(np.abs(u)/np.sqrt(2) + np.pi/4)

kernels = {
    "Gaussien": K_gauss,
    "Épanechnikov": K_epan,
    "Triangulaire": K_tri,
    "Uniforme": K_uniform,
    "Biweight": K_biweight,
    "Silverman": K_silverman,
}

# --- Estimateur à noyau ---
def kde(x_grid, data, h, K):
    u = (x_grid[:, None] - data[None, :]) / h
    return (K(u).sum(axis=1)) / (len(data) * h)

# Grille de tracé
x_min = np.quantile(X, 0.01) - 3*h_sil
x_max = np.quantile(X, 0.99) + 3*h_sil
x_grid = np.linspace(x_min, x_max, 1200)

# Estimations avec le même h_sil
estimates = {name: kde(x_grid, X, h_sil, K) for name, K in kernels.items()}

# Densité vraie Laplace(0,1) pour repère (optionnel)
def laplace_pdf(x, loc=0.0, scale=1.0):
    return 0.5/scale * np.exp(-np.abs(x-loc)/scale)
true_pdf = laplace_pdf(x_grid, 0.0, 1.0)

# Tracé (AUCUN histogramme)
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, y in estimates.items():
    ax.plot(x_grid, y, label=name, linewidth=2)
ax.plot(x_grid, true_pdf, linestyle="--", linewidth=2, label="Densité vraie (Laplace)", alpha=0.9)

ax.set_xlabel("x")
ax.set_ylabel("densité estimée")
ax.set_title(f"Règle de Silverman (n={n}) — h_sil ≈ {h_sil:.4f}")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("silverman_multi_kernels.png", dpi=160)

print("h_sil ≈", h_sil)


import matplotlib.patheffects as pe  # <-- pour le contour blanc (optionnel)

# --- palette + styles bien contrastés ---
colors = {
    "Gaussien":        "#1f77b4",  # bleu
    "Épanechnikov":    "#ff7f0e",  # orange
    "Triangulaire":    "#2ca02c",  # vert
    "Uniforme":        "#d62728",  # rouge
    "Biweight":        "#9467bd",  # violet
    "Silverman":       "#8c564b",  # marron
}
styles = {
    "Gaussien":        "-",
    "Épanechnikov":    "--",
    "Triangulaire":    "-.",
    "Uniforme":        ":",
    "Biweight":        "-",
    "Silverman":       "--",
}

# --- TRACÉ (remplace ton bloc fig, ax = plt.subplots(...) jusqu’au savefig) ---
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.set_facecolor("white")

# lignes plus épaisses + léger contour blanc pour bien voir
effect = [pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()]

for k, y in estimates.items():
    ax.plot(
        x_grid, y,
        label=k,
        color=colors[k],
        linestyle=styles[k],
        linewidth=2.4,
        alpha=0.95,
        zorder=3,
        path_effects=effect,   # retire cette ligne si tu ne veux pas de contour
    )

# densité vraie en noir pointillé, au-dessus
ax.plot(
    x_grid, true_pdf,
    linestyle="--",
    linewidth=2.6,
    color="black",
    alpha=0.9,
    label="Densité vraie (Laplace)",
    zorder=4,
    path_effects=effect,       # idem : optionnel
)

ax.set_xlabel("x")
ax.set_ylabel("densité estimée")
ax.set_title(f"Règle de Silverman (n={n}) — h_sil ≈ {h_sil:.4f}")
ax.grid(True, alpha=0.25)
leg = ax.legend(loc="best", frameon=True, framealpha=0.9, ncol=1)
fig.tight_layout()
fig.savefig("silverman_multi_kernels.png", dpi=180)




