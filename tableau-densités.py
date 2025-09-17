# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 20:18:39 2025

@author: fayem
"""

# -*- coding: utf-8 -*-
"""
Comparaison quantitative via le MISE (M=1000, n=200)
- 3 densités : bimodale, gaussienne, smooth comb
- 4 noyaux : gaussien, epanechnikov, uniforme, silverman
- h choisi par CV leave-one-out pour chaque noyau
- MISE calculé par intégration numérique
- Génère 3 tableaux LaTeX avec moyenne ± IC95%
"""

import numpy as np
from scipy.stats import iqr, norm
from pathlib import Path

# ----------------------------
# Paramètres globaux
# ----------------------------
M = 50       # nb de répétitions
n = 200         # taille échantillon
G = 2000        # nb de points de grille pour l'intégration (plus => plus précis)
seed = 12345    # graine pour reproductibilité
np.random.seed(seed)

# ----------------------------
# Noyaux
# ----------------------------
def K_gaussian(u):
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5*u*u)

def K_epanechnikov(u):
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1.0
    out[mask] = 0.75*(1 - u[mask]*u[mask])
    return out

def K_uniform(u):
    return 0.5 * (np.abs(u) <= 1.0).astype(float)

def K_silverman(u):
    a = np.abs(u)/np.sqrt(2)
    return 0.5*np.exp(-a)*np.sin(a + np.pi/4)

KERNELS = {
    "Gaussien": K_gaussian,
    "Épanechnikov": K_epanechnikov,
    "Uniforme": K_uniform,
    "Silverman": K_silverman,
}

# ----------------------------
# Densités vraies + simulateurs
# ----------------------------
def p_bimodal(x):
    return 0.5*norm.pdf(x, loc=-2, scale=1) + 0.5*norm.pdf(x, loc= 2, scale=1)

def r_bimodal(n):
    z = np.random.binomial(1, 0.5, size=n)
    return np.where(z==1, np.random.normal(-2,1,size=n), np.random.normal(2,1,size=n))

def p_gaussian(x):
    return norm.pdf(x, loc=0, scale=1)

def r_gaussian(n):
    return np.random.normal(0,1,size=n)

# Smooth comb: M bosses gaussiennes entre -2 et 2, sigma=0.15
MU_GRID = np.linspace(-2, 2, 9)
SIG_SC = 0.15
def p_smoothcomb(x):
    vals = np.zeros_like(x, dtype=float)
    for mu in MU_GRID:
        vals += norm.pdf(x, loc=mu, scale=SIG_SC)
    return vals / len(MU_GRID)

def r_smoothcomb(n):
    idx = np.random.randint(0, len(MU_GRID), size=n)
    return np.random.normal(loc=MU_GRID[idx], scale=SIG_SC, size=n)

DENSITIES = {
    "bimodale": (p_bimodal, r_bimodal),
    "gaussienne": (p_gaussian, r_gaussian),
    "smoothcomb": (p_smoothcomb, r_smoothcomb),
}

# ----------------------------
# Outils : grille x, KDE, CV, ISE
# ----------------------------
def silverman_href(x):
    """Référence de Silverman (ROT) pour fixer une échelle de h (gaussien)."""
    s = np.std(x, ddof=1)
    sigma = min(s, iqr(x)/1.34)
    return 0.9 * sigma * (len(x) ** (-1/5))

def kde_on_grid(x, h, K, x_grid):
    u = (x_grid[:, None] - x[None, :]) / h      # (G, n)
    return np.mean(K(u), axis=1) / h            # (G,)

def loo_term_at_data(x, h, K):
    """Somme des densités LOO en xi : sum_i p_{-i,h}(xi).
       Implémentation vectorisée via matrice de noyaux."""
    n = len(x)
    U = (x[:, None] - x[None, :]) / h           # (n, n)
    Kmat = K(U)
    np.fill_diagonal(Kmat, 0.0)                 # j ≠ i
    # p_{-i,h}(xi) = (1/((n-1)h)) * sum_{j≠i} K((xi-xj)/h)
    p_loo = np.sum(Kmat, axis=1) / ((n-1)*h)
    return np.sum(p_loo)                         # somme sur i

def cv_score(x, h, K, x_grid):
    """CV(h) = ∫ \hat p_h^2 dx - (2/n) ∑ p_{-i,h}(X_i)   (intégration numérique)
       On approxime l'intégrale sur x_grid (trapezes)."""
    kde = kde_on_grid(x, h, K, x_grid)
    term1 = np.trapz(kde*kde, x_grid)
    term2 = (2/len(x)) * loo_term_at_data(x, h, K)
    return term1 - term2

def ise_against_truth(x, h, K, x_grid, p_true):
    kde = kde_on_grid(x, h, K, x_grid)
    diff2 = (kde - p_true(x_grid))**2
    return np.trapz(diff2, x_grid)

# ----------------------------
# Boucle principale (séquentielle)
# ----------------------------
def one_density_tables(density_name, M, n, kernels=KERNELS, G=2000, verbose=True):
    p_true, rgen = DENSITIES[density_name]
    results = {k: [] for k in kernels.keys()}   # ISEs
    chosen_h = {k: [] for k in kernels.keys()}  # h choisis

    for m in range(1, M+1):
        x = rgen(n)
        # échelle & grille pour le h (autour de h_ref)
        h_ref = max(1e-3, silverman_href(x))
        h_grid = np.linspace(0.3*h_ref, 2.0*h_ref, 41)  # 41 valeurs entre 0.3 et 2.0 fois href

        # grille x pour intégrations (dépend de h_max pour couvrir les queues)
        h_max = h_grid.max()
        xmin = np.min(x) - 4*h_max
        xmax = np.max(x) + 4*h_max
        x_grid = np.linspace(xmin, xmax, G)

        for kname, Kfun in kernels.items():
            # sélection h par CV
            cv_vals = [cv_score(x, h, Kfun, x_grid) for h in h_grid]
            hstar = h_grid[int(np.argmin(cv_vals))]
            chosen_h[kname].append(hstar)
            # ISE avec le h* retenu
            ise = ise_against_truth(x, hstar, Kfun, x_grid, p_true)
            results[kname].append(ise)

        if verbose and (m % 50 == 0):
            print(f"[{density_name} | n={n}] {m}/{M}")

    # stats (moyenne et IC 95%)
    table_rows = []
    for kname in kernels.keys():
        arr = np.array(results[kname], dtype=float)
        mean = arr.mean()
        se = arr.std(ddof=1) / np.sqrt(M)
        ic = 1.96 * se
        table_rows.append((kname, mean, ic))

    return table_rows, chosen_h

def write_latex_table(filename, caption, label, rows):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{|c|c|}\n\\hline\n")
        f.write("\\textbf{Noyau} & \\textbf{MISE moyen $\\pm$ IC 95\\,\\%} \\\\\n\\hline\n")
        for name, mean, ic in rows:
            f.write(f"{name} & ${mean:.4f} \\;\\pm\\; {ic:.5f}$ \\\\\n\\hline\n")
        f.write("\\end{tabular}\n\\end{table}\n")

# ----------------------------
# Exécution : 3 tableaux (n=200, M=1000)
# ----------------------------
if __name__ == "__main__":
    outdir = Path.cwd() / "resultats_mise"
    outdir.mkdir(exist_ok=True, parents=True)

    # Bimodale
    rows_bi, _ = one_density_tables("bimodale", M=M, n=n, G=G, verbose=True)
    write_latex_table(
        outdir / "tab_mise_bimodale_n200.tex",
        "Densité \\textbf{bimodale} ($M=1000$, $n=200$) — MISE moyen $\\pm$ IC 95\\,\\%",
        "tab:mise-bimodale-n200",
        rows_bi
    )

    # Gaussienne
    rows_g, _ = one_density_tables("gaussienne", M=M, n=n, G=G, verbose=True)
    write_latex_table(
        outdir / "tab_mise_gaussienne_n200.tex",
        "Densité \\textbf{gaussienne} ($M=1000$, $n=200$) — MISE moyen $\\pm$ IC 95\\,\\%",
        "tab:mise-gaussienne-n200",
        rows_g
    )

    # Smooth comb
    rows_sc, _ = one_density_tables("smoothcomb", M=M, n=n, G=G, verbose=True)
    write_latex_table(
        outdir / "tab_mise_smoothcomb_n200.tex",
        "Densité \\textbf{smooth comb} ($M=1000$, $n=200$) — MISE moyen $\\pm$ IC 95\\,\\%",
        "tab:mise-smoothcomb-n200",
        rows_sc
    )

    # Affiche un récapitulatif dans la console
    print("\n--- Récapitulatif (MISE moyen ± IC95%) ---")
    for title, rows in [("Bimodale", rows_bi), ("Gaussienne", rows_g), ("Smooth comb", rows_sc)]:
        print(f"\n{title}:")
        for name, mean, ic in rows:
            print(f"  {name:<12s} : {mean:.4f} ± {ic:.5f}")
    print(f"\nFichiers LaTeX écrits dans: {outdir.resolve()}")
