"""
Figura 1 (Seccion 5.2): mediana por bloques de 100 episodios con rango
intercuartilico, para las dos corridas de MsPacman, sobre la banda del
baseline aleatorio medido.

Imprime tambien los valores numericos que la figura representa, para poder
contrastarlos con la tabla del paper.

Uso:  python fig1_learning_curves.py
Requiere: pandas, numpy, matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os

_parent = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
sys.path.insert(0, _parent)

# --------------------------------------------------------------- parametros
F_BASE = os.path.join(_parent, "random_baseline_pacman_results.csv")
F_S42 = os.path.join(_parent, "efo_tl_results_pacman_seed42.csv")
F_S7 = os.path.join(_parent, "efo_tl_results_pacman_seed7.csv")

BLOCK = 100                                    # tamano del bloque, en episodios
TRANSFER_EPISODES = [600, 800, 1000, 1200, 1400, 1600]
YLIM = (120, 560)

C42, C7, CB = "#1f4e79", "#c1666b", "#7a7a7a"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def blocks(df, width=BLOCK):
    """Mediana, Q25 y Q75 por bloque de `width` episodios, sobre los 5 agentes."""
    g = df.groupby(df.Episode // width).Reward
    centres = (g.median().index.values * width) + width / 2
    return (centres,
            g.median().values,
            g.quantile(0.25).values,
            g.quantile(0.75).values)


# ------------------------------------------------------------------- datos
base = pd.read_csv(F_BASE).Reward.values
s42 = pd.read_csv(F_S42)
s7 = pd.read_csv(F_S7)

b_med = np.median(base)
b_q25, b_q75 = np.percentile(base, [25, 75])

# ------------------------------------------------------------------ figura
fig, ax = plt.subplots(figsize=(5.2, 3.0))

ax.axhspan(b_q25, b_q75, color=CB, alpha=0.16, zorder=0)
ax.axhline(b_med, color=CB, ls="--", lw=1.1, zorder=1)

for df, col, lab in ((s42, C42, "Seed 42"), (s7, C7, "Seed 7")):
    x, med, q25, q75 = blocks(df)
    ax.fill_between(x, q25, q75, color=col, alpha=0.13, lw=0, zorder=2)
    ax.plot(x, med, color=col, lw=1.6, zorder=3)

for ep in TRANSFER_EPISODES:
    ax.axvline(ep, color="k", lw=0.7, ls=":", alpha=0.45, zorder=1)
    ax.plot(ep, YLIM[1], marker="v", ms=3.5, color="k", clip_on=False, zorder=5)

ax.set_xlabel("Episode")
ax.set_ylabel("Reward per episode")
ax.set_xlim(0, 1800)
ax.set_ylim(*YLIM)
ax.set_xticks(range(0, 1801, 300))
ax.legend(handles=[
    Line2D([], [], color=C42, lw=1.6, label="Seed 42"),
    Line2D([], [], color=C7, lw=1.6, label="Seed 7"),
    Line2D([], [], color=CB, lw=1.1, ls="--", label="Random baseline"),
    Line2D([], [], color="k", lw=0.7, ls=":", label="Transfer event"),
], loc="upper left", frameon=False, ncol=2, columnspacing=1.2, handlelength=1.6)

fig.savefig("fig1_learning_curves.pdf")
fig.savefig("fig1_learning_curves.png")
print("[guardado] fig1_learning_curves.pdf / .png\n")

# ------------------------------------------------------------ verificacion
print(f"BASELINE   n={len(base)}  Q25={b_q25:.0f}  mediana={b_med:.0f}  Q75={b_q75:.0f}")
print(f"           banda gris de la figura = [{b_q25:.0f}, {b_q75:.0f}]\n")

for df, lab in ((s42, "SEED 42"), (s7, "SEED 7")):
    x, med, q25, q75 = blocks(df)
    dentro = ((med >= b_q25) & (med <= b_q75)).sum()
    solapa = ((q25 <= b_q75) & (q75 >= b_q25)).sum()
    print(f"{lab}  ({len(med)} bloques de {BLOCK} episodios)")
    print(f"  mediana por bloque: min={med.min():.0f}  max={med.max():.0f}")
    print(f"  medianas dentro de la banda del baseline : {dentro}/{len(med)}")
    print(f"  bloques cuya banda solapa con el baseline: {solapa}/{len(med)}")

    fin = df[df.Episode >= 1600].Reward
    print(f"  ultimos 200 episodios (n={len(fin)}): "
          f"Q25={fin.quantile(.25):.0f}  mediana={fin.median():.0f}  "
          f"Q75={fin.quantile(.75):.0f}  media={fin.mean():.2f}")
    print(f"  ultimos 3 bloques -> mediana {med[-3:].astype(int)}\n")