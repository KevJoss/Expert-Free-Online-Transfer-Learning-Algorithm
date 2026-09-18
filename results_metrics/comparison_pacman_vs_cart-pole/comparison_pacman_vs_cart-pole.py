"""
Seccion 5.3 - Analisis de la seleccion de fuente.
 
Extrae directamente de los logs (sin copiar numeros a mano):
  - que agente fue elegido maestro en cada evento   -> "Teacher Selected: Agent N"
  - la incertidumbre media de los 5 agentes          -> "[UNCERTAINTY] A0=... | A1=..."
 
Y los cruza con la recompensa reciente de cada agente, tomada del CSV.
 
Uso: python comparison_pacman_vs_cart-pole.py
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# carpeta padre (results_metrics/) donde estan los .log y .csv
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))

WINDOW = 100          # episodios previos al evento usados para medir desempeno

RUNS = [                                                                  
    ("CP-C  Cart-Pole  seed 42", PARENT / "seed42_cart-pole.log", PARENT / "seed42_efo_tl_results.csv"),
    ("MP-1  MsPacman   seed 42", PARENT / "seed42_pacman.log", PARENT / "efo_tl_results_pacman_seed42.csv"),
    ("MP-2  MsPacman   seed 7",  PARENT / "seed7_pacman.log",    PARENT / "efo_tl_results_pacman_seed7.csv"),
]


def parse_log(path):
    """Devuelve {episodio: {'teacher': int, 'unc': [5 floats] o None}}."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    events, ep, unc = {}, None, None
    for line in txt.splitlines():
        m = re.search(r"\[TRANSFER EVENT\]\s+Episode\s+(\d+)", line)
        if m:
            ep, unc = int(m.group(1)), None
            continue
        m = re.findall(r"A\d=([0-9.eE+-]+)", line)
        if "[UNCERTAINTY]" in line and m:
            unc = [float(v) for v in m]
            continue
        m = re.search(r"Teacher Selected:\s+Agent\s+(\d+)", line)
        if m and ep is not None:
            events[ep] = {"teacher": int(m.group(1)), "unc": unc}
    return events


print(f"ventana de desempeno = {WINDOW} episodios previos a cada evento\n")
resumen = []

for nombre, log, csv in RUNS:
    ev = parse_log(log)
    p = pd.read_csv(csv).pivot_table(index="Episode", columns="Agent", values="Reward")

    print("=" * 74)
    print(nombre)
    print(f"  eventos con transferencia efectiva: {sorted(ev)}")
    ranks, rhos, coh = [], [], []

    for epi in sorted(ev):
        t = ev[epi]["teacher"]
        u = ev[epi]["unc"]
        rew = p.loc[epi - WINDOW:epi - 1].mean().values
        rank = int(pd.Series(rew).rank(ascending=False)[t])
        ranks.append(rank)

        if u is not None:
            u = np.array(u)
            argmin_ok = int(np.argmin(u)) == t          # coherencia log <-> criterio
            coh.append(argmin_ok)
            rho = stats.spearmanr(u, rew).statistic
            rhos.append(rho)
            print(f"  ep{epi:5d} | teacher=A{t} rank {rank}/5 | rho={rho:+.2f} "
                  f"| U_media={u.mean():.6f} spread={(u.max()/u.min()-1)*100:5.0f}% "
                  f"| argmin==teacher: {argmin_ok}")
        else:
            print(f"  ep{epi:5d} | teacher=A{t} rank {rank}/5 | (sin datos de incertidumbre)")

    linea = f"  --> rango medio del maestro = {np.mean(ranks):.2f}/5   ranks={ranks}"
    if rhos:
        linea += f"\n  --> rho medio = {np.mean(rhos):+.3f}   negativos: {sum(r < 0 for r in rhos)}/{len(rhos)}"
        linea += f"\n  --> coherencia argmin(U) == teacher del log: {sum(coh)}/{len(coh)}"
    print(linea)
    resumen.append((nombre, np.mean(ranks), np.mean(rhos) if rhos else None))

print("\n" + "=" * 74)
print("RESUMEN")
print(f"{'run':28s} {'rango medio':>12s} {'rho medio':>11s}")
for n, r, rh in resumen:
    print(f"{n:28s} {r:11.2f}/5 {('%+.3f' % rh) if rh is not None else '     n/a':>11s}")

# ---- evolucion de la magnitud de la incertidumbre
print("\nMAGNITUD DE LA INCERTIDUMBRE (media de los 5 agentes)")
print(f"{'evento':>8s} {'Cart-Pole':>12s} {'MsPacman s7':>13s}")
cp, mp = parse_log(PARENT / "seed42_cart-pole.log"), parse_log(PARENT / "seed7_pacman.log")
for epi in sorted(cp):
    a = np.mean(cp[epi]["unc"]) if cp[epi]["unc"] else float("nan")
    b = np.mean(mp[epi]["unc"]) if mp.get(epi, {}).get("unc") else float("nan")
    print(f"{epi:8d} {a:12.6f} {b:13.6f}")