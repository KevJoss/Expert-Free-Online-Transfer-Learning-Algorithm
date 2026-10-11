"""
Section 5.3 - Source (Teacher) Selection Analysis.

Parses execution logs directly to extract:
  - Selected teacher agent for each transfer event ("Teacher Selected: Agent N")
  - Average uncertainty values across the 5 concurrent agents ("[UNCERTAINTY] A0=... | A1=...")

Cross-references these selections against empirical agent performance (rewards from CSV)
to evaluate:
  1. Teacher performance ranking (rank 1 to 5, where 1 is best)
  2. Spearman rank correlation (rho) between uncertainty and performance

Usage:
    python comparison_pacman_vs_cart-pole.py
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Parent directory (results_metrics/) containing logs and CSV files
PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))

# Number of preceding episodes used to measure recent performance
WINDOW = 100

RUNS = [
    ("CP-C  Cart-Pole  seed 42", PARENT / "seed42_cart-pole.log", PARENT / "seed42_efo_tl_results.csv"),
    ("MP-1  MsPacman   seed 42", PARENT / "seed42_pacman.log",    PARENT / "efo_tl_results_pacman_seed42.csv"),
    ("MP-2  MsPacman   seed 7",  PARENT / "seed7_pacman.log",     PARENT / "efo_tl_results_pacman_seed7.csv"),
]


def parse_log(path):
    """
    Parses execution log to extract transfer events.
    Returns: dict mapping episode -> {'teacher': int, 'unc': list of 5 floats or None}
    """
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


print(f"Performance window = {WINDOW} episodes prior to each transfer event\n")
resumen = []

for name, log, csv in RUNS:
    ev = parse_log(log)
    # Pivot rewards: Index = Episode, Columns = Agent 0..4
    p = pd.read_csv(csv).pivot_table(index="Episode", columns="Agent", values="Reward")

    print("=" * 74)
    print(name)
    print(f"  Transfer events detected: {sorted(ev)}")
    ranks, rhos, coh = [], [], []

    for epi in sorted(ev):
        t = ev[epi]["teacher"]
        u = ev[epi]["unc"]
        # Mean reward per agent over the preceding WINDOW episodes
        rew = p.loc[epi - WINDOW:epi - 1].mean().values
        # Rank agents from 1 (highest reward) to 5 (lowest reward)
        rank = int(pd.Series(rew).rank(ascending=False)[t])
        ranks.append(rank)

        if u is not None:
            u = np.array(u)
            # Verify if argmin(uncertainty) matches the selected teacher in logs
            argmin_ok = int(np.argmin(u)) == t
            coh.append(argmin_ok)
            # Spearman correlation between uncertainty and performance (expected: negative)
            rho = stats.spearmanr(u, rew).statistic
            rhos.append(rho)
            print(f"  ep{epi:5d} | teacher=A{t} rank {rank}/5 | rho={rho:+.2f} "
                  f"| mean_U={u.mean():.6f} spread={(u.max()/u.min()-1)*100:5.0f}% "
                  f"| argmin==teacher: {argmin_ok}")
        else:
            print(f"  ep{epi:5d} | teacher=A{t} rank {rank}/5 | (no uncertainty data in log)")

    line = f"  --> Average teacher rank = {np.mean(ranks):.2f}/5   ranks={ranks}"
    if rhos:
        line += f"\n  --> Mean Spearman rho = {np.mean(rhos):+.3f}   negative rhos: {sum(r < 0 for r in rhos)}/{len(rhos)}"
        line += f"\n  --> Consistency argmin(U) == logged teacher: {sum(coh)}/{len(coh)}"
    print(line)
    resumen.append((name, np.mean(ranks), np.mean(rhos) if rhos else None))

print("\n" + "=" * 74)
print("SUMMARY")
print(f"{'Run':28s} {'Avg Rank':>12s} {'Mean Rho':>11s}")
for n, r, rh in resumen:
    print(f"{n:28s} {r:11.2f}/5 {('%+.3f' % rh) if rh is not None else '     n/a':>11s}")

# ---- Evolution of uncertainty magnitude across episodes ----
print("\nUNCERTAINTY MAGNITUDE EVOLUTION (Mean across 5 agents)")
print(f"{'Episode':>8s} {'Cart-Pole':>12s} {'MsPacman s7':>13s}")
cp, mp = parse_log(PARENT / "seed42_cart-pole.log"), parse_log(PARENT / "seed7_pacman.log")
for epi in sorted(cp):
    a = np.mean(cp[epi]["unc"]) if cp[epi]["unc"] else float("nan")
    b = np.mean(mp[epi]["unc"]) if mp.get(epi, {}).get("unc") else float("nan")
    print(f"{epi:8d} {a:12.6f} {b:13.6f}")