import pandas as pd, numpy as np
import os
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '../../experiments/results'))

WINDOW = 100

# Uncertainities from the log 
UNC_CP = {600:[0.016538,0.019233,0.016566,0.018764,0.015571],
          800:[0.005546,0.005281,0.005374,0.005080,0.004559],
         1000:[0.002880,0.003376,0.002179,0.003508,0.002018],
         1200:[0.001953,0.001043,0.000827,0.000795,0.001377],
         1400:[0.000012,0.000009,0.000035,0.000009,0.000011],
         1600:[0.000007,0.000011,0.000003,0.000002,0.000009]}

UNC_MP = {600:[0.003398,0.003502,0.003630,0.003562,0.003446],
          800:[0.003478,0.003442,0.005854,0.005586,0.003521],
         1000:[0.003275,0.003955,0.005044,0.004095,0.003315],
         1200:[0.003532,0.003681,0.003661,0.003536,0.003454],
         1400:[0.003662,0.003680,0.003652,0.003581,0.003284],
         1600:[0.003699,0.003604,0.004022,0.004000,0.003352]}

for nombre, csv, UNC in [
        ('CP-C Cart-Pole', os.path.join(_ROOT, 'cartpole', 'seed42_efo_tl_results.csv'),       UNC_CP),
        ('MP-2 Pacman s7', os.path.join(_ROOT, 'pacman', 'seed7', 'efo_tl_results_pacman_seed7.csv'), UNC_MP)]:

    p = pd.read_csv(csv).pivot_table(index='Episode', columns='Agent', values='Reward')
    print(f"\n=== {nombre} ===")
    rhos = []
    for ep, u in UNC.items():
        u   = np.array(u)
        rew = p.loc[ep-WINDOW:ep-1].mean().values
        rho = stats.spearmanr(u, rew).statistic
        rhos.append(rho)
        # Check: el minimo de U debe ser el maestro del log
        print(f"  ep{ep}: rho={rho:+.2f} | argmin(U)=A{np.argmin(u)}")
    print(f"  Average rho = {np.mean(rhos):+.3f} | negative: {sum(r<0 for r in rhos)}/6")