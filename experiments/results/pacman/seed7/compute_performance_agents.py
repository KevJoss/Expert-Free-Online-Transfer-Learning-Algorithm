import pandas as pd, numpy as np
import os

WINDOW = 100

TEACHERS = {
    'CP-C  Cart-Pole': (os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cartpole/seed42_efo_tl_results.csv')),
                        {600:4, 800:4, 1000:4, 1200:3, 1400:1, 1600:3}),
    'MP-1  Pacman s42': (os.path.abspath(os.path.join(os.path.dirname(__file__), '../seed42/efo_tl_results_pacman_seed42.csv')),
                        {600:0, 800:0, 1000:2, 1200:0, 1400:2, 1600:1}),
    'MP-2  Pacman s7': (os.path.join(os.path.dirname(__file__), 'efo_tl_results_pacman_seed7.csv'),
                        {600:0, 800:1, 1000:0, 1200:4, 1400:4, 1600:4}),
}
for nombre, (csv, teachers) in TEACHERS.items():
    p = pd.read_csv(csv).pivot_table(index='Episode', columns='Agent', values='Reward')
    print(f"\n=== {nombre} ===")
    ranks = []
    for ep, t in teachers.items():
        rew  = p.loc[ep-WINDOW:ep-1].mean()          # Average per agent
        rank = int(rew.rank(ascending=False)[t])     # Teacher position
        ranks.append(rank)
        print(f"  ep{ep}: teacher=A{t}  range={rank}/5   "
              f"performance={rew.round(1).values}")
    print(f"  average range = {sum(ranks)}/{len(ranks)} = {np.mean(ranks):.2f}")