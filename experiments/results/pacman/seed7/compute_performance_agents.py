"""
Evaluation script to compute the performance ranking (1 to 5) of the selected teacher agent
across transfer events in Cart-Pole and MsPacman experiments.
"""
import os
import numpy as np
import pandas as pd

# Window of recent episodes prior to the transfer event used to compute average reward
WINDOW = 100

# Mapping of experiment runs: (CSV path, {transfer_episode: selected_teacher_id})
TEACHERS = {
    'CP-C  Cart-Pole': (os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cartpole/seed42_efo_tl_results.csv')),
                        {600: 4, 800: 4, 1000: 4, 1200: 3, 1400: 1, 1600: 3}),
    'MP-1  Pacman s42': (os.path.abspath(os.path.join(os.path.dirname(__file__), '../seed42/efo_tl_results_pacman_seed42.csv')),
                        {600: 0, 800: 0, 1000: 2, 1200: 0, 1400: 2, 1600: 1}),
    'MP-2  Pacman s7': (os.path.join(os.path.dirname(__file__), 'efo_tl_results_pacman_seed7.csv'),
                        {600: 0, 800: 1, 1000: 0, 1200: 4, 1400: 4, 1600: 4}),
}

for name, (csv, teachers) in TEACHERS.items():
    # Load rewards per episode across all 5 agents (columns: Agent 0..4)
    p = pd.read_csv(csv).pivot_table(index='Episode', columns='Agent', values='Reward')
    print(f"\n=== {name} ===")
    ranks = []

    for ep, t in teachers.items():
        # Compute mean reward for each agent in the [ep - WINDOW, ep - 1] window
        rew = p.loc[ep - WINDOW:ep - 1].mean()
        # Rank agents from 1 (best reward) to 5 (worst reward) and retrieve teacher's rank
        rank = int(rew.rank(ascending=False)[t])
        ranks.append(rank)
        print(f"  ep{ep}: teacher=A{t}  rank={rank}/5   "
              f"performance={rew.round(1).values}")

    # Compute and display mean rank of the chosen teachers across all transfer events
    print(f"  average rank = {sum(ranks)}/{len(ranks)} = {np.mean(ranks):.2f}/5")