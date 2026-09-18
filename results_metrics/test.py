from scipy import stats
import pandas as pd

for f in ['efo_tl_results_pacman_seed42.csv', 'efo_tl_results_pacman_seed7.csv']:
    p = pd.read_csv(f).pivot_table(index='Episode', columns='Agent', values='Reward')
    rhos = [stats.spearmanr(p.index, p[a].values).statistic for a in range(5)]
    print(f, [f'A{a}:{r:+.3f}' for a, r in enumerate(rhos)])