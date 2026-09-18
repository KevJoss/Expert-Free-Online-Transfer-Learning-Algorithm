import pandas as pd, numpy as np
# R takes only the reward values
_parent = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

R   = pd.read_csv(os.path.join(_parent, 'random_baseline_pacman_results.csv')).Reward.values
s42 = pd.read_csv(os.path.join(_parent, 'efo_tl_results_pacman_seed42.csv'))
s7  = pd.read_csv(os.path.join(_parent, 'efo_tl_results_pacman_seed7.csv'))

# Take the reward values from seed 42 and seed 7
w42 = s42[s42.Episode < 100].Reward.values
w7  = s7 [s7.Episode  < 100].Reward.values


# print(s42[s42.Episode < 100].shape) # Here I can see that have a total of 500 records on the range of less 100 episodes

# Compute with numpy the total number of the registers for untrained ranges (for ech of one compute the mean and the median)
for nombre, x in [('baseline', R), ('warmup42', w42), ('warmup7', w7)]:
    print(nombre, len(x), round(x.mean(), 2), np.median(x))

# Compute the Cliff Delta -> Responds to: Of I take two random values from a and b. Which probably is that a migth be greather than b?
def cliff(a, b):
    # Sort a and b
    a, b = np.sort(a), np.sort(b)
    # Count how many times (a > b)
    gt = np.searchsorted(b, a, 'left').sum()
    # Count how many victories have or equal (a >= b)
    ge = np.searchsorted(b, a, 'right').sum()
    
    return (gt - (len(a)*len(b) - ge)) / (len(a)*len(b))

print('delta 42:', round(cliff(w42, R), 4))   # +0.0572
print('delta  7:', round(cliff(w7,  R), 4))   # +0.0099



rng = np.random.default_rng(1)  # Generador aleatorio con semilla fija
def ci_dif(a, b, B=20000):
    # Paso 1: Simular 20,000 "experimentos"
    s = [
        rng.choice(a, len(a), True).mean()  # Media de muestra bootstrap de a
        -
        rng.choice(b, len(b), True).mean()  # Media de muestra bootstrap de b
        for _ in range(B)                    # Repetir B=20,000 veces
    ]
    # Paso 2: Tomar los extremos del 95% central
    return np.percentile(s, [2.5, 97.5])


print('dif 42:', round(w42.mean()-R.mean(), 2), ci_dif(w42, R).round(1))
print('dif  7:', round(w7.mean() -R.mean(), 2), ci_dif(w7,  R).round(1))