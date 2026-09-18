import re, os, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOGS = os.path.abspath(os.path.join(_HERE, '../experiments/results'))


def parse_log(path):
    """Parse timing entries from a log file."""
    text = open(path, encoding='utf-8', errors='replace').read()
    pattern = r'episode=\s*(\d+)\s+elapsed_s=\s*([\d.]+)\s+cum_steps=\s*(\d+)\s+mean_ep_len=\s*([\d.]+)'
    return [(int(ep), float(elapsed), int(steps), float(ep_len))
            for ep, elapsed, steps, ep_len in re.findall(pattern, text)]


LOG_FILES = [
    ('CP-C  Cart-Pole',  os.path.join(_LOGS, 'cartpole', 'execution_logs', 'seed42_cart-pole.log')),
    ('MP-1  Pacman s42', os.path.join(_LOGS, 'pacman',   'execution_logs', 'seed42_pacman.log')),
    ('MP-2  Pacman s7',  os.path.join(_LOGS, 'pacman',   'execution_logs', 'seed7_pacman.log')),
]

for name, log_path in LOG_FILES:
    records = [x for x in parse_log(log_path) if x[2] < 2e7]  # discard cum_steps glitch
    last_ep, total_time, total_steps, last_ep_len = records[-1]

    # Compute seconds-per-step after warm-up (episode > 150)
    sec_per_step = [
        (records[i][1] - records[i-1][1]) / (records[i][2] - records[i-1][2])
        for i in range(1, len(records)) if records[i][0] > 150
    ]

    print(f'{name}')
    print(f'   {total_time:.1f} s = {total_time/3600:.2f} h | {total_steps:,} steps | {total_steps//5:,} per agent')
    print(f'   s/step = {np.mean(sec_per_step):.5f} ± {np.std(sec_per_step):.5f}')
    print(f'   mean_ep_len {records[0][3]:.1f} -> {last_ep_len:.1f}  ({last_ep_len/records[0][3]:.2f}x)')