"""
random_baseline_pacman.py
=========================
Random-action baseline for ALE/MsPacman-v5.

Instead of training a Dueling-DQN policy, each "agent" selects actions
uniformly at random via ``env.action_space.sample()``.  This provides a
lower-bound reference to compare against the EFo-TL algorithm.

Configuration is hardcoded (no JSON dependency) to match the values used
in ``pacman_params.json``.

Outputs
-------
- ``results/pacman/random_baseline_results.csv``
    CSV with columns ``Simulation,Agent,Episode,Reward`` — identical
    format to ``efo_tl_results.csv`` so existing analysis scripts work
    unchanged.
- ``random_baseline_pacman.log``
    Environment configuration, per-block progress, and final statistics
    (mean, median, IQR, percentiles).
"""

import os
import sys
import time
import csv
import numpy as np
import gymnasium as gym
import ale_py

# ---------------------------------------------------------------------------
# Hardcoded Configuration (mirrors pacman_params.json)
# ---------------------------------------------------------------------------

ENV_NAME          = "ALE/MsPacman-v5"
NUM_SIMULATIONS   = 1
NUM_EPISODES      = 500
NUM_AGENTS        = 5
SEED              = 42

# Paths (relative to this script's directory)
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(_SCRIPT_DIR, "results", "pacman")
RESULTS_CSV  = os.path.join(RESULTS_DIR, "random_baseline_results.csv")
LOG_FILE     = os.path.join(_SCRIPT_DIR, "random_baseline_pacman.log")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _init_log():
    """Create (or overwrite) the log file with a header."""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"[LOG STARTED] random_baseline_pacman.py — "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")


def _log(text: str):
    """Append a line to the log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def _log_and_print(text: str):
    """Print to stdout AND append to the log file."""
    print(text)
    _log(text)

# ---------------------------------------------------------------------------
# Environment introspection
# ---------------------------------------------------------------------------

def log_env_config(env: gym.Env):
    """
    Inspect and log every observable attribute of the ALE environment
    (frameskip, sticky-action probability, repeat_action_probability, etc.).
    """
    header = "=" * 70
    _log_and_print(header)
    _log_and_print("  ENVIRONMENT CONFIGURATION")
    _log_and_print(header)

    # Basic info
    _log_and_print(f"  env.spec.id               : {env.spec.id if env.spec else 'N/A'}")
    _log_and_print(f"  env.spec.max_episode_steps : {env.spec.max_episode_steps if env.spec else 'N/A'}")
    _log_and_print(f"  observation_space          : {env.observation_space}")
    _log_and_print(f"  action_space               : {env.action_space}")

    # Walk through wrapper stack to find ALE-specific settings
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env

    # AtariEnv / ALE settings (may or may not exist depending on version)
    for attr in [
        "frameskip", "_frameskip",
        "repeat_action_probability",
        "full_action_space",
        "obs_type",
        "render_mode",
    ]:
        val = getattr(inner, attr, "N/A")
        _log_and_print(f"  {attr:<28s}: {val}")

    # ALE interface object (ale.lib / ale_py)
    ale = getattr(inner, "ale", None)
    if ale is not None:
        for ale_attr in [
            "getFloat:repeat_action_probability",
            "getInt:frame_skip",
            "getBool:color_averaging",
        ]:
            method, key = ale_attr.split(":")
            try:
                val = getattr(ale, method)(key)
                _log_and_print(f"  ale.{method}('{key}') : {val}")
            except Exception:
                pass

    _log_and_print(header + "\n")

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_and_log_statistics(all_rewards: np.ndarray):
    """
    Compute and log summary statistics over ALL rewards collected.

    Parameters
    ----------
    all_rewards : np.ndarray
        1-D array with every single episode reward across all agents.
    """
    mean   = np.mean(all_rewards)
    median = np.median(all_rewards)
    q25    = np.percentile(all_rewards, 25)
    q75    = np.percentile(all_rewards, 75)
    iqr    = q75 - q25
    p5     = np.percentile(all_rewards, 5)
    p10    = np.percentile(all_rewards, 10)
    p90    = np.percentile(all_rewards, 90)
    p95    = np.percentile(all_rewards, 95)
    std    = np.std(all_rewards)
    min_r  = np.min(all_rewards)
    max_r  = np.max(all_rewards)

    header = "=" * 70
    _log_and_print("")
    _log_and_print(header)
    _log_and_print("  FINAL STATISTICS  —  Random Baseline (all agents, all episodes)")
    _log_and_print(header)
    _log_and_print(f"  Total episodes        : {len(all_rewards)}")
    _log_and_print(f"  Mean reward           : {mean:.4f}")
    _log_and_print(f"  Median reward         : {median:.4f}")
    _log_and_print(f"  Std deviation         : {std:.4f}")
    _log_and_print(f"  Min / Max reward      : {min_r:.1f} / {max_r:.1f}")
    _log_and_print(f"  Q25 (25th percentile) : {q25:.4f}")
    _log_and_print(f"  Q75 (75th percentile) : {q75:.4f}")
    _log_and_print(f"  IQR (Q75 − Q25)      : {iqr:.4f}")
    _log_and_print(f"  P5  (5th percentile)  : {p5:.4f}")
    _log_and_print(f"  P10 (10th percentile) : {p10:.4f}")
    _log_and_print(f"  P90 (90th percentile) : {p90:.4f}")
    _log_and_print(f"  P95 (95th percentile) : {p95:.4f}")
    _log_and_print(header)

    # Per-agent breakdown
    _log_and_print("")
    _log_and_print("  PER-AGENT BREAKDOWN")
    _log_and_print("-" * 70)
    # all_rewards is ordered: agent0_ep0, agent1_ep0, ..., agent4_ep0, agent0_ep1, ...
    # Reshape to (NUM_EPISODES, NUM_AGENTS) then transpose → (NUM_AGENTS, NUM_EPISODES)
    per_agent = all_rewards.reshape(NUM_EPISODES, NUM_AGENTS).T
    for agent_id in range(NUM_AGENTS):
        a_rewards = per_agent[agent_id]
        _log_and_print(
            f"  Agent {agent_id}:  mean={np.mean(a_rewards):.2f}  "
            f"median={np.median(a_rewards):.2f}  "
            f"std={np.std(a_rewards):.2f}  "
            f"min={np.min(a_rewards):.0f}  max={np.max(a_rewards):.0f}"
        )
    _log_and_print("-" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_random_simulation(sim_id: int):
    """
    Run one simulation: NUM_AGENTS "agents" each play NUM_EPISODES episodes
    using purely random actions.

    Returns
    -------
    all_rows : list[dict]
        List of row dicts ready for CSV writing.
    all_rewards : list[float]
        Flat list of rewards in the same order as all_rows.
    """

    np.random.seed(SEED + sim_id)

    _log_and_print(f"\n{'='*70}")
    _log_and_print(f"  RUNNING Random Baseline (MsPacman-v5)")
    _log_and_print(f"  Simulation: {sim_id} | Agents: {NUM_AGENTS} | Episodes: {NUM_EPISODES}")
    _log_and_print(f"{'='*70}\n")

    # Create one env per agent
    envs = []
    gym.register_envs(ale_py)
    for i in range(NUM_AGENTS):
        env = gym.make(ENV_NAME, obs_type="ram", render_mode=None)
        envs.append(env)

    # Log environment config once (from the first env)
    log_env_config(envs[0])

    all_rows    = []
    all_rewards = []
    start_time  = time.time()

    for episode in range(NUM_EPISODES):
        for agent_id, env in enumerate(envs):
            state, info = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward

            all_rows.append({
                "Simulation": sim_id,
                "Agent": agent_id,
                "Episode": episode,
                "Reward": ep_reward,
            })
            all_rewards.append(ep_reward)

        # Progress report every 50 episodes
        if (episode + 1) % 50 == 0:
            elapsed = time.time() - start_time
            recent_rewards = all_rewards[-(NUM_AGENTS * 50):]
            avg = np.mean(recent_rewards)
            _log_and_print(
                f"  Episode {episode + 1:>4d}/{NUM_EPISODES} | "
                f"Elapsed: {elapsed:>7.1f}s | "
                f"Last-50-ep Avg Reward: {avg:.2f}"
            )

    # Clean up
    for env in envs:
        env.close()

    total_elapsed = time.time() - start_time
    _log_and_print(f"\n  [DONE] Simulation {sim_id} finished in {total_elapsed:.1f}s")

    return all_rows, all_rewards


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _init_log()

    _log_and_print("=" * 70)
    _log_and_print("  RANDOM BASELINE EXPERIMENT — MsPacman-v5")
    _log_and_print(f"  Simulations: {NUM_SIMULATIONS} | Agents: {NUM_AGENTS} | "
                   f"Episodes/sim: {NUM_EPISODES} | Seed: {SEED}")
    _log_and_print("=" * 70)

    cumulative_rows    = []
    cumulative_rewards = []

    for sim_id in range(NUM_SIMULATIONS):
        rows, rewards = run_random_simulation(sim_id)
        cumulative_rows.extend(rows)
        cumulative_rewards.extend(rewards)

    # ---- Save CSV --------------------------------------------------------
    fieldnames = ["Simulation", "Agent", "Episode", "Reward"]
    with open(RESULTS_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cumulative_rows)
    _log_and_print(f"\n  [CSV SAVED] {RESULTS_CSV}")

    # ---- Final statistics ------------------------------------------------
    all_rewards_arr = np.array(cumulative_rewards)
    compute_and_log_statistics(all_rewards_arr)

    _log_and_print(f"\n  [LOG SAVED] {LOG_FILE}")
    _log_and_print("  Experiment complete.\n")
