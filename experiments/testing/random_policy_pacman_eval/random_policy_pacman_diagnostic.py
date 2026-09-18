import os
import sys

# ---------------------------------------------------------------------------
# Imports with clear error messages
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
except ImportError:
    sys.exit("[ERROR] 'gymnasium' is not installed. Run: pip install gymnasium")

try:
    import numpy as np
except ImportError:
    sys.exit("[ERROR] 'numpy' is not installed. Run: pip install numpy")

try:
    import pandas as pd
except ImportError:
    sys.exit("[ERROR] 'pandas' is not installed. Run: pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")          # windowless backend
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("[ERROR] 'matplotlib' is not installed. Run: pip install matplotlib")

try:
    import ale_py  # noqa: F401  -- required to register ALE environments
except ImportError:
    sys.exit("[ERROR] 'ale-py' is not installed. Run: pip install ale-py")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_ID        = "ALE/MsPacman-v5"
NUM_EPISODES  = 150
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH      = os.path.join(SCRIPT_DIR, "pacman_random_policy_results.csv")
PNG_PATH      = os.path.join(SCRIPT_DIR, "pacman_random_policy_histogram.png")

# ---------------------------------------------------------------------------
# Create environment (without artificial max_episode_steps)
# ---------------------------------------------------------------------------
print(f"\n[INFO] Creating environment: {ENV_ID}")
print(f"[INFO] obs_type=ram  |  render_mode=None  |  max_episode_steps=NO LIMIT")

try:
    env = gym.make(ENV_ID, obs_type="ram", render_mode=None)
except Exception as e:
    sys.exit(f"[ERROR] Could not create environment '{ENV_ID}': {e}")

# ---------------------------------------------------------------------------
# Initial observation space verification (done once)
# ---------------------------------------------------------------------------
first_obs, _ = env.reset()
print(f"\n[INFO] env_id       : {ENV_ID}")
print(f"[INFO] obs shape    : {first_obs.shape}")
print(f"[INFO] obs dtype    : {first_obs.dtype}")

if first_obs.shape != (128,):
    print(
        f"\n[WARNING] State does NOT have 128 elements. "
        f"Shape obtained: {first_obs.shape}. "
        f"Verify that obs_type='ram' is being respected by the environment."
    )
else:
    print("[INFO] Correct - Shape: (128,) -- 128-element RAM vector.")

print(f"\n[INFO] Starting diagnostic with random policy -- {NUM_EPISODES} episodes...\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = []

for ep in range(1, NUM_EPISODES + 1):

    obs, info = env.reset()
    terminated = False
    truncated  = False
    total_steps  = 0
    total_reward = 0.0

    lives_start = None
    lives_end   = None
    first_step  = True

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_steps  += 1
        total_reward += reward

        current_lives = info.get("lives", None)

        if first_step:
            lives_start = current_lives
            first_step  = False

        lives_end = current_lives   # updated until the last step

    # -----------------------------------------------------------------------
    # Lives behavior verification
    # -----------------------------------------------------------------------
    if lives_end is not None and lives_end != 0:
        print(
            f"WARNING: episode {ep} ended with lives={lives_end} (did not reach 0)"
        )

    results.append({
        "episode_id"  : ep,
        "total_steps" : total_steps,
        "total_reward": total_reward,
        "lives_start" : lives_start,
        "lives_end"   : lives_end,
        "terminated"  : terminated,
        "truncated"   : truncated,
    })

    # Progress every 10 episodes
    if ep % 10 == 0:
        print(f"  [Progress] Episode {ep}/{NUM_EPISODES} | steps={total_steps} | reward={total_reward:.1f}")

env.close()

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv(CSV_PATH, index=False)

# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
steps = df["total_steps"].values.astype(float)
N     = len(steps)

mean   = np.mean(steps)
median = np.median(steps)
std    = np.std(steps)
vmin   = np.min(steps)
vmax   = np.max(steps)
p25    = np.percentile(steps, 25)
p50    = np.percentile(steps, 50)
p75    = np.percentile(steps, 75)
p90    = np.percentile(steps, 90)
p95    = np.percentile(steps, 95)
p99    = np.percentile(steps, 99)

lives_nonzero = int((df["lives_end"].fillna(0) != 0).sum())

print(f"\n=== Pure Random Policy Results - Pac-Man RAM ({N} episodes) ===")
print(f"Mean:               {mean:.2f}")
print(f"Median:             {median:.2f}")
print(f"Std deviation:      {std:.2f}")
print(f"Minimum:            {vmin:.2f}")
print(f"Maximum:            {vmax:.2f}")
print(f"Percentile 25:      {p25:.2f}")
print(f"Percentile 50:      {p50:.2f}")
print(f"Percentile 75:      {p75:.2f}")
print(f"Percentile 90:      {p90:.2f}")
print(f"Percentile 95:      {p95:.2f}")
print(f"Percentile 99:      {p99:.2f}")
print(f"Episodes with lives_end != 0: {lives_nonzero} out of {N}")

# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(steps, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(median, color="green",  linewidth=2, linestyle="--", label=f"Median = {median:.0f}")
ax.axvline(p90,    color="red",    linewidth=2, linestyle="--", label=f"Percentile 90 = {p90:.0f}")

ax.set_title(
    "Episode Duration Distribution - Pac-Man RAM (random policy)",
    fontsize=13, fontweight="bold"
)
ax.set_xlabel("Steps per episode", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(PNG_PATH, dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
print(f"\n[OUTPUT] CSV saved at : {CSV_PATH}")
print(f"[OUTPUT] PNG saved at : {PNG_PATH}")