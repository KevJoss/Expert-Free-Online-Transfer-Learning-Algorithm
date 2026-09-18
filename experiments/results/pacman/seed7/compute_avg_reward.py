"""
Compute the average reward for each agent in the last 100 episodes.
Source: efo_tl_results_seed7.csv
"""

import pandas as pd
import os

# Path to the CSV file (same directory as this script)
csv_path = os.path.join(os.path.dirname(__file__), "efo_tl_results_seed7.csv")

# Read data
df = pd.read_csv(csv_path)

# Filter episodes 0 to 99
df_filtered = df[df["Episode"].between(0, 99)]

# Compute average reward per agent
avg_reward = df_filtered.groupby("Agent")["Reward"].mean()

# Show results
print("=" * 45)
print("Average reward (episodes 0-99)")
print("=" * 45)
for agent, reward in avg_reward.items():
    print(f"  Agent {agent}: {reward:.2f}")
print("-" * 45)
print(f"  Global Average: {avg_reward.mean():.2f}")
print("=" * 45)
