"""

Usage
-----
    # List available models for a scenario:
    python evaluate_saved_agents.py --scenario cartpole --list
    python evaluate_saved_agents.py --scenario pacman --list

    # Watch CartPole – sim 0, agent 0 (3 episodes by default):
    python evaluate_saved_agents.py --scenario cartpole --sim 0 --agent 0

    # Watch Pacman – sim 0, agent 2, 5 episodes:
    python evaluate_saved_agents.py --scenario pacman --sim 0 --agent 2 --num_episodes 5

    # Run without visual window (headless):
    python evaluate_saved_agents.py --scenario cartpole --sim 0 --agent 0 --no-render

    # Override config or models directory manually:
    python evaluate_saved_agents.py --scenario cartpole --config configs/cart-pole_params.json
    python evaluate_saved_agents.py --scenario cartpole --models_dir results/cartpole/models/
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import torch
import gymnasium as gym
import ale_py

current_directory = os.path.dirname(__file__)
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(os.path.join(parent_directory, "agents"))

# dueling_network is used to rebuild the architecture before loading weights
from dueling_network import dueling_network

gym.register_envs(ale_py)

# ---------------------------------------------------------------------------
# Scenario → default config and models_dir mapping
# ---------------------------------------------------------------------------
SCENARIO_DEFAULTS = {
    "cartpole": {
        "config": os.path.join("configs", "cart-pole_params.json"),
        "models_dir": os.path.join("results", "cartpole", "models"),
    },
    "pacman": {
        "config": os.path.join("configs", "pacman_params.json"),
        "models_dir": os.path.join("results", "pacman", "models"),
    },
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Evaluate a saved EFo-TL agent visually.\n"
        "Choose the scenario with --scenario {cartpole,pacman}."
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--scenario",
    type=str,
    choices=["cartpole", "pacman"],
    required=True,
    help="Environment scenario to evaluate: 'cartpole' or 'pacman'.",
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help=(
        "Path to JSON configuration file. "
        "If omitted, the default config for the chosen scenario is used:\n"
        "  cartpole -> configs/cart-pole_params.json\n"
        "  pacman   -> configs/pacman_params.json"
    ),
)
parser.add_argument(
    "--sim",
    type=int,
    default=0,
    help="Simulation ID of the model to load (default: 0).",
)
parser.add_argument(
    "--agent",
    type=int,
    default=0,
    help="Agent ID within the simulation (default: 0).",
)
parser.add_argument(
    "--models_dir",
    type=str,
    default=None,
    help=(
        "Root directory containing saved models. "
        "Overrides the path from config and scenario defaults."
    ),
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=3,
    help="Number of full game episodes to play (default: 3).",
)
parser.add_argument(
    "--no-render",
    action="store_true",
    help="Disable visual rendering (run headlessly).",
)
parser.add_argument(
    "--list",
    action="store_true",
    help="Print all available saved model files for the chosen scenario and exit.",
)
args = parser.parse_args()

scenario = args.scenario.lower()
scenario_defaults = SCENARIO_DEFAULTS[scenario]

# ---------------------------------------------------------------------------
# Determine Configuration File
# ---------------------------------------------------------------------------
config_file = args.config or scenario_defaults["config"]

if not os.path.exists(config_file):
    print(f"[ERROR] Configuration file not found: {config_file}")
    print(f"  Expected location: {os.path.abspath(config_file)}")
    sys.exit(1)

with open(config_file, "r") as f:
    parameters = json.load(f)

dqn_params = parameters["dqn_agent"]

# Priority: CLI flag > config file value > scenario default
MODELS_DIR = (
    args.models_dir
    or parameters["experiment"].get("models_dir", None)
    or scenario_defaults["models_dir"]
)

print(f"\n  Scenario    : {scenario.upper()}")
print(f"  Config      : {config_file}")
print(f"  Models dir  : {MODELS_DIR}")

# ---------------------------------------------------------------------------
# List available models and exit if --list flag is set
# ---------------------------------------------------------------------------
if args.list:
    pattern = os.path.join(MODELS_DIR, "**", "*.pth")
    models = sorted(glob.glob(pattern, recursive=True))
    if not models:
        print(f"\nNo saved models found under: {MODELS_DIR}")
        print(f"  Absolute path checked: {os.path.abspath(MODELS_DIR)}")
    else:
        print(f"\nFound {len(models)} saved model(s) in '{MODELS_DIR}':\n")
        for m in models:
            print(f"  {m}")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Locate the model file
# ---------------------------------------------------------------------------
model_path = os.path.join(MODELS_DIR, f"sim_{args.sim}", f"agent_{args.agent}.pth")

if not os.path.exists(model_path):
    print(f"\n[ERROR] Model file not found: {model_path}")
    print(f"  Absolute path checked: {os.path.abspath(model_path)}")
    print(f"  Run with '--list' to see all available models for this scenario.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Initialize Environment dynamically
# ---------------------------------------------------------------------------
ENV_NAME = parameters["training"].get("env_name", "ALE/MsPacman-v5")
MAX_EPISODE_STEPS = parameters["training"].get("max_episode_steps", None)
render_mode = None if args.no_render else "human"

kwargs = {}
if MAX_EPISODE_STEPS is not None:
    kwargs["max_episode_steps"] = MAX_EPISODE_STEPS

if "CartPole" in ENV_NAME:
    env = gym.make(ENV_NAME, render_mode=render_mode, **kwargs)
    norm_factor = 1.0
else:
    env = gym.make(ENV_NAME, obs_type="ram", render_mode=render_mode, **kwargs)
    norm_factor = 255.0

state_dim = (
    env.observation_space.shape[0]
    if len(env.observation_space.shape) > 0
    else env.observation_space.n
)
action_dim = env.action_space.n

print(f"\n{'='*65}")
print(f"  Evaluating  : {scenario.upper()} — Simulation {args.sim} | Agent {args.agent}")
print(f"  Environment : {ENV_NAME}")
print(f"  State dim   : {state_dim}  |  Action dim : {action_dim}  |  Hidden: {dqn_params['hidden_dim']}")
print(f"  Model file  : {model_path}")
print(f"  Episodes    : {args.num_episodes}")
print(f"  Render      : {'disabled (headless)' if args.no_render else 'enabled'}")
print(f"{'='*65}\n")

# ---------------------------------------------------------------------------
# Rebuild the dueling_network architecture and load saved weights
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = dueling_network(
    state_dim=state_dim,
    action_dim=action_dim,
    n_latent_var=dqn_params["hidden_dim"],
).to(device)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()  # Switch to evaluation / inference mode

print(f"Model loaded successfully from: {model_path}")

# ---------------------------------------------------------------------------
# Run evaluation episodes
# ---------------------------------------------------------------------------
total_rewards = []

for episode in range(args.num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0
    step_count = 0

    print(f"\nEpisode {episode + 1}/{args.num_episodes} starting...")

    while not done:
        # Select the best action greedily (no exploration)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state / norm_factor).unsqueeze(0).to(device)
            action = model.get_action(state_tensor)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        state = next_state
        episode_reward += reward
        step_count += 1

    total_rewards.append(episode_reward)
    print(f"  Episode {episode + 1} finished | Steps: {step_count} | Score: {episode_reward:.1f}")

env.close()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"  Evaluation Summary — {scenario.upper()} | Sim {args.sim} | Agent {args.agent} [{ENV_NAME}]")
print(f"{'='*65}")
print(f"  Episodes played  : {args.num_episodes}")
print(f"  Mean score       : {np.mean(total_rewards):.1f}")
print(f"  Best score       : {np.max(total_rewards):.1f}")
print(f"  Worst score      : {np.min(total_rewards):.1f}")
print(f"  All scores       : {[round(float(r), 1) for r in total_rewards]}")
