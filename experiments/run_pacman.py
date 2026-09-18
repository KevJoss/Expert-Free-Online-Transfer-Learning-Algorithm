import os
import sys
import json
import time
import csv
import io
import re
import numpy as np
import torch
import gymnasium as gym

# Import the libraries from de project
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.join(current_directory, '..')
agents_directory = os.path.join(parent_directory, 'agents')
sys.path.append(parent_directory)
sys.path.append(agents_directory)

from summary_parameters import print_experiment_summary
from dueling_network import agent as AGENT, set_seed
from SARSrnd import UE as UncertaintyEstimator
from transferBuffer import buffer as TransferBuffer
from transferCoreEngine import select_teacher, transfer_higher_delta_confidence

# ---------------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------------

# Load the JSON configuration file "configs/pacman_params.json"
CONFIG_FILE = "configs/pacman_params.json"

with open(CONFIG_FILE, "r") as f:
    parameters = json.load(f)


ENV_NAME           = parameters["training"].get("env_name")
NUM_SIMULATIONS = parameters['experiment']['num_simulations']
NUM_EPISODES       = parameters["training"]["num_episodes"]
NUM_AGENTS         = parameters["training"]["num_agents"]
MAX_EPISODE_STEPS  = parameters["training"].get("max_episode_steps")
BUFFER_MEMORY_SIZE = parameters["training"].get("buffer_memory_size")


dqn_params         = parameters["dqn_agent"]
WARMUP_EPISODES    = dqn_params["warmup_episodes"]
TRAIN_FREQUENCY    = dqn_params.get("train_frequency")

transfer_params       = parameters["transfer"]
TRANSFER_FREQUENCY    = transfer_params["transfer_frequency"]
SOURCE_SELECTION_MODE = transfer_params["source_selection_mode"]
EP_START_SHARING      = transfer_params["ep_start_sharing"]
EVALUATION_RANGE      = transfer_params["evaluation_range"]
BATCH_TRANSFER_SIZE   = transfer_params["batch_transfer_size"]
RND_MINIBATCH_SIZE    = transfer_params.get("rnd_minibatch_size")
RESULTS_CSV = parameters["experiment"]["results_csv"]
MODELS_DIR  = parameters["experiment"]["models_dir"]

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging infrastructure
# ---------------------------------------------------------------------------

# Log file lives next to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE    = os.path.join(_SCRIPT_DIR, "exploration_pacman.log")

def _log(text: str):
    """Append *text* to the log file (creates file on first call)."""
    with open(LOG_FILE, "a", encoding="utf-8") as _f:
        _f.write(text + "\n")


class _TargetNetworkCounter:
    """
    Wraps sys.stdout so that every line containing 'Updating target network.'
    is silently counted.  All other lines are forwarded to the real stdout.
    When flush_and_log() is called the accumulated count is written both to
    stdout and to the log file (e.g.  'Updating target network. (x17)').
    """
    _PATTERN = re.compile(r"Updating target network\.")

    def __init__(self, real_stdout):
        self._real   = real_stdout
        self._count  = 0
        self._buffer = ""

    # ---- stream interface -------------------------------------------------
    def write(self, text):
        self._buffer += text
        # Process complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._process_line(line)

    def _process_line(self, line):
        if self._PATTERN.search(line):
            self._count += 1          # count silently, do NOT print yet
        else:
            self._real.write(line + "\n")
            _log(line)               # mirror every visible line to the log

    def flush(self):
        self._real.flush()

    # ---- public helpers ---------------------------------------------------
    def flush_pending_text(self):
        """Flush any buffered partial line (called before a stats print)."""
        if self._buffer:
            self._process_line(self._buffer)
            self._buffer = ""

    def flush_and_log(self):
        """
        If target-network updates were counted since the last call,
        emit a single collapsed line to stdout AND to the log file.
        Resets the counter afterwards.
        """
        self.flush_pending_text()
        if self._count > 0:
            msg = f"Updating target network. (x{self._count})"
            self._real.write(msg + "\n")
            _log(msg)
            self._count = 0

    @property
    def count(self):
        return self._count


# Install the interceptor once at module level
_stdout_interceptor = _TargetNetworkCounter(sys.stdout)
sys.stdout = _stdout_interceptor

# Initialise the log file NOW (fresh start) so everything printed from this
# point forward — including the experiment summary and DDQN init messages —
# is captured before run_pacman_simulation() is even called.
with open(LOG_FILE, "w", encoding="utf-8") as _f:
    _f.write(f"[LOG STARTED] exploration_pacman.py — {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

# ---------------------------------------------------------------------------
# One simulation Function
# ---------------------------------------------------------------------------


def run_pacman_simulation(sim_id, seed, models_dir, param_file):
    """
    Runs one complete EFo-TL training session for pacman environment.
    """

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"  RUNNING EF-OnTL algorithm (MsPacman-v5)")
    print(f"  Device: {device} | Agents: {NUM_AGENTS} | Episodes: {NUM_EPISODES}")
    print(f"{'='*70}\n")


    def make_env():
        return gym.make(ENV_NAME, obs_type="ram", render_mode=None)

    sample_env = make_env()
    state_dim = sample_env.observation_space.shape[0]
    action_dim = sample_env.action_space.n
    norm_factor = 255.0 
    sample_env.close()

    # Parse hidden_dim parameter for author's agent
    h_dim = dqn_params["hidden_dim"]
    n_latent_var = h_dim if isinstance(h_dim, list) else [h_dim, h_dim]


    # -----------------------------------------------------------------------
    # Initialization of the agents
    # -----------------------------------------------------------------------
    agents_data = []
    
    for i in range(NUM_AGENTS):
        agent_name = f"author_sim{sim_id}_agent_{i}"
        
        # Agent instance
        agent = AGENT(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=dqn_params["learning_rate"],
            betas=tuple(dqn_params["betas"]),
            gamma=dqn_params["gamma"],
            memory_size=dqn_params["replay_buffer_size"],
            batch_size=dqn_params["batch_size"],
            name=agent_name,
            n_latent_var=n_latent_var,
            update_iter=dqn_params["target_update_interval"],
            max_episode=NUM_EPISODES,
            exploration_mode=dqn_params["exploration_mode"],
            device=device,
            TEMPERATURE=dqn_params["temperature"]
        )

        # RND Uncertainty Estimator instance 
        uncertainty_estimator = UncertaintyEstimator(
            n_s=state_dim,
            n_a=1,
            rew_dim=1,
            lr=1e-3,
            id=f"author_ue_{i}",
            s_encode_size=1024,
            n_latent_var=[],
            device=device,
            minibatch_size=RND_MINIBATCH_SIZE
        )

        # Transfer Buffer instance
        transfer_buffer = TransferBuffer(
            memory_size=BUFFER_MEMORY_SIZE,
            id=f"author_buf_{i}",
            with_n_step_return=False
        )


        env_i = make_env()
        env_i.reset(seed=seed * 1000 + i)      # UNIQUE seed
        env_i.action_space.seed(seed * 1000 + i)

        agents_data.append({
            "id": i,
            "env": env_i,
            "agent": agent,
            "ue": uncertainty_estimator,
            "buffer": transfer_buffer,
            "rewards_history": [],
            "total_steps": 0
        })

    # Main Training Loop
    all_rows = []
    start_time = time.time()
    train_calls = 0

    # --- Logging / stats accumulators ------------------------------------
    cum_steps_all   = 0          # total env steps across ALL agents (cumulative)
    # Per-block (every 50 eps) accumulators — reset each block
    block_ep_lens   = []         # list of ep lengths (one entry per agent*ep)
    block_truncated = 0          # episodes that ended by max_time_steps

    # Append a simulation-start separator (log already has the startup output)
    _log(f"\n{'='*70}")
    _log(f"[SIM {sim_id} START] {time.strftime('%Y-%m-%d %H:%M:%S')} "
         f"| ENV: {ENV_NAME} | agents: {NUM_AGENTS} | episodes: {NUM_EPISODES}")
    _log("=" * 70 + "\n")

    for episode in range(NUM_EPISODES):
        for data in agents_data:
            env = data["env"]
            agent = data["agent"]
            uncertainty_estimator = data["ue"]
            transfer_buffer = data["buffer"]

            state, info = env.reset()
            state = state / norm_factor
            done = False
            ep_reward = 0
            ep_steps  = 0          # steps in this episode (for ep-length tracking)
            ep_truncated = False   # did this episode end by time limit?

            while not done:
                # Step 1: Action selection using get_action method
                action, entropy = agent.get_action(state, training=True)

                # Step 2: Environment step
                next_state, reward, terminated, truncated, info = env.step(action)
                next_state = next_state / norm_factor
                done = terminated or truncated
                data["total_steps"] += 1
                ep_steps += 1

                # Detect truncation by time limit
                if truncated and not terminated:
                    ep_truncated = True

                # Step 3: Store the observation  in agent's local Prioritized Experience Replay buffer
                agent.observe(state, action, reward, next_state, done)

                # Step 4: Calculate uncertainty & store in transfer buffer
                uncertainty = uncertainty_estimator.get_uncertainty(state, action, reward, next_state)
                transfer_buffer.push(state, action, reward, next_state, done, uncertainty)

                # Step 5: Learn RND network for uncertainty estimator
                uncertainty_estimator.learn(state, action, reward, next_state)

                # Step 6: Policy update
                if (episode >= WARMUP_EPISODES and 
                    data["total_steps"] % TRAIN_FREQUENCY == 0 and 
                    agent.replay_buffer.size() >= dqn_params["batch_size"]):
                    agent.train()
                    train_calls += 1

                state = next_state
                ep_reward += reward

            # --- Update block accumulators (after each agent episode) ----
            cum_steps_all   += ep_steps
            block_ep_lens.append(ep_steps)
            if ep_truncated:
                block_truncated += 1

            data["rewards_history"].append(ep_reward)
            agent.increment_episode()

            all_rows.append({
                "Simulation": sim_id,
                "Agent": data["id"],
                "Episode": episode,
                "Reward": ep_reward
            })

        # Step 7: Transfer Event using core engine
        if episode % TRANSFER_FREQUENCY == 0 and episode > 0:
            # Flush any pending target-network count before the transfer block
            _stdout_interceptor.flush_and_log()
            
            transfer_header = f"\n[TRANSFER EVENT] Episode {episode}/{NUM_EPISODES}"
            print(transfer_header)
            _log(transfer_header)

            
            # Select Teacher using transferCoreEngine.select_teacher
            raw_agents = [d["agent"] for d in agents_data]
            raw_bufs   = [d["buffer"] for d in agents_data]

            teacher_agent = select_teacher(
                agents=raw_agents,
                mode=SOURCE_SELECTION_MODE,
                transfer_buffers=raw_bufs,
                ep_start_sharing=EP_START_SHARING,
                evaluation_range=EVALUATION_RANGE
            )

            if teacher_agent is not None:
                # Find teacher data index
                teacher_idx = [d["agent"] for d in agents_data].index(teacher_agent)
                teacher_buf = agents_data[teacher_idx]["buffer"]
                teacher_line = f"  ★ Teacher Selected: Agent {teacher_idx}"
                print(teacher_line)
                _log(teacher_line)

                for target_data in agents_data:
                    if target_data["id"] == teacher_idx:
                        continue
                    
                    # Apply Transfer using transfer_higher_delta_confidence
                    num_avail = len(teacher_buf)
                    if num_avail > 0:
                        b_size = min(BATCH_TRANSFER_SIZE, num_avail)
                        transferred = transfer_higher_delta_confidence(
                            sourceBuffer=teacher_buf,
                            target_agent=target_data["agent"],
                            target_estimator=target_data["ue"],
                            B=b_size
                        )
                        xfer_line = f"    ↳ Transferred {transferred} experiences -> Agent {target_data['id']}"
                        print(xfer_line)
                        _log(xfer_line)
            else:
                skip_line1 = f"NUM_EPISODES // TRANSFER_FREQUENCY = {NUM_EPISODES // TRANSFER_FREQUENCY}"
                skip_line2 = f"  ℹ Transfer skipped (episode {episode} < start {EP_START_SHARING} or no teacher)"
                print(skip_line1)
                print(skip_line2)
                _log(skip_line1)
                _log(skip_line2)

        if (episode + 1) % 100 == 0:
            _stdout_interceptor.flush_and_log()
            elapsed = time.time() - start_time
            avg_rew = np.mean([d["rewards_history"][-1] for d in agents_data])
            ep_line  = (f"Episode {episode + 1}/{NUM_EPISODES} | Train Calls: {train_calls} "
                        f"| Elapsed: {elapsed:.1f}s | Recent Avg Reward: {avg_rew:.2f}")
            print(ep_line)
            _log(ep_line)

        # --- Every 50 episodes: 6-metric stats block ---------------------
        if (episode + 1) % 50 == 0:
            _stdout_interceptor.flush_and_log()
            elapsed = time.time() - start_time

            mean_ep  = float(np.mean(block_ep_lens))  if block_ep_lens else 0.0
            max_ep   = int(max(block_ep_lens))         if block_ep_lens else 0

            stats_lines = [
                "",
                f"[STATS] episode={episode + 1:>6d}  elapsed_s={elapsed:>9.1f}  "
                f"cum_steps={cum_steps_all:>10d}",
                f"        mean_ep_len={mean_ep:>7.1f}  max_ep_len={max_ep:>5d}  "
                f"n_truncated={block_truncated:>4d}  (block of {len(block_ep_lens)} agent-eps)",
                "",
            ]
            for sl in stats_lines:
                print(sl)
                _log(sl)

            # Reset per-block accumulators
            block_ep_lens   = []
            block_truncated = 0
            
    # Close environments & save models
    sim_models_dir = os.path.join(MODELS_DIR, f"sim_{sim_id}")
    os.makedirs(sim_models_dir, exist_ok=True)

    for data in agents_data:
        data["env"].close()
        model_file = os.path.join(sim_models_dir, f"agent_{data['id']}.pth")
        torch.save(data["agent"].Qnet.state_dict(), model_file)
        print(f"  [Saved Model] {model_file}")

    # Save CSV
    fieldnames = ["Simulation", "Agent", "Episode", "Reward"]
    with open(RESULTS_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[SUCCESS] experiment completed. Results saved to {RESULTS_CSV}")

    # Return the per-agent reward history: list of NUM_AGENTS lists
    return agents_data


# ---------------------------------------------------------------------------
# Print Experiment Summary
# ---------------------------------------------------------------------------

print_experiment_summary(CONFIG_FILE, parameters)


# ---------------------------------------------------------------------------
# Run all simulations
# ---------------------------------------------------------------------------


for sim_id in range(NUM_SIMULATIONS):
    result = run_pacman_simulation(sim_id=sim_id, seed=7, models_dir=MODELS_DIR, param_file=CONFIG_FILE)



if __name__ == "__main__":
    pass
