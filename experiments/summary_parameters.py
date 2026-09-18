import sys
import gymnasium as gym
import ale_py

def print_experiment_summary(config_file: str, params: dict):
    env_name = params['training'].get('env_name')
    max_episode_steps = params['training'].get('max_episode_steps')

    try:
        kwargs = {}
        if max_episode_steps:
            kwargs["max_episode_steps"] = max_episode_steps
        if "CartPole" in env_name:
            sample_env = gym.make(env_name, **kwargs)
        else:
            sample_env = gym.make(env_name, obs_type="ram", max_episode_steps=max_episode_steps)
        
        # Obtain the attributes of the environments
        state_dim = sample_env.observation_space.shape[0] if len(sample_env.observation_space.shape) > 0 else sample_env.observation_space.n
        action_dim = sample_env.action_space.n
        max_timestep_val = getattr(sample_env.spec, 'max_episode_steps', max_episode_steps or "N/A")
        sample_env.close()

        
    except Exception as e:
        print("The attributes state_dim, action_dim and max_timestep_val ")
        print("dont found in the environment")
        print(e)
        sys.exit(1)

    # Obtain the parameters of the LP method and Transfer Algorithm
    dqn = params['dqn_agent']
    tr = params['training']
    tf = params['transfer']


    # Obtain the parameters of the neural network architecture
    h_dim = dqn['hidden_dim']
    if isinstance(h_dim, (list, tuple)):
        h1 = h_dim[0]
        h2 = h_dim[1] if len(h_dim) > 1 else h_dim[0]
    else:
        h1, h2 = h_dim, h_dim

    # SHOW THE RESULTS IN THE CONSOLE
    print("=" * 75)
    print(f"  EXPERIMENT CONFIGURATION SUMMARY  [{config_file}]")
    print("=" * 75)

    print("\n--- DUELING DQN NETWORK PARAMETERS ---")
    print(f"  Input Layer           : FC({state_dim}, {h1})")
    print(f"  Hidden Layer          : FC({h1}, {h2})")
    print(f"  Advantage Layer       : FC({h2}, {action_dim})")
    print(f"  Value Layer           : FC({h2}, 1)")
    print(f"  Activation            : ReLU")
    print(f"  Loss                  : MSE")
    print(f"  Optimizer             : Adam")
    print(f"  Learning Rate         : {dqn['learning_rate']}")
    print(f"  Betas                 : {dqn.get('betas')}")
    print(f"  Gamma                 : {dqn['gamma']}")
    print(f"  Mini Batch Size       : {dqn['batch_size']}")
    print(f"  Policy Update Step    : Cada {dqn.get('train_frequency')} timesteps")
    print(f"  Exploration Mode      : {dqn['exploration_mode']} (Temp={dqn.get('temperature')})")
    print(f"  Replay Buffer Size    : {dqn['replay_buffer_size']:,}")
    print(f"  Ep. Start Training    : {dqn['warmup_episodes']} episodes")

    print("\n--- EXPERT-FREE ONLINE TRANSFER LEARNING (EF-OnTL) PARAMETERS ---")
    print(f"  Number of agents      : {tr['num_agents']}")
    print(f"  Transfer Frequency    : Cada {tf['transfer_frequency']} episodes")
    print(f"  Transfer Buffer Cap.  : {tr.get('buffer_memory_size'):,} tuples")
    print(f"  Source Selection      : {tf['source_selection_mode']}")
    print(f"  Filtering Criteria    : Higher Delta Confidence (TCS)")
    print(f"  Budget (Presupuesto)  : {tf['batch_transfer_size']} tuples/event")
    print(f"  Episode Start Transfer: {tf['ep_start_sharing']} episodes")
    print(f"  Max Timestep          : {max_timestep_val} timesteps/episode")
    print(f"  Max Episode           : {tr['num_episodes']} episodes")
    print("=" * 75 + "\n")