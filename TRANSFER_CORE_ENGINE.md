# Transfer Core Engine Documentation (`transferCoreEngine.py`)

This document explains the architecture and operational mechanics of the knowledge transfer engine implemented in [`transferCoreEngine.py`](./transferCoreEngine.py). It provides an in-depth explanation of the two core mechanisms used in this research project: **`estimate_uncertainty`** and teacher selection by **`avg_uncertainty`**, detailing how states and data flow through the entire system.

---

## 1. Key Research Functions

### 1.1 `estimate_uncertainty(target_estimator, batch)`

#### What does it do?
It quantifies the **epistemic uncertainty** of the **target agent (student)** when evaluating a set of state-action transitions collected by another agent (the teacher).

```python
def estimate_uncertainty(target_estimator, batch):
    max_batch_size = 32
    idxs = [x for x in range(0, len(batch), max_batch_size)]
    uncertainties = []
    for idx in idxs[:-1]:
        uncertainties = np.hstack((uncertainties, target_estimator._get_multi_uncertainty(batch[idx:idx+max_batch_size]).detach().cpu().numpy()))
    idx = idxs[-1]
    uncertainties = np.hstack((uncertainties, target_estimator._get_multi_uncertainty(batch[idx:]).detach().cpu().numpy()))
    return uncertainties
```

* **Inputs**:
  * `target_estimator`: The uncertainty estimator belonging to the student agent (an instance of `SARSrnd` based on **Random Network Distillation (RND)**, defined in [`SARSrnd.py`](./SARSrnd.py)).
  * `batch`: A matrix of concatenated transitions $(s, a, r, s')$ extracted from the teacher's transfer buffer.
* **Processing**:
  1. Partitions the input transitions into sub-batches of maximum size 32 (`max_batch_size = 32`) to prevent GPU/CPU memory spikes during batched neural network forward passes.
  2. Calls `target_estimator._get_multi_uncertainty(...)`, which computes the mean squared error (MSE) between a fixed, randomly initialized target network and a trained predictor network.
  3. Detaches the PyTorch tensors from the computation graph (`.detach().cpu().numpy()`) and horizontally concatenates them into a continuous 1D NumPy array.
* **Output**: A 1D NumPy array containing the scalar uncertainty value for every evaluated transition.

---

### 1.2 `select_teacher` with `avg_uncertainty` Mode

#### What does it do?
Dynamically elects **which concurrent agent will act as the teacher (knowledge source)** during a transfer event.

```python
elif(mode == "avg_uncertainty"):
    for p in agents:
        if(p.episode < ep_start_sharing):
            return None  # Warm-up phase: too soon to share
    avg_uncertainty = []
    for i in range(len(agents)):
        avg_uncertainty.append(transfer_buffers[i].get_avg_uncertainty())

    teacher = agents[avg_uncertainty.index(min(avg_uncertainty))]
```

* **Selection Criterion**:
  * Unlike traditional performance-based heuristics (`best_performance`), this mode evaluates the agents' **epistemic familiarity with the environment**.
  * **Selects the agent with the LOWEST average uncertainty (`min(avg_uncertainty)`)**.
  * **Theoretical Motivation**: A lower prediction error in the RND network indicates that the agent has extensively explored and consolidated dynamics for the visited state space. Therefore, it is the most reliable agent to mentor peers.
* **Warm-up Check (`ep_start_sharing`)**:
  * Prevents premature transfer during initial exploratory phases where uncertainty estimates have not yet stabilized.

---

## 2. Module Interconnections & System Architecture

Neither function operates in isolation. They serve as the decision and evaluation core connecting RND networks, memory buffers, and the agent training loops.


### Component Interaction Summary

| Component                     | File                                                                                                            | Interaction with Core Functions                                                                                                     |
| :---------------------------- | :-------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| **`SARSrnd`**                 | [`SARSrnd.py`](./SARSrnd.py)                                                                                    | Contains the RND architecture. `estimate_uncertainty` directly invokes `_get_multi_uncertainty(batch)`.                             |
| **`transferBuffer`**          | [`transferBuffer.py`](./transferBuffer.py)                                                                      | Stores transitions and step-level uncertainties. `select_teacher` calls `transfer_buffers[i].get_avg_uncertainty()`.                |
| **`filter_delta_confidence`** | [`transferCoreEngine.py`](./transferCoreEngine.py)                                                              | Computes $\Delta\text{conf} = \text{Unc}_{\text{teacher}} - \text{Unc}_{\text{student}}$ using outputs from `estimate_uncertainty`. |
| **Experiment Runner**         | [`experiments/run_cartpole.py`](./experiments/run_cartpole.py) / [`run_pacman.py`](./experiments/run_pacman.py) | Drives the simulation loop and triggers transfer events every `TRANSFER_FREQUENCY` episodes.                                        |

---

## 3. Data & State (`state`) Lifecycle

1. **State Observation**: The environment emits `state`. The agent selects an action $a$ via its $\epsilon$-greedy policy.
2. **Step Execution**: The environment steps forward, producing $(s, a, r, s', done)$.
3. **Local Uncertainty Calculation**: The agent's local RND estimator evaluates the transition novelty via `get_uncertainty(...)`.
4. **Buffer Storage**: The full transition tuple plus the uncertainty value is saved in `transfer_buffer`.
5. **Teacher Selection**: Every $N$ episodes, `select_teacher` compares `get_avg_uncertainty()` across agents and picks the one with the lowest value.
6. **Cross-Agent Evaluation**: The candidate transitions from the teacher are fed into `estimate_uncertainty` using the **student's** estimator.
7. **Delta Filtering**:
   $$\Delta\text{conf} = \text{Unc}_{\text{teacher}} - \text{Unc}_{\text{student}}$$
   Large negative values highlight experiences where the teacher is highly certain and the student is highly uncertain.
8. **Knowledge Injection**: The top $B$ experiences with the largest confidence advantage are injected into the student's learning pipeline via `target_agent.learn_from_samples(...)`.

---

## 4. Overall Overview of `transferCoreEngine.py`

In addition to the two primary functions, the module provides a suite of supporting transfer strategies:

* **`estimate_surprise(target_agent, batch)`**: Evaluates the Temporal Difference (TD) loss of the target agent over a batch to quantify how surprising the experiences are.
* **`filter_delta_confidence(sourceBuffer, target_estimator)`**: Pairs the teacher's stored uncertainties with the student's evaluated uncertainties from `estimate_uncertainty` to yield $\Delta\text{conf}$.
* **`transfer_higher_delta_confidence(...)`**: Sorts candidate transitions by $\Delta\text{conf}$ and transfers the top $B$ samples.
* **`random_transfer(...)`**: Baseline strategy transferring a uniform random sample of $B$ transitions from the teacher's buffer.
* **`transfer_by_higher_loss_and_confidence_mixed(...)`**: A hybrid strategy that normalizes and balances 50% target surprise and 50% confidence delta to prioritize high-impact transitions.
