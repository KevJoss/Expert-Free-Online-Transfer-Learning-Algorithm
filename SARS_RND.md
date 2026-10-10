# SARS-RND: Epistemic Uncertainty Estimation (`SARSrnd.py`)

This document provides a comprehensive explanation of the **SARS-RND (Random Network Distillation applied to State-Action-Reward-Next_State transitions)** mechanism implemented in [`SARSrnd.py`](./SARSrnd.py). It focuses on explaining how reinforcement learning agents quantify their level of uncertainty, how the network is trained online, and why prediction error reliably reflects state familiarity.

---

## 1. What is SARS-RND?

In Reinforcement Learning, **epistemic uncertainty** refers to the agent's lack of knowledge about the environment due to limited exploration. Knowing its own uncertainty allows an agent to assess whether it is acting in familiar territory or in unexplored regions.

Traditional count-based exploration counts visits $N(s)$ to each state. However, in continuous state spaces (such as CartPole physics or coordinate-based Pacman), identical states are rarely visited twice, making discrete lookup tables impossible.

**SARS-RND** solves this by generalizing the count-based intuition to continuous domains using **Random Network Distillation (RND)** (Burda et al., OpenAI, 2018), extending it from raw states $s$ to complete transition tuples:
$$x = [s, a, r, s'] \in \mathbb{R}^{2 \cdot |S| + |A| + |R|}$$

---

## 2. Core Architecture: The Two Networks

In [`SARSrnd.py`](./SARSrnd.py), the `UE` (Uncertainty Estimator) class initializes two neural networks with identical Multi-Layer Perceptron (MLP) architectures:

### 2.1 The Target Network (`rnd_net`)
* **Role**: Deterministic pseudo-hash function.
* **Initialization**: Initialized with fixed random weights.
* **Training**: **Never trained or modified** (`torch.no_grad()`, `eval()` mode).
* **Behavior**: For any transition vector $x$, it maps $x$ to a fixed, deterministic target embedding $y_1 \in \mathbb{R}^{1024}$.
  * For state $A$, it always outputs $y_A$.
  * For state $B$, it outputs a distinct target $y_B$.

### 2.2 The Predictor Network (`pred_net`)
* **Role**: Trainable distillation network.
* **Initialization**: Initialized with different random weights.
* **Training**: **Trained online via backpropagation** (`RMSprop`, learning rate `1e-3`).
* **Objective**: Learn to predict the output of the target network $y_1$ solely on the transitions $x$ experienced by the agent.

---

## 3. How Agents Obtain Their Level of Uncertainty

### 3.1 The Uncertainty Metric

Uncertainty is defined as the **Mean Squared Error (MSE)** between the output of the predictor network and the output of the fixed target network for a given transition $x$:

$$\mathcal{U}(s, a, r, s') = \frac{1}{D} \sum_{k=1}^{D} \left( \text{pred\_net}(x)_k - \text{rnd\_net}(x)_k \right)^2$$

Where $D$ is the embedding dimension (`s_encode_size = 1024`).

In [`SARSrnd.py`](./SARSrnd.py):
```python
def _compute_uncertainty(self, x):
    x = torch.FloatTensor(np.array(x)).to(self.device)
    with torch.no_grad():
        y1 = self.rnd_net(x)   # Fixed target embedding (never changes)
    y2 = self.pred_net(x)       # Current predictor embedding
    return ((y2 - y1) ** 2)     # Squared prediction error
```

### 3.2 Why Prediction Error Measures Familiarity

1. **Unvisited or Rare Transitions (High Uncertainty)**:
   * When an agent enters a new region of the state space, `pred_net` has never been trained on transitions with similar feature distributions.
   * Because `pred_net` and `rnd_net` started with different random weights, their outputs for unfamiliar inputs are uncorrelated.
   * As a result, the distance $\|y_2 - y_1\|^2$ is large $\implies$ **High Uncertainty**.

2. **Frequently Visited Transitions (Low Uncertainty)**:
   * Each time the agent visits a transition, `learn()` runs gradient descent on `pred_net` to pull its output closer to $y_1$.
   * Because the target $y_1$ is completely static and deterministic, this is a standard supervised regression task.
   * Neural networks easily minimize regression loss on recurrent training samples.
   * As a result, the distance $\|y_2 - y_1\|^2 \to 0$ $\implies$ **Low Uncertainty**.

---

## 4. The Online Training Flow

In the agent's interaction loop (e.g., `experiments/run_cartpole.py`), the uncertainty estimation and training follow a strict sequence at every environment step:

```
Step 1: Environment emits transition (s, a, r, s')
Step 2: Agent computes uncertainty -> get_uncertainty(s, a, r, s')
Step 3: Uncertainty is stored into transfer_buffer
Step 4: Predictor network learns on the transition -> learn(s, a, r, s')
```

### Detailed Execution of `get_uncertainty` and `learn`

#### 1. Evaluation (`get_uncertainty`)
```python
def get_uncertainty(self, s, a, r, n_s):
    x = np.concatenate((s, a, r, n_s))
    return self._compute_uncertainty(x).mean().tolist()
```
* Evaluates the current transition **without updating model weights**.
* The resulting scalar represents the novelty of that experience at the exact moment it happened.

#### 2. Model Update (`learn`)
```python
def learn(self, s, a, r, n_s):
    x = np.concatenate((s, a, r, n_s))
    self.temp_batch.append(x)
    
    complessive_loss = self._get_multi_uncertainty(self.temp_batch)
    self.optimizer.zero_grad()
    complessive_loss.mean().backward()  # Backprop only through pred_net
    self.optimizer.step()
    self.temp_batch = []
    return complessive_loss[-1].tolist()
```
* Takes the transition vector $x$ and performs a gradient descent step on `pred_net`.
* The target network `rnd_net` is frozen, ensuring that gradients only update the predictor's parameters.
* This incremental update immediately improves `pred_net`'s accuracy for subsequent encounters with similar states.

---

## 5. Why RND Avoids the "Noisy TV" Problem

A classic alternative to RND is predicting the next state directly: $\text{model}(s, a) \approx s'$.

However, next-state prediction suffers from the **Noisy TV problem**:
* If the environment has inherent stochasticity or noisy observations (e.g., random background movement or sensor noise), next-state prediction error will **always remain high**, regardless of how many times the agent visits that state.
* The agent would mistakenly identify pure random noise as "high novelty/uncertainty".

**Why SARS-RND is immune to this**:
* The target function `rnd_net` is purely deterministic. Given the exact same vector $x = [s, a, r, s']$, it produces the exact same output $y_1$ every single time.
* Therefore, prediction error depends strictly on whether `pred_net` has been trained on that input distribution, isolating **pure epistemic uncertainty**.

---

## 6. How SARS-RND Connects to the Rest of the Project

SARS-RND serves as the measurement foundation for the entire transfer learning algorithm:

1. **Step-Level Tagging (`transferBuffer.py`)**:
   * Every experience pushed into the buffer carries its step uncertainty: `transfer_buffer.push(s, a, r, s', done, uncertainty)`.

2. **Teacher Election (`transferCoreEngine.select_teacher`)**:
   * In `avg_uncertainty` mode, the algorithm queries `transfer_buffer.get_avg_uncertainty()`.
   * The agent with the **lowest average uncertainty** is elected as teacher because its `pred_net` has the lowest mean error, proving it has accumulated the most consolidated knowledge of the explored environment.

3. **Cross-Agent Evaluation (`transferCoreEngine.estimate_uncertainty`)**:
   * During transfer, the student's `SARSrnd` instance evaluates the teacher's stored experiences using `_get_multi_uncertainty(batch)`.
   * This reveals whether the teacher's experiences are novel to the student, driving the confidence delta calculation:
     $$\Delta\text{conf} = \text{Uncertainty}_{\text{teacher}} - \text{Uncertainty}_{\text{student}}$$
