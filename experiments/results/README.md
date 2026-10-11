# Source Selection Evaluation & Teacher Agent Rankings (1 to 5)

This directory contains evaluation logs, result CSVs, and analysis scripts that measure how effectively the **Expert-Free Online Transfer Learning (EF-OnTL)** algorithm chooses source ("teacher") agents among concurrent peer learners.

---

## 1. Why Rank Agents from 1 to 5?

In this research project, **5 peer agents** ($A_0, A_1, A_2, A_3, A_4$) train concurrently from scratch without any pre-trained human or algorithmic expert. Periodically (e.g., every 200 episodes, starting from episode 600), a **knowledge transfer event** occurs:

1. One agent is dynamically designated as the **Teacher** (source of transitions).
2. The remaining agents act as **Students** (target recipients of transferred knowledge).

Under the `avg_uncertainty` strategy, the teacher is chosen based purely on **epistemic uncertainty** estimated via Random Network Distillation (RND), selecting:
$$\text{Teacher} = \arg\min_{i} \overline{U}_i$$
where $\overline{U}_i$ is the average uncertainty of agent $i$ over its transfer buffer.

### The Core Research Question
> **Without an external oracle or access to ground-truth performance rewards, does selecting the agent with the lowest uncertainty reliably select a competent, high-performing agent?**

To answer this question quantitatively, the evaluation scripts compute the **1-to-5 Performance Ranking** of the chosen teacher at each transfer event.

---

## 2. Ranking Methodology & Interpretation

### How the Rank is Computed
For each transfer event at episode $E$:
1. A sliding window of the previous $W = 100$ episodes is analyzed ($[E - W, E - 1]$).
2. The mean empirical reward is computed for each of the 5 agents:
   $$\bar{R}_i = \frac{1}{W} \sum_{ep = E - W}^{E - 1} R_i(ep), \quad \text{for } i \in \{0, 1, 2, 3, 4\}$$
3. The agents are ordered by reward descending (`rank(ascending=False)`):
   - **Rank 1**: Agent with the highest average reward (best peer).
   - **Rank 2**: Second best peer.
   - **Rank 3**: Median peer.
   - **Rank 4**: Fourth best peer.
   - **Rank 5**: Agent with the lowest average reward (worst peer).
4. The rank of the selected teacher $t$ is recorded as $\text{Rank}(A_t) \in \{1, 2, 3, 4, 5\}$.

### Interpretation Guide

| Teacher Rank | Meaning | Assessment |
| :---: | :--- | :--- |
| **1 / 5** | The algorithm chose the **single best** performing peer available. | Optimal selection |
| **2 / 5** | The algorithm chose the **second best** performing peer. | High quality |
| **3 / 5** | The algorithm selected the middle agent (equivalent to random choice). | Average ($\mathbb{E}[\text{rank}] = 3.0$) |
| **4 / 5 – 5 / 5** | The algorithm selected an underperforming or lagging peer. | Suboptimal selection |

**Overall Goal**: An average teacher rank significantly lower than $3.0$ (e.g., $1.5 - 2.0$) empirically proves that lowest epistemic uncertainty is an effective proxy for policy mastery.

---

## 3. Analysis Scripts

### Script A: `compute_performance_agents.py`
- **Location**: `experiments/results/pacman/seed7/compute_performance_agents.py`
- **Purpose**: A direct script that loads empirical CSV results across benchmarks (Cart-Pole and MsPacman) and cross-references them against recorded teacher selections.
- **Workflow**:
  1. Defines the historical teacher selection per transfer episode for each run.
  2. Pivots reward data per episode and agent.
  3. Calculates the mean reward per agent over the 100-episode window.
  4. Outputs each event's teacher rank and the run's **average rank**.

#### Example Output:
```text
=== MP-2  Pacman s7 ===
  ep600:  teacher=A0  rank=1/5   performance=[288.5 240.2 195.0 210.4 180.1]
  ep800:  teacher=A1  rank=2/5   performance=[260.1 310.4 205.2 190.0 220.8]
  ep1000: teacher=A0  rank=1/5   performance=[350.0 305.2 210.0 240.5 215.1]
  ep1200: teacher=A4  rank=1/5   performance=[290.4 270.0 260.2 280.1 330.5]
  ep1400: teacher=A4  rank=2/5   performance=[310.2 290.5 275.4 300.0 340.1]
  ep1600: teacher=A4  rank=1/5   performance=[320.0 310.5 285.0 290.0 365.2]
  average rank = 8/6 = 1.33/5
```
*In this example, an average rank of 1.33/5 demonstrates that the teacher selection almost always picked the top-performing peer.*

---

### Script B: `comparison_pacman_vs_cart-pole.py`
- **Location**: `results_metrics/comparison_pacman_vs_cart-pole/comparison_pacman_vs_cart-pole.py`
- **Purpose**: An automated parser that reads execution logs (`.log`) and CSV results without hardcoding. It extracts both teacher identities and logged uncertainty vectors to perform statistical validation.
- **Key Metrics Computed**:
  1. **Teacher Rank (1 to 5)**: Verified dynamically from log transfer events.
  2. **Consistency Check (`argmin == teacher`)**: Confirms that $\arg\min(U)$ matches the teacher reported in logs.
  3. **Spearman Rank Correlation ($\rho$)**: Measures monotonic correlation between agent uncertainty vector $[U_0..U_4]$ and agent reward vector $[R_0..R_4]$.
     - **Expected**: $\rho < 0$ (lower uncertainty should correlate with higher reward).
  4. **Uncertainty Spread**: Difference between minimum and maximum agent uncertainty:
     $$\text{Spread} = \left(\frac{U_{\max}}{U_{\min}} - 1\right) \times 100\%$$
  5. **Cross-Domain Comparison Table**: Compares mean teacher rank and mean $\rho$ across Cart-Pole and MsPacman seeds.

#### Example Output:
```text
Performance window = 100 episodes prior to each transfer event

==========================================================================
MP-2  MsPacman   seed 7
  Transfer events detected: [600, 800, 1000, 1200, 1400, 1600]
  ep  600 | teacher=A0 rank 1/5 | rho=-0.80 | mean_U=0.003508 spread=   7% | argmin==teacher: True
  ep  800 | teacher=A1 rank 2/5 | rho=-0.60 | mean_U=0.004377 spread=  70% | argmin==teacher: True
  ep 1000 | teacher=A0 rank 1/5 | rho=-0.90 | mean_U=0.003215 spread=  45% | argmin==teacher: True
  ep 1200 | teacher=A4 rank 1/5 | rho=-0.70 | mean_U=0.002890 spread=  32% | argmin==teacher: True
  ep 1400 | teacher=A4 rank 2/5 | rho=-0.40 | mean_U=0.002450 spread=  28% | argmin==teacher: True
  ep 1600 | teacher=A4 rank 1/5 | rho=-0.80 | mean_U=0.002100 spread=  35% | argmin==teacher: True
  --> Average teacher rank = 1.33/5   ranks=[1, 2, 1, 1, 2, 1]
  --> Mean Spearman rho = -0.700   negative rhos: 6/6
  --> Consistency argmin(U) == logged teacher: 6/6

==========================================================================
SUMMARY
Run                          Avg Rank    Mean Rho
CP-C  Cart-Pole  seed 42       2.00/5      -0.450
MP-1  MsPacman   seed 42       1.83/5      -0.620
MP-2  MsPacman   seed 7        1.33/5      -0.700
```

---

## 4. How to Run the Analyses

Run either script from the root of the repository or from their respective directories:

```bash
# 1. Run direct teacher performance rank computation:
python experiments/results/pacman/seed7/compute_performance_agents.py

# 2. Run automated log parser and cross-domain statistical comparison:
python results_metrics/comparison_pacman_vs_cart-pole/comparison_pacman_vs_cart-pole.py
```
