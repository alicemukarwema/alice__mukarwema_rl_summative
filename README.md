# Smart Farm Disease Management - RL Agent

## Overview

This project implements a **Reinforcement Learning solution** for intelligent farm disease management. An RL agent learns to minimize disease spread while maximizing crop yield through strategic decision-making.

### Problem Statement

**"How can an AI agent optimally manage disease spread in a farm to minimize losses while maintaining cost-effectiveness?"**

In agriculture, diseases like bacterial blight, fungal infections, and crop viruses can devastate entire harvests. Manual interventions are costly and reactive. This project demonstrates how RL agents can learn to:
- Monitor crop health in real-time
- Predict disease spread patterns
- Select cost-effective treatments
- Balance immediate cost with long-term yield preservation

## Environment Design

### Smart Crop Disease Management System

A custom Gymnasium-compliant environment simulating a 5×5 farm grid with 25 plants.

#### State Space (Observations)

The agent observes:
- **Crop Health**: 0-100 per plant (% vitality)
- **Disease Severity**: 0-100 per plant (infection level)
- **Soil Moisture**: 0-100% (affects disease spread)
- **Weather Conditions**:
  - Temperature (0-100°C range)
  - Humidity (0-100%)
  - Rainfall (0-100mm)
- **Days Since Infection**: 0-30 per plant
- **Treatment History**: Count of interventions
- **Total Infected Plants**: Number of currently infected plants

#### Action Space

The agent can take 6 discrete actions:

| Action | Cost | Effect | Use Case |
|--------|------|--------|----------|
| **0 - Monitor** | $0 | Observe without intervention | Low-risk monitoring |
| **1 - Spray Fungicide** | $5 | Reduces disease by 70%, instant effect | Active infection control |
| **2 - Apply Neem Oil** | $2 | Reduces disease by 50%, gradual effect | Budget-conscious treatment |
| **3 - Remove Plant** | $10 | Eliminates plant + disease, prevents spread | Severe infection containment |
| **4 - Ventilation** | $1 | Reduces humidity, slows disease | Environmental control |
| **5 - Irrigation** | $0.50 | Optimal soil moisture | Disease prevention |

#### Reward Structure

```
Reward = (Healthy Plants × 0.5) + (Infected Plants × -2.0) - (Total Cost × 0.1)
```

**Components:**
- **Health Reward**: +0.5 per healthy plant (encourages maintenance)
- **Infection Penalty**: -2.0 per infected plant (discourages spread)
- **Cost Penalty**: -0.1× total spent (encourages efficiency)

#### Terminal Conditions

Episode ends when:
- Max 100 timesteps reached (episode truncates)
- All plants removed or dead (termination)

#### Environment Dynamics

- **Disease Spread**: Spreads to neighboring plants (4-connectivity)
- **Natural Progression**: Health declines ~0.5 per step without treatment
- **Infection Impact**: Accelerates health decline when infected
- **Weather Effects**: Humidity and rainfall increase spread probability
- **Soil Moisture**: Decreases naturally, affects disease susceptibility

## Project Structure

```
alice__mukarwema_rl_summative/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py            # Gymnasium environment (SmartCropDiseaseEnv)
│   └── rendering.py             # Pygame visualization components
├── training/
│   ├── dqn_training.py          # DQN (value-based) training with 10 experiments
│   └── pg_training.py           # PPO & A2C (policy gradient) with 10 each
├── models/
│   ├── dqn/                     # Saved DQN models & metrics
│   ├── pg/                      # Saved PPO/A2C models & metrics
│   └── *.pkl                    # Model checkpoints
├── analysis/
│   ├── visualizer.py            # Hyperparameter analysis & plotting
│   ├── *.png                    # Generated analysis plots
│   └── summary.html             # HTML results summary
├── logs/
│   └── *.csv                    # Training logs
├── demo_random_actions.py       # Static visualization (random agent demo)
├── main.py                      # Run best-performing agent
├── requirements.txt             # Dependencies (MANDATORY)
└── README.md                    # This file
```

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/username/alice_mukarwema_rl_summative
cd alice_mukarwema_rl_summative
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. View Environment Demo (Random Actions)

Demonstrates environment mechanics without training:
```bash
python demo_random_actions.py
```

**Output:**
- Pygame visualization of farm grid
- Real-time health/disease/action info
- Terminal verbose logging
- No learning involved (baseline for comparison)

### 2. Train RL Agents

#### Train DQN (Value-Based)
```bash
python -m training.dqn_training
```
- Trains 10 experiments with different hyperparameters
- Tests: learning rate, buffer size, batch size, gamma, exploration
- Saves models to `models/dqn/`
- ~15-20 minutes total

#### Train Policy Gradient Methods
```bash
python -m training.pg_training
```
- Trains 10 PPO experiments (state-of-the-art policy gradient)
- Trains 10 A2C experiments (REINFORCE-based with baseline)
- Tests: learning rate, trajectory length, batch size, entropy, gamma
- Saves models to `models/pg/`
- ~25-30 minutes total

### 3. Run Best-Performing Agent

#### View DQN Agent
```bash
python main.py --algorithm dqn --episodes 3
```

#### View PPO Agent
```bash
python main.py --algorithm ppo --episodes 3
```

#### View Both
```bash
python main.py --algorithm all --episodes 5
```

**Features:**
- Real-time Pygame visualization
- Verbose terminal output showing:
  - Actions taken per step
  - Rewards received
  - Plant health status
  - Cost tracking
- Episode summaries with statistics
- Performance metrics

### 4. Generate Analysis Plots

```bash
python analysis/visualizer.py
```

**Generates:**
1. `analysis/dqn_curves.png` - DQN hyperparameter effects
2. `analysis/ppo_curves.png` - PPO hyperparameter effects
3. `analysis/algorithm_comparison.png` - Algorithm comparison
4. `analysis/summary.html` - Interactive HTML summary

## Hyperparameter Tuning Strategy

### DQN (10 Experiments)

| Parameter | Tested Values | Notes |
|-----------|---------------|-------|
| Learning Rate | 1e-4, 5e-4, 1e-3 | Controls gradient step size |
| Buffer Size | 5k, 10k, 50k | Experience replay memory |
| Batch Size | 16, 32, 64 | Mini-batch gradient steps |
| Gamma | 0.95, 0.99, 0.999 | Discount factor (future value) |
| Exploration Fraction | 0.1, 0.3, 0.5 | ε-greedy schedule |
| Train Frequency | 4 | Update interval |

**Key Insights:**
- Higher learning rates (5e-4) generally perform better for this task
- Buffer size 10k-50k provides good experience diversity
- Gamma 0.99 balances short/long-term rewards well
- Exploration fraction 0.3 allows sufficient policy exploration

### PPO (10 Experiments)

| Parameter | Tested Values | Notes |
|-----------|---------------|-------|
| Learning Rate | 1e-4, 5e-4, 1e-3 | Policy update step size |
| Trajectory Length (n_steps) | 64, 128, 256 | Samples per update |
| Batch Size | 16, 32, 64 | Policy gradient batch |
| Entropy Coefficient | 0.0, 0.01, 0.05 | Exploration regularization |
| Gamma | 0.95, 0.99, 0.999 | Discount factor |
| GAE Lambda | 0.9, 0.95, 0.98 | Advantage estimation |
| Clip Range | 0.2 | Policy update constraint |

**Key Insights:**
- Learning rate 5e-4 shows consistent stability
- Entropy coefficient 0.01 balances exploration/exploitation
- Trajectory length 128 provides good update variance
- Gamma 0.99 with GAE lambda 0.95 works well

### A2C / REINFORCE (10 Experiments)

| Parameter | Tested Values | Notes |
|-----------|---------------|-------|
| Learning Rate | 1e-4, 5e-4, 1e-3 | REINFORCE step size |
| n_steps (episode length) | 3, 5, 10 | Trajectory length |
| Entropy Coefficient | 0.0, 0.01, 0.05 | Policy regularization |
| Normalize Advantage | True, False | Variance reduction |

**Key Insights:**
- REINFORCE with baseline (A2C) more stable than vanilla
- Lower learning rate (1e-4) beneficial due to policy-only updates
- Entropy coefficient 0.01 crucial for exploration
- Advantage normalization significantly improves stability

## Expected Performance

### Baseline (Random Actions)
- Episode Reward: -50 to +20
- Avg Healthy Plants: 8-12 / 25
- Treatment Success: Low (trial & error)

### Trained Agents (Typical)
- **DQN**: Episode Reward +100-250 | Healthy Plants: 18-22/25
- **PPO**: Episode Reward +150-350 | Healthy Plants: 20-24/25
- **A2C**: Episode Reward +80-200 | Healthy Plants: 16-21/25

*Note: Performance varies based on random initialization and exploration.*

## Algorithm Comparison

### DQN (Value-Based)
- **Approach**: Learns action value function Q(s,a)
- **Strengths**: Sample-efficient, off-policy learning
- **Weaknesses**: Can be unstable with continuous rewards
- **Best For**: Finite action spaces, experience replay benefits

### PPO (Policy Gradient)
- **Approach**: Learns policy directly with clipped surrogate loss
- **Strengths**: Stable, state-of-the-art performance
- **Weaknesses**: Higher sample complexity
- **Best For**: Complex environments, continuous learning

### A2C (REINFORCE-based)
- **Approach**: Policy gradient with value baseline
- **Strengths**: Simple, baseline reduces variance
- **Weaknesses**: Lower sample efficiency than PPO
- **Best For**: Educational purposes, stable training

## Visualization Features

### Pygame Rendering
- **Color Coding**:
  - 🟢 Green: Healthy plants (disease ≤ 30%)
  - 🔴 Red: Infected plants (disease > 30%)
  - 🟣 Dark Red: Severely infected (disease > 60%)
  - ⬜ Gray: Removed plants

- **Info Panel** shows:
  - Current step / episode
  - Healthy & infected plant counts
  - Average health & disease severity
  - Weather conditions (temperature, humidity, rainfall)
  - Soil moisture level
  - Total cost spent
  - Current action & reward

### Terminal Output
- Real-time logging every 5-10 steps
- Action names (not just indices)
- Reward tracking
- Plant statistics
- Episode summaries

## Testing & Validation

### Environment Validation
- ✅ Action space properly defined (6 discrete actions)
- ✅ Observation space structured (Dict with 7 sub-spaces)
- ✅ Disease spread mechanics working (neighbors affected)
- ✅ Reward structure producing meaningful signals
- ✅ Terminal conditions working (episode end detection)

### Agent Validation
- ✅ Models training without errors
- ✅ Deterministic inference works (best_policy mode)
- ✅ All 4 algorithms compatible with environment
- ✅ Hyperparameter variations tested (10 each)

## Troubleshooting

### ImportError: No module named 'gymnasium'
```bash
pip install gymnasium
```

### pygame display issues on headless systems
```bash
# Render to images instead
python main.py --no-render
```

### Out of memory with large buffer sizes
```python
# In dqn_training.py, reduce buffer_size:
'buffer_size': 5000  # Was 50000
```

### Models not found when running main.py
Ensure training completed:
```bash
ls models/dqn/*.zip  # Check for saved models
```

## References

### Libraries Used
- **Gymnasium**: Environment creation standard
- **Stable-Baselines3**: RL algorithm implementations
- **Pygame**: 2D visualization
- **NumPy/Pandas**: Numerical computing
- **Matplotlib**: Analysis plotting

### Key Papers
1. DQN: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
2. PPO: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
3. A2C: Mnih et al. (2016) - "Asynchronous Methods for Deep Reinforcement Learning"

### Agricultural Context
- Disease spread modeling
- Cost-benefit analysis of treatments
- Environmental factor interactions

## Future Enhancements

1. **Multi-agent**: Multiple farms competing for resources
2. **3D Visualization**: Unity/Panda3D for realistic farm simulation
3. **Market Integration**: Dynamic pricing for crops/treatments
4. **Seasonal Dynamics**: Year-long simulation with crop rotation
5. **Real Data**: Integration with actual farm disease datasets
6. **Transfer Learning**: Pre-train on similar diseases, fine-tune for new ones

## Author

**Alice Mukarwema**
MLU Course: Advanced Reinforcement Learning
Date: January 2026

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
1. Check README troubleshooting section
2. Review environment/training logs
3. Validate environment with demo_random_actions.py
4. Check hyperparameter ranges in training scripts

---

**Last Updated**: April 2026
**Status**: Complete & Tested
