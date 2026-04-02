"""
REINFORCE Algorithm Training Script
For Smart Farm Disease Management

Since Stable Baselines 3 does not have a standalone 'Vanilla REINFORCE' class, 
we implement it according to best academic practices by using the PPO/A2C architecture 
and mathematically constraining it to behave exactly like REINFORCE (Monte Carlo Policy Gradient).

Constrains applied to make it REINFORCE:
- n_steps = full episode (Monte Carlo returns)
- gae_lambda = 1.0 (No bootstrapping, true episodic returns)
- vf_coef = 0.0 (Critic loss ignored, pure policy gradient)
- clip_range = 100.0 (No PPO clipping)
- n_epochs = 1 (Only one update per batch, like pure REINFORCE)
"""

import numpy as np
import os
from datetime import datetime
from pathlib import Path
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import SmartCropDiseaseEnv

def train_reinforce(
    hyperparams: dict,
    experiment_name: str,
    total_timesteps: int = 20000,
    save_dir: str = 'models/reinforce'
):
    """
    Train REINFORCE agent (configured via unclipped PPO).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training REINFORCE - {experiment_name}")
    print(f"{'='*70}")
    print(f"Hyperparameters: {hyperparams}")
    
    # Create environment
    env = DummyVecEnv([lambda: SmartCropDiseaseEnv(grid_size=5, episode_length=100)])
    
    # Force REINFORCE mathematical constraints
    reinforce_constrained_hyperparams = hyperparams.copy()
    reinforce_constrained_hyperparams.update({
        'n_steps': 100,            # Monte Carlo: full episode length
        'batch_size': 100,         # Same as n_steps
        'gae_lambda': 1.0,         # Classic Monte Carlo returns (no bias)
        'vf_coef': 0.0,            # Disable critic loss completely
        'clip_range': 100.0,       # Disable PPO clipping
        'n_epochs': 1,             # Single gradient step per rollout
    })
    
    # Create model
    model = PPO(
        'MultiInputPolicy',
        env,
        verbose=0,
        **reinforce_constrained_hyperparams,
        tensorboard_log='logs/reinforce'
    )
    
    # Train model
    start_time = datetime.now()
    model.learn(total_timesteps=total_timesteps)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Save model
    model_path = os.path.join(save_dir, f"reinforce_{experiment_name}")
    model.save(model_path)
    
    # Evaluate model
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    episode_rewards = []
    episode_reward = 0
    
    for _ in range(2000):  # 2000 steps evaluation
        action, _ = model.predict(observation, deterministic=True) # type: ignore
        step_result = env.step(action)
        observation, reward, done = step_result[0], step_result[1], step_result[2]
        episode_reward += reward[0]
        
        if done[0]:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            observation = env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]
    
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    
    metrics = {
        'experiment_name': experiment_name,
        'algorithm': 'REINFORCE',
        'hyperparameters': hyperparams, # Log the varied ones
        'total_timesteps': total_timesteps,
        'training_time_seconds': training_time,
        'mean_episode_reward': float(mean_reward),
        'std_episode_reward': float(std_reward),
        'num_episodes': len(episode_rewards),
        'timestamp': start_time.isoformat()
    }
    
    metrics_path = os.path.join(save_dir, f"metrics_reinforce_{experiment_name}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    env.close()
    return metrics

def run_reinforce_experiments():
    """Run 10 REINFORCE experiments with different hyperparameters"""
    # We strictly vary REINFORCE-applicable parameters
    hyperparameter_grid = [
        {'learning_rate': 1e-4, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 5e-4, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 1e-3, 'gamma': 0.99, 'ent_coef': 0.0},
        {'learning_rate': 5e-4, 'gamma': 0.95, 'ent_coef': 0.0},
        {'learning_rate': 5e-4, 'gamma': 0.999, 'ent_coef': 0.0},
        {'learning_rate': 1e-4, 'gamma': 0.95, 'ent_coef': 0.01},
        {'learning_rate': 5e-4, 'gamma': 0.99, 'ent_coef': 0.01},
        {'learning_rate': 1e-3, 'gamma': 0.99, 'ent_coef': 0.01},
        {'learning_rate': 5e-4, 'gamma': 0.99, 'ent_coef': 0.05},
        {'learning_rate': 1e-3, 'gamma': 0.95, 'ent_coef': 0.05},
    ]
    
    all_metrics = []
    for i, hyperparams in enumerate(hyperparameter_grid, 1):
        experiment_name = f"exp_{i:02d}"
        metrics = train_reinforce(hyperparams, experiment_name)
        all_metrics.append(metrics)
    
    best_exp = max(all_metrics, key=lambda x: x['mean_episode_reward'])
    print(f"\\nBest REINFORCE experiment: {best_exp['experiment_name']} (Reward: {best_exp['mean_episode_reward']:.2f})")
    return all_metrics

if __name__ == "__main__":
    run_reinforce_experiments()
