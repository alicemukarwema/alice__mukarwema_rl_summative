"""
DQN (Deep Q-Network) Training Script
Value-based RL method for Smart Farm Disease Management
"""

import numpy as np
import os
from datetime import datetime
from pathlib import Path
import json

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.custom_env import SmartCropDiseaseEnv


class RewardLoggingCallback(BaseCallback):
    """Custom callback to log rewards and metrics"""
    
    def __init__(self, verbose=0, log_dir='logs'):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_costs = []
    
    def _on_step(self) -> bool:
        return True


def train_dqn(
    hyperparams: dict,
    experiment_name: str,
    total_timesteps: int = 50000,
    save_dir: str = 'models/dqn'
):
    """
    Train DQN agent with specified hyperparameters.
    
    Args:
        hyperparams: Dictionary of hyperparameters
        experiment_name: Name for this experiment
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
    
    Returns:
        Dictionary with training metrics
    """
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training DQN - {experiment_name}")
    print(f"{'='*70}")
    print(f"Hyperparameters: {hyperparams}")
    
    # Create environment
    env = DummyVecEnv([lambda: SmartCropDiseaseEnv(grid_size=5, episode_length=100)])
    
    # Create DQN model
    model = DQN(
        'MultiInputPolicy',
        env,
        verbose=1,
        **hyperparams,
        tensorboard_log='logs/dqn'
    )
    
    # Train model
    print(f"\nTraining for {total_timesteps} timesteps...")
    start_time = datetime.now()
    
    model.learn(total_timesteps=total_timesteps)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Save model
    model_path = os.path.join(save_dir, f"dqn_{experiment_name}")
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Evaluate model
    print(f"\nEvaluating model...")
    observation = env.reset()
    # Extract observation from DummyVecEnv reset (returns tuple)
    if isinstance(observation, tuple):
        observation = observation[0]
    episode_rewards = []
    episode_reward = 0
    
    for _ in range(5000):  # 5000 steps for evaluation
        action, _ = model.predict(observation, deterministic=True)  # type: ignore
        step_result = env.step(action)
        observation = step_result[0]
        reward = step_result[1]
        done = step_result[2]
        episode_reward += reward[0]
        
        # Check for episode end
        if done[0]:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            observation = env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]
    
    # Calculate metrics
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0
    std_reward = np.std(episode_rewards) if episode_rewards else 0
    
    metrics = {
        'experiment_name': experiment_name,
        'hyperparameters': hyperparams,
        'total_timesteps': total_timesteps,
        'training_time_seconds': training_time,
        'mean_episode_reward': float(mean_reward),
        'std_episode_reward': float(std_reward),
        'num_episodes': len(episode_rewards),
        'timestamp': start_time.isoformat()
    }
    
    print(f"\nTraining Results:")
    print(f"  Mean Episode Reward: {mean_reward:.2f}")
    print(f"  Std Episode Reward: {std_reward:.2f}")
    print(f"  Training Time: {training_time:.1f}s")
    
    # Save metrics
    metrics_path = os.path.join(save_dir, f"metrics_{experiment_name}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    env.close()
    return metrics


def run_dqn_experiments():
    """Run multiple DQN experiments with different hyperparameters"""
    
    # Define hyperparameter combinations
    hyperparameter_grid = [
        # Learning rate variations
        {
            'learning_rate': 1e-4,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
        },
        {
            'learning_rate': 5e-4,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
        },
        {
            'learning_rate': 1e-3,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
        },
        # Buffer size variations
        {
            'learning_rate': 5e-4,
            'buffer_size': 5000,
            'learning_starts': 500,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
        },
        {
            'learning_rate': 5e-4,
            'buffer_size': 50000,
            'learning_starts': 5000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
        },
        # Batch size variations
        {
            'learning_rate': 5e-4,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 16,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
        },
        {
            'learning_rate': 5e-4,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 64,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
        },
        # Gamma variations (discount factor)
        {
            'learning_rate': 5e-4,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.95,
            'train_freq': 4,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
        },
        {
            'learning_rate': 5e-4,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.999,
            'train_freq': 4,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
        },
        # Exploration variations
        {
            'learning_rate': 5e-4,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.01,
        },
        {
            'learning_rate': 5e-4,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'exploration_fraction': 0.5,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
        },
    ]
    
    all_metrics = []
    
    for i, hyperparams in enumerate(hyperparameter_grid, 1):
        experiment_name = f"exp_{i:02d}"
        metrics = train_dqn(
            hyperparams=hyperparams,
            experiment_name=experiment_name,
            total_timesteps=50000
        )
        all_metrics.append(metrics)
    
    # Save summary
    summary_path = 'models/dqn/summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"\n{'='*70}")
    print("DQN Training Summary")
    print(f"{'='*70}")
    print(f"Total experiments: {len(all_metrics)}")
    
    best_exp = max(all_metrics, key=lambda x: x['mean_episode_reward'])
    print(f"\nBest experiment: {best_exp['experiment_name']}")
    print(f"  Mean Reward: {best_exp['mean_episode_reward']:.2f}")
    print(f"  Hyperparameters: {best_exp['hyperparameters']}")
    
    return all_metrics


if __name__ == "__main__":
    metrics = run_dqn_experiments()
