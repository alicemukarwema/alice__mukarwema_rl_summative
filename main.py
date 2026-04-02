"""
Main Entry Point - Run Best Performing RL Agent
Demonstrates the trained agent in action with visualization
"""

import os
import sys
import numpy as np
from pathlib import Path

from stable_baselines3 import DQN, PPO
from environment.custom_env import SmartCropDiseaseEnv
from environment.rendering import EnvironmentVisualizer


def find_best_model(model_dir: str = 'models/dqn') -> tuple:
    """Find the best trained model based on metrics"""
    metrics_files = list(Path(model_dir).glob('metrics_*.json'))
    
    if not metrics_files:
        print(f"No metrics found in {model_dir}")
        return None, None
    
    import json
    best_metric = None
    best_file = None
    
    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            metric = json.load(f)
            if best_metric is None or metric['mean_episode_reward'] > best_metric['mean_episode_reward']:
                best_metric = metric
                best_file = metrics_file
    
    return best_file, best_metric


def run_best_dqn_agent(num_episodes: int = 5, render: bool = True):
    """Run the best DQN agent"""
    print("\n" + "="*70)
    print("SMART FARM - BEST DQN AGENT DEMONSTRATION")
    print("="*70)
    
    # Find best model
    best_file, best_metric = find_best_model('models/dqn')
    
    if best_file is None:
        print("No trained DQN model found. Please train models first.")
        return
    
    print(f"\nBest DQN Model: {best_metric['experiment_name']}")
    print(f"Mean Reward: {best_metric['mean_episode_reward']:.2f}")
    print(f"Hyperparameters:")
    for key, value in best_metric['hyperparameters'].items():
        print(f"  {key}: {value}")
    
    # Load model
    model_path = str(best_file).replace('metrics_', 'dqn_').replace('.json', '')
    model = DQN.load(model_path)
    print(f"\nLoaded model from: {model_path}")
    
    # Create environment
    env = SmartCropDiseaseEnv(grid_size=5, episode_length=100, render_mode="human")
    if render:
        visualizer = EnvironmentVisualizer(env, grid_size=5, render_fps=10)
    
    action_names = ["Monitor", "Fungicide", "Neem Oil", "Remove", "Ventilation", "Irrigation"]
    
    all_episode_rewards = []
    all_episode_costs = []
    all_max_healthy = []
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_cost = 0
        max_healthy = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 70)
        
        for step in range(100):
            # Get action from trained model
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)  # Convert numpy array to int
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_cost += info['action_cost']
            max_healthy = max(max_healthy, info['healthy_plants'])
            
            if render:
                visualizer.render_step(action, reward, info)
            
            # Print verbose output
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1:3d} | Action: {action_names[action]:12s} | "
                      f"Reward: {reward:7.2f} | Healthy: {info['healthy_plants']:2d} | "
                      f"Infected: {info['infected_plants']:2d}")
            
            if terminated or truncated:
                break
        
        all_episode_rewards.append(episode_reward)
        all_episode_costs.append(episode_cost)
        all_max_healthy.append(max_healthy)
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Total Cost: ${episode_cost:.2f}")
        print(f"  Max Healthy Plants: {max_healthy}/25")
    
    if render:
        visualizer.close()
    
    env.close()
    
    print("\n" + "="*70)
    print("DQN AGENT PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Average Episode Reward: {np.mean(all_episode_rewards):.2f} ± {np.std(all_episode_rewards):.2f}")
    print(f"Average Total Cost: ${np.mean(all_episode_costs):.2f}")
    print(f"Average Max Healthy Plants: {np.mean(all_max_healthy):.1f}/25")
    print("="*70)


def run_best_ppo_agent(num_episodes: int = 5, render: bool = True):
    """Run the best PPO agent"""
    print("\n" + "="*70)
    print("SMART FARM - BEST PPO AGENT DEMONSTRATION")
    print("="*70)
    
    # Find best model
    best_file, best_metric = find_best_model('models/pg')
    
    if best_file is None:
        print("No trained PPO model found. Please train models first.")
        return
    
    # Filter for PPO
    for file in Path('models/pg').glob('metrics_ppo_*.json'):
        import json
        with open(file, 'r') as f:
            metric = json.load(f)
            if best_file is None or metric['mean_episode_reward'] > best_metric['mean_episode_reward']:
                best_metric = metric
                best_file = file
    
    print(f"\nBest PPO Model: {best_metric['experiment_name']}")
    print(f"Mean Reward: {best_metric['mean_episode_reward']:.2f}")
    print(f"Hyperparameters:")
    for key, value in best_metric['hyperparameters'].items():
        print(f"  {key}: {value}")
    
    # Load model
    model_path = str(best_file).replace('metrics_', '').replace('.json', '')
    model = PPO.load(model_path)
    print(f"\nLoaded model from: {model_path}")
    
    # Create environment
    env = SmartCropDiseaseEnv(grid_size=5, episode_length=100, render_mode="human")
    if render:
        visualizer = EnvironmentVisualizer(env, grid_size=5, render_fps=10)
    
    action_names = ["Monitor", "Fungicide", "Neem Oil", "Remove", "Ventilation", "Irrigation"]
    
    all_episode_rewards = []
    all_episode_costs = []
    all_max_healthy = []
    
    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_cost = 0
        max_healthy = 0
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 70)
        
        for step in range(100):
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)  # Convert numpy array to int
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_cost += info['action_cost']
            max_healthy = max(max_healthy, info['healthy_plants'])
            
            if render:
                visualizer.render_step(action, reward, info)
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1:3d} | Action: {action_names[action]:12s} | "
                      f"Reward: {reward:7.2f} | Healthy: {info['healthy_plants']:2d} | "
                      f"Infected: {info['infected_plants']:2d}")
            
            if terminated or truncated:
                break
        
        all_episode_rewards.append(episode_reward)
        all_episode_costs.append(episode_cost)
        all_max_healthy.append(max_healthy)
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Total Cost: ${episode_cost:.2f}")
        print(f"  Max Healthy Plants: {max_healthy}/25")
    
    if render:
        visualizer.close()
    
    env.close()
    
    print("\n" + "="*70)
    print("PPO AGENT PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Average Episode Reward: {np.mean(all_episode_rewards):.2f} ± {np.std(all_episode_rewards):.2f}")
    print(f"Average Total Cost: ${np.mean(all_episode_costs):.2f}")
    print(f"Average Max Healthy Plants: {np.mean(all_max_healthy):.1f}/25")
    print("="*70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trained RL agents for Smart Farm")
    parser.add_argument('--algorithm', type=str, default='dqn', 
                       choices=['dqn', 'ppo', 'all'],
                       help='Which algorithm to run')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    if args.algorithm in ['dqn', 'all']:
        run_best_dqn_agent(num_episodes=args.episodes, render=not args.no_render)
    
    if args.algorithm in ['ppo', 'all']:
        run_best_ppo_agent(num_episodes=args.episodes, render=not args.no_render)


if __name__ == "__main__":
    main()
