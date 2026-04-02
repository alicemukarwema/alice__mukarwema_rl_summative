#!/usr/bin/env python3
"""
Enhanced Game-Style Visualization for Smart Farm RL Agent
Shows the trained agent managing the farm in an interactive, game-like display
"""

import pygame
import numpy as np
from environment.custom_env import SmartCropDiseaseEnv
from environment.rendering import EnvironmentVisualizer
from stable_baselines3 import PPO, DQN


def run_game_demonstration(algorithm: str = 'ppo', episodes: int = 3):
    """
    Run the RL agent with enhanced game-style visualization.
    
    Args:
        algorithm: 'ppo' or 'dqn'
        episodes: Number of episodes to run
    """
    
    print("=" * 80)
    print("🎮 SMART FARM DISEASE MANAGEMENT - GAME VISUALIZATION")
    print("=" * 80)
    print(f"\n🤖 Loading trained {algorithm.upper()} agent...")
    
    # Create environment
    env = SmartCropDiseaseEnv(grid_size=5, episode_length=100, render_mode="human")
    
    # Load trained model
    if algorithm.lower() == 'ppo':
        model = PPO.load('models/pg/ppo_exp_08')  # Best PPO: reward 328.00
        print("✅ Loaded Best PPO Model (Exp 08)")
        print("   Mean Reward: 328.00")
        print("   Configuration: LR=0.0005, n_steps=128, Batch=32, Ent_coef=0.0")
    else:
        model = DQN.load('models/dqn/dqn_exp_06')  # Best DQN: reward 315.46
        print("✅ Loaded Best DQN Model (Exp 06)")
        print("   Mean Reward: 315.46")
        print("   Configuration: LR=0.0005, Buffer=10K, Batch=16, Gamma=0.99")
    
    # Create visualizer
    visualizer = EnvironmentVisualizer(env, grid_size=5, render_fps=10)
    
    action_names = ["Monitor", "Fungicide", "Neem Oil", "Remove", "Ventilation", "Irrigation"]
    action_colors = {
        0: "🔍",  # Monitor
        1: "💉",  # Fungicide
        2: "🌿",  # Neem Oil
        3: "✂️",   # Remove
        4: "💨",  # Ventilation
        5: "💧",  # Irrigation
    }
    
    all_episode_rewards = []
    all_episode_costs = []
    all_max_healthy = []
    
    print(f"\n🎮 Starting game with {episodes} episodes...\n")
    print("=" * 80)
    
    for episode in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0
        episode_cost = 0
        max_healthy = 0
        
        print(f"\n🎮 EPISODE {episode + 1}/{episodes}")
        print("=" * 80)
        print(f"Initial State: 🟢 Healthy plants | 🔴 Infected plants | 💰 Cost tracking")
        print("-" * 80)
        
        for step in range(100):
            # Get action from trained model
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_cost += info['action_cost']
            max_healthy = max(max_healthy, info['healthy_plants'])
            
            # Render the game
            visualizer.render_step(action, reward, info)
            
            # Print game-style output
            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1:3d} | {action_colors.get(action, '❓')} {action_names[action]:12s} | "
                      f"💰 ${info['action_cost']:5.2f} | "
                      f"🟢 {info['healthy_plants']:2d} | "
                      f"🔴 {info['infected_plants']:2d} | "
                      f"⭐ {reward:7.2f}")
            
            if terminated or truncated:
                print(f"\n⚠️  Episode ended at step {step + 1}")
                break
        
        all_episode_rewards.append(episode_reward)
        all_episode_costs.append(episode_cost)
        all_max_healthy.append(max_healthy)
        
        print("\n" + "-" * 80)
        print(f"📊 EPISODE {episode + 1} SUMMARY")
        print("-" * 80)
        print(f"  ⭐ Total Reward:      {episode_reward:8.2f}")
        print(f"  💰 Total Cost:        ${episode_cost:8.2f}")
        print(f"  🟢 Max Healthy:       {max_healthy:8}/25")
        print(f"  🎯 Efficiency:        {(max_healthy/25)*100:8.1f}% healthy maintained")
    
    # Close visualizer
    visualizer.close()
    env.close()
    
    # Show final summary
    print("\n" + "=" * 80)
    print(f"🏆 GAME COMPLETE - {algorithm.upper()} AGENT PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"📈 Episodes Played:              {episodes}")
    print(f"⭐ Average Episode Reward:       {np.mean(all_episode_rewards):8.2f} ± {np.std(all_episode_rewards):.2f}")
    print(f"💰 Average Total Cost:           ${np.mean(all_episode_costs):8.2f}")
    print(f"🟢 Average Max Healthy Plants:   {np.mean(all_max_healthy):8.1f}/25")
    print(f"🎯 Average Efficiency:           {(np.mean(all_max_healthy)/25)*100:8.1f}%")
    print("=" * 80)
    
    if algorithm.lower() == 'ppo':
        print("\n✨ PPO Agent Summary:")
        print("  • Uses Policy Optimization with PPO algorithm")
        print("  • Best performing configuration: entropy_coef = 0.0")
        print("  • Stable and consistent reward: 328.00")
        print("  • Excellent disease management strategy")
    else:
        print("\n✨ DQN Agent Summary:")
        print("  • Uses Deep Q-Network for value-based learning")
        print("  • Optimal batch size: 16 (small batches better)")
        print("  • Strong performance: 315.46 reward")
        print("  • Good balance between exploration and exploitation")
    
    print("\n🎮 Thanks for watching the Smart Farm Game Demo! 🌾")
    print("=" * 80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Farm RL Agent Game Demo")
    parser.add_argument('--algorithm', type=str, default='ppo', 
                       choices=['ppo', 'dqn'],
                       help='Algorithm to demonstrate (default: ppo)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to play (default: 3)')
    
    args = parser.parse_args()
    
    run_game_demonstration(algorithm=args.algorithm, episodes=args.episodes)


if __name__ == "__main__":
    main()
