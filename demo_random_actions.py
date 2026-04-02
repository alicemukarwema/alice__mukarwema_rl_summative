"""
Static Visualization Demo
Demonstrates the environment with random actions (no RL training).
This shows the environment can be visualized and interactions work correctly.
"""

import numpy as np
from environment.custom_env import SmartCropDiseaseEnv
from environment.rendering import EnvironmentVisualizer
import time


def main():
    """Run static visualization with random actions"""
    print("=" * 70)
    print("SMART FARM DISEASE MANAGEMENT - RANDOM ACTION DEMO")
    print("=" * 70)
    print("\nEnvironment: A 5x5 farm with crops susceptible to disease spread")
    print("Objective: Manage disease spread to maximize healthy crops")
    print("\nActions:")
    print("  0 - Monitor (do nothing)")
    print("  1 - Spray Fungicide (cost=5, effectiveness=70%)")
    print("  2 - Apply Neem Oil (cost=2, effectiveness=50%)")
    print("  3 - Remove Infected Plants (cost=10 per plant)")
    print("  4 - Improve Ventilation (cost=1, reduce humidity)")
    print("  5 - Adjust Irrigation (cost=0.5, optimal moisture)")
    print("\n" + "=" * 70)
    print("Starting random action simulation...")
    print("=" * 70 + "\n")
    
    # Create environment
    env = SmartCropDiseaseEnv(grid_size=5, episode_length=50, render_mode="human")
    
    # Create visualizer
    visualizer = EnvironmentVisualizer(env, grid_size=5, render_fps=5)
    
    # Reset environment
    observation, info = env.reset()
    print(f"Episode started. Initial state:")
    print(f"  Healthy plants: {np.sum(observation['crop_health'] > 70)}/25")
    print(f"  Infected plants: {np.sum(observation['disease_severity'] > 30)}")
    print()
    
    action_names = ["Monitor", "Fungicide", "Neem Oil", "Remove", "Ventilation", "Irrigation"]
    
    # Run episode with random actions
    total_reward = 0
    max_health_plants = 0
    total_cost = 0
    
    for step in range(50):
        # Take random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_cost += info['action_cost']
        max_health_plants = max(max_health_plants, info['healthy_plants'])
        
        # Render
        visualizer.render_step(action, reward, info)
        
        # Print step info
        if (step + 1) % 5 == 0:
            print(f"Step {step + 1:3d} | Action: {action_names[action]:12s} | "
                  f"Reward: {reward:7.2f} | Cost: ${info['action_cost']:.1f} | "
                  f"Healthy: {info['healthy_plants']:2d} | "
                  f"Infected: {info['infected_plants']:2d}")
        
        if terminated:
            print(f"\nEpisode terminated at step {step + 1}")
            break
    
    print("\n" + "=" * 70)
    print("EPISODE SUMMARY")
    print("=" * 70)
    print(f"Total Steps: {step + 1}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Cost: ${total_cost:.2f}")
    print(f"Max Healthy Plants: {max_health_plants}/25")
    print(f"Final Healthy Plants: {observation['crop_health'][observation['crop_health'] > 70].size}/25")
    print(f"Final Infected Plants: {np.sum(observation['disease_severity'] > 30)}")
    print("\nNote: This is a RANDOM ACTION demo to demonstrate environment mechanics.")
    print("      Trained RL agents will show much better performance!")
    print("=" * 70)
    
    visualizer.close()


if __name__ == "__main__":
    main()
