#!/usr/bin/env python3
"""
Quick test of the RL training pipeline
Tests: Environment, DQN, PPO without heavy computation
"""

import sys
import numpy as np

print("\n" + "="*80)
print("SMART FARM RL - QUICK TEST")
print("="*80 + "\n")

# Test 1: Environment
print("✓ TEST 1: Loading Custom Environment...")
try:
    from environment.custom_env import SmartCropDiseaseEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    env = DummyVecEnv([lambda: SmartCropDiseaseEnv(grid_size=3, episode_length=10)])
    obs = env.reset()
    print("  ✅ Environment loaded and reset successfully!")
    print(f"  - Observation type: {type(obs)}")
    
    # Get action and step
    action = np.array([env.action_space.sample()])
    result = env.step(action)
    print(f"  - Step executed successfully (4 returns)")
    env.close()
except Exception as e:
    print(f"  ❌ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: DQN
print("\n✓ TEST 2: Training DQN (100 steps)...")
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from environment.custom_env import SmartCropDiseaseEnv
    
    env = DummyVecEnv([lambda: SmartCropDiseaseEnv(grid_size=3, episode_length=10)])
    model = DQN('MultiInputPolicy', env, learning_rate=5e-4, verbose=0)
    model.learn(total_timesteps=100)
    print("  ✅ DQN training successful!")
    print(f"  - Model trained for 100 timesteps")
    env.close()
except Exception as e:
    print(f"  ❌ DQN test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: PPO
print("\n✓ TEST 3: Training PPO (100 steps)...")
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from environment.custom_env import SmartCropDiseaseEnv
    
    env = DummyVecEnv([lambda: SmartCropDiseaseEnv(grid_size=3, episode_length=10)])
    model = PPO('MultiInputPolicy', env, learning_rate=5e-4, n_steps=64, verbose=0)
    model.learn(total_timesteps=100)
    print("  ✅ PPO training successful!")
    print(f"  - Model trained for 100 timesteps")
    env.close()
except Exception as e:
    print(f"  ❌ PPO test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nYou can now run:")
print("  python train_all.py      # Full training")
print("  python demo_random_actions.py  # Quick demo")
print("  python main.py --algorithm dqn --episodes 5  # Run best DQN")
print("="*80 + "\n")
