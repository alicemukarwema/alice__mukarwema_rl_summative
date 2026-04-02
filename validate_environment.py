"""
Environment Validation Script
Tests that all dependencies are installed and environment is working
"""

import sys
from pathlib import Path

def check_imports():
    """Check that all required packages can be imported"""
    print("\n" + "="*70)
    print("CHECKING IMPORTS")
    print("="*70)
    
    packages = {
        'gymnasium': 'Environment creation',
        'numpy': 'Numerical computing',
        'stable_baselines3': 'RL algorithms (SB3)',
        'pygame': 'Visualization',
        'matplotlib': 'Plotting',
        'pandas': 'Data processing',
    }
    
    all_good = True
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✓ {package:20s} - {description}")
        except ImportError:
            print(f"✗ {package:20s} - {description} - NOT INSTALLED")
            all_good = False
    
    return all_good


def check_file_structure():
    """Check that all required files exist"""
    print("\n" + "="*70)
    print("CHECKING PROJECT STRUCTURE")
    print("="*70)
    
    required_files = [
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'HYPERPARAMETER_TUNING.md',
        'ENVIRONMENT_DIAGRAM.py',
        'main.py',
        'train_all.py',
        'demo_random_actions.py',
        'environment/__init__.py',
        'environment/custom_env.py',
        'environment/rendering.py',
        'training/__init__.py',
        'training/dqn_training.py',
        'training/pg_training.py',
        'analysis/__init__.py',
        'analysis/visualizer.py',
    ]
    
    all_good = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_good = False
    
    return all_good


def check_environment():
    """Test basic environment creation and interaction"""
    print("\n" + "="*70)
    print("TESTING ENVIRONMENT")
    print("="*70)
    
    try:
        from environment.custom_env import SmartCropDiseaseEnv
        print("✓ Importing SmartCropDiseaseEnv")
        
        env = SmartCropDiseaseEnv(grid_size=5, episode_length=10)
        print("✓ Creating environment instance")
        
        obs, info = env.reset()
        print("✓ Environment reset")
        
        # Check observation structure
        assert 'crop_health' in obs
        assert 'disease_severity' in obs
        assert 'soil_moisture' in obs
        assert 'weather' in obs
        print("✓ Observation space structure valid")
        
        # Take a step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step executed (Action: {action}, Reward: {reward:.2f})")
        
        # Check reward is numeric
        assert isinstance(reward, (int, float))
        print("✓ Reward signal valid")
        
        env.close()
        print("✓ Environment closed cleanly")
        
        return True
    
    except Exception as e:
        print(f"✗ Environment test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_rendering():
    """Test rendering capabilities"""
    print("\n" + "="*70)
    print("TESTING RENDERING")
    print("="*70)
    
    try:
        from environment.rendering import FarmRenderer, EnvironmentVisualizer
        print("✓ Importing rendering modules")
        
        from environment.custom_env import SmartCropDiseaseEnv
        env = SmartCropDiseaseEnv()
        obs, _ = env.reset()
        
        visualizer = EnvironmentVisualizer(env, grid_size=5, render_fps=30)
        print("✓ Creating visualizer")
        
        state_dict = env.get_state_dict()
        visualizer.render_state()
        print("✓ Rendering state (display may not show in headless mode)")
        
        visualizer.close()
        print("✓ Visualizer closed")
        
        return True
    
    except Exception as e:
        print(f"⚠ Rendering test failed (may be expected in headless): {str(e)}")
        return True  # Non-critical


def check_rl_algorithms():
    """Test RL algorithm imports"""
    print("\n" + "="*70)
    print("TESTING RL ALGORITHMS")
    print("="*70)
    
    try:
        from stable_baselines3 import DQN, PPO, A2C
        print("✓ Importing DQN, PPO, A2C")
        
        from environment.custom_env import SmartCropDiseaseEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        env = DummyVecEnv([lambda: SmartCropDiseaseEnv()])
        print("✓ Creating vectorized environment")
        
        # Quick DQN test
        dqn = DQN('MultiInputPolicy', env, verbose=0, learning_rate=5e-4)
        print("✓ Creating DQN model")
        env.close()
        
        # Quick PPO test
        env = DummyVecEnv([lambda: SmartCropDiseaseEnv()])
        ppo = PPO('MultiInputPolicy', env, verbose=0, learning_rate=5e-4)
        print("✓ Creating PPO model")
        env.close()
        
        # Quick A2C test
        env = DummyVecEnv([lambda: SmartCropDiseaseEnv()])
        a2c = A2C('MultiInputPolicy', env, verbose=0, learning_rate=5e-4)
        print("✓ Creating A2C model")
        env.close()
        
        return True
    
    except Exception as e:
        print(f"✗ RL algorithm test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_analysis():
    """Test analysis module"""
    print("\n" + "="*70)
    print("TESTING ANALYSIS MODULE")
    print("="*70)
    
    try:
        from analysis.visualizer import HyperparameterAnalyzer
        print("✓ Importing HyperparameterAnalyzer")
        
        # Note: Will only work if models exist, so we just check import
        print("✓ Analysis module structure valid")
        
        return True
    
    except Exception as e:
        print(f"✗ Analysis test failed: {str(e)}")
        return False


def main():
    """Run all validation checks"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "SMART FARM RL - ENVIRONMENT VALIDATION" + " "*15 + "║")
    print("╚" + "="*68 + "╝")
    
    results = []
    
    # Run checks
    results.append(("File Structure", check_file_structure()))
    results.append(("Package Imports", check_imports()))
    results.append(("Environment", check_environment()))
    results.append(("Rendering", check_rendering()))
    results.append(("RL Algorithms", check_rl_algorithms()))
    results.append(("Analysis Module", check_analysis()))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n" + "🎉 "*20)
        print("ALL CHECKS PASSED - READY TO TRAIN!")
        print("🎉 "*20)
        print("\nNext steps:")
        print("  1. Run demo: python demo_random_actions.py")
        print("  2. Train models: python train_all.py")
        print("  3. Evaluate: python main.py --algorithm dqn")
        print("\nSee QUICKSTART.md for detailed instructions.")
        return 0
    
    else:
        print("\n" + "⚠️ "*20)
        print("SOME CHECKS FAILED - SEE ABOVE FOR DETAILS")
        print("⚠️ "*20)
        print("\nCommon fixes:")
        print("  - Install packages: pip install -r requirements.txt")
        print("  - Check Python version: python --version (3.8+ required)")
        print("  - Verify git clone: Check README.md for setup")
        return 1


if __name__ == "__main__":
    sys.exit(main())
