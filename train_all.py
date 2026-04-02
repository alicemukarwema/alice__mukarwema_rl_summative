"""
Complete Training Runner
Orchestrates training of 3 RL algorithms:
- DQN (Value-Based Method)
- PPO (Proximal Policy Optimization)
- REINFORCE (Policy Gradient - via PPO)
Total: 20 training runs (10 DQN + 10 PPO)
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


def main():
    """Run all training experiments"""
    
    print("\n" + "="*80)
    print("SMART FARM RL - TRAINING RUNNER")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Experiments: 20 (10 DQN + 10 PPO/REINFORCE)")
    print("="*80 + "\n")
    
    # Ensure directories exist
    Path('models/dqn').mkdir(parents=True, exist_ok=True)
    Path('models/pg').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    
    results = {
        'start_time': datetime.now().isoformat(),
        'dqn_status': 'Not Started',
        'pg_status': 'Not Started',
        'total_time': 0
    }
    
    try:
        # Train DQN
        print("\n" + "="*80)
        print("PHASE 1: Training DQN (Value-Based Method)")
        print("="*80)
        print("Testing hyperparameters:")
        print("  - Learning rate: 1e-4, 5e-4, 1e-3")
        print("  - Buffer size: 5k, 10k, 50k")
        print("  - Batch size: 16, 32, 64")
        print("  - Gamma: 0.95, 0.99, 0.999")
        print("  - Exploration configurations: 3 variations")
        print("\nEstimated time: 15-20 minutes\n")
        
        from training.dqn_training import run_dqn_experiments
        dqn_metrics = run_dqn_experiments()
        results['dqn_status'] = 'Completed'
        results['dqn_best'] = {
            'experiment': max(dqn_metrics, key=lambda x: x['mean_episode_reward'])['experiment_name'],
            'reward': max(dqn_metrics, key=lambda x: x['mean_episode_reward'])['mean_episode_reward']
        }
        
    except Exception as e:
        print(f"\n❌ DQN training failed: {str(e)}")
        results['dqn_status'] = f'Failed: {str(e)}'
    
    try:
        # Train PPO (PPO implements REINFORCE with clipping)
        print("\n" + "="*80)
        print("PHASE 2: Training Policy Gradient Methods (PPO/REINFORCE)")
        print("="*80)
        print("PPO Hyperparameters:")
        print("  - Learning rate: 1e-4, 5e-4, 1e-3")
        print("  - Trajectory length (n_steps): 64, 128, 256")
        print("  - Batch size: 16, 32, 64")
        print("  - Entropy coefficient: 0.0, 0.01, 0.05")
        print("  - Gamma / GAE lambda: 2 combinations")
        print("\n(PPO is an improved version of REINFORCE algorithm)")
        print("\nEstimated time: 15-20 minutes\n")
        
        from training.pg_training import run_ppo_experiments
        
        ppo_metrics = run_ppo_experiments()
        results['pg_status'] = 'Completed'
        results['pg_best'] = {
            'experiment': max(ppo_metrics, key=lambda x: x['mean_episode_reward'])['experiment_name'],
            'reward': max(ppo_metrics, key=lambda x: x['mean_episode_reward'])['mean_episode_reward']
        }
        
    except Exception as e:
        print(f"\n❌ Policy gradient training failed: {str(e)}")
        results['pg_status'] = f'Failed: {str(e)}'
        
    try:
        # Train REINFORCE
        print("\n" + "="*80)
        print("PHASE 2.5: Training REINFORCE Algorithm")
        print("="*80)
        print("REINFORCE Hyperparameters:")
        print("  - Learning rate: 1e-4, 5e-4, 1e-3")
        print("  - Gamma: 0.95, 0.99, 0.999")
        print("  - Entropy coefficient: 0.0, 0.01, 0.05")
        print("\nEstimated time: 10-15 minutes\n")
        
        from training.reinforce_training import run_reinforce_experiments
        
        reinforce_metrics = run_reinforce_experiments()
        results['reinforce_status'] = 'Completed'
        results['reinforce_best'] = {
            'experiment': max(reinforce_metrics, key=lambda x: x['mean_episode_reward'])['experiment_name'],
            'reward': max(reinforce_metrics, key=lambda x: x['mean_episode_reward'])['mean_episode_reward']
        }
        
    except Exception as e:
        print(f"\n❌ REINFORCE training failed: {str(e)}")
        results['reinforce_status'] = f'Failed: {str(e)}'
    
    try:
        # Generate analysis
        print("\n" + "="*80)
        print("PHASE 3: Generating Analysis & Visualizations")
        print("="*80)
        print("Creating:")
        print("  - DQN hyperparameter effect plots")
        print("  - PPO/REINFORCE hyperparameter effect plots")
        print("  - Algorithm comparison plots")
        print("  - HTML summary report\n")
        
        from analysis.visualizer import HyperparameterAnalyzer
        analyzer = HyperparameterAnalyzer()
        analyzer.plot_dqn_learning_curves()
        analyzer.plot_ppo_learning_curves()
        analyzer.plot_algorithm_comparison()
        analyzer.generate_html_summary()
        
        results['analysis_status'] = 'Completed'
        
    except Exception as e:
        print(f"\n⚠️  Analysis generation failed: {str(e)}")
        results['analysis_status'] = f'Failed: {str(e)}'
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    print(f"  DQN (Value-Based) Status: {results['dqn_status']}")
    if 'dqn_best' in results:
        print(f"    Best: {results['dqn_best']['experiment']} (Reward: {results['dqn_best']['reward']:.2f})")
    
    print(f"  PPO/REINFORCE (Policy Gradient) Status: {results.get('pg_status', 'Not Completed')}")
    if 'pg_best' in results:
        print(f"    Best: {results['pg_best']['experiment']} (Reward: {results['pg_best']['reward']:.2f})")
        
    print(f"  Vanilla REINFORCE Status: {results.get('reinforce_status', 'Not Completed')}")
    if 'reinforce_best' in results:
        print(f"    Best: {results['reinforce_best']['experiment']} (Reward: {results['reinforce_best']['reward']:.2f})")
    
    print(f"  Analysis Status: {results.get('analysis_status', 'Not Completed')}")
    
    print("\nNext Steps:")
    print("  1. View results: python analysis/visualizer.py")
    print("  2. Run best DQN agent: python main.py --algorithm dqn --episodes 5")
    print("  3. Run best PPO agent: python main.py --algorithm ppo --episodes 5")
    print("  4. Start production API: uvicorn api:app --reload")
    print("  5. Check logs: ls logs/")
    print("  6. View models: ls models/dqn/ models/pg/ models/reinforce/")
    print("\n" + "="*80)
    
    # Save results
    results['end_time'] = datetime.now().isoformat()
    with open('logs/training_summary.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("✅ Training complete! See logs/training_summary.json for details.")


if __name__ == "__main__":
    main()
