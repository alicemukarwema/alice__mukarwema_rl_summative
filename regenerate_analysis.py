#!/usr/bin/env python3
"""
Quick PPO Analysis Regenerator
Regenerates PPO curves and analysis plots fast
"""

from analysis.visualizer import HyperparameterAnalyzer
import sys

print("="*80)
print("REGENERATING PPO ANALYSIS WITH LATEST DATA")
print("="*80)

try:
    analyzer = HyperparameterAnalyzer()
    
    print(f"\n✓ DQN experiments found: {len(analyzer.dqn_metrics)}")
    print(f"✓ PPO experiments found: {len(analyzer.ppo_metrics)}")
    
    if not analyzer.ppo_metrics:
        print("\n❌ No PPO metrics found!")
        sys.exit(1)
    
    print("\n📊 Regenerating PPO hyperparameter curves...")
    analyzer.plot_ppo_learning_curves()
    
    print("📊 Regenerating algorithm comparison...")
    analyzer.plot_algorithm_comparison()
    
    print("📊 Regenerating HTML summary...")
    analyzer.generate_html_summary()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS REGENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nFiles updated:")
    print("  - analysis/ppo_curves.png")
    print("  - analysis/algorithm_comparison.png")
    print("  - analysis/summary.html")
    print("\nOpen analysis/summary.html in your browser to view results!")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
