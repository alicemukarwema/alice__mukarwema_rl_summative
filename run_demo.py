#!/usr/bin/env python3
"""
Smart Farm RL - Interactive Demo Menu
Choose which test/demo to run
"""

import subprocess
import sys
from pathlib import Path

def print_header():
    print("\n" + "="*80)
    print("SMART FARM RL - INTERACTIVE DEMO MENU")
    print("="*80)

def print_menu():
    print("\n🎯 Choose what to run:\n")
    print("1. Quick Test (2-3 min)      - Verify environment & quick training")
    print("2. Run DQN Demo (1 min)      - Watch trained DQN agent (5 episodes)")
    print("3. Run PPO Demo (1 min)      - Watch trained PPO agent (5 episodes)")
    print("4. Run Random Actions (1 min)- Baseline: random agent behavior")
    print("5. Full Training (30-40 min) - Train all 20 agents (10 DQN + 10 PPO)")
    print("6. View Analysis Results     - Check if analysis files exist")
    print("7. Check Logs & Models       - List saved models and logs")
    print("0. Exit")
    print()

def run_quick_test():
    print("\n" + "="*80)
    print("Running Quick Test...")
    print("="*80)
    subprocess.run([sys.executable, "quick_test.py"])

def run_dqn_demo():
    print("\n" + "="*80)
    print("Running DQN Agent Demo (5 episodes)...")
    print("="*80)
    subprocess.run([sys.executable, "main.py", "--algorithm", "dqn", "--episodes", "5"])

def run_ppo_demo():
    print("\n" + "="*80)
    print("Running PPO Agent Demo (5 episodes)...")
    print("="*80)
    subprocess.run([sys.executable, "main.py", "--algorithm", "ppo", "--episodes", "5"])

def run_random_demo():
    print("\n" + "="*80)
    print("Running Random Agent Demo...")
    print("="*80)
    subprocess.run([sys.executable, "demo_random_actions.py"])

def run_full_training():
    print("\n" + "="*80)
    print("⚠️  STARTING FULL TRAINING (30-40 minutes)")
    print("="*80)
    print("This will train 20 agents:")
    print("  - 10 DQN agents (value-based)")
    print("  - 10 PPO agents (policy gradient)")
    print("")
    response = input("Continue? (yes/no): ").strip().lower()
    if response == "yes":
        subprocess.run([sys.executable, "train_all.py"])
    else:
        print("Cancelled.")

def check_analysis():
    print("\n" + "="*80)
    print("📊 Analysis Files Status")
    print("="*80)
    
    analysis_dir = Path("analysis")
    if not analysis_dir.exists():
        print("❌ analysis/ directory does not exist (run training first)")
        return
    
    files = list(analysis_dir.glob("*"))
    if not files:
        print("❌ No analysis files found (run training first)")
        return
    
    print(f"✅ Found {len(files)} files in analysis/:\n")
    for f in sorted(files):
        size = f.stat().st_size if f.is_file() else "-"
        print(f"  - {f.name}")
    
    print("\n📖 To view HTML summary, open:")
    print("  analysis/summary.html")

def check_logs_models():
    print("\n" + "="*80)
    print("📦 Models & Logs Status")
    print("="*80)
    
    # Check models
    dqn_models = list(Path("models/dqn").glob("dqn_*.zip"))
    pg_models = list(Path("models/pg").glob("ppo_*.zip"))
    
    print(f"\n✅ DQN Models: {len(dqn_models)} found")
    for m in sorted(dqn_models)[:5]:  # Show first 5
        print(f"  - {m.name}")
    if len(dqn_models) > 5:
        print(f"  ... and {len(dqn_models) - 5} more")
    
    print(f"\n✅ PPO Models: {len(pg_models)} found")
    for m in sorted(pg_models)[:5]:  # Show first 5
        print(f"  - {m.name}")
    if len(pg_models) > 5:
        print(f"  ... and {len(pg_models) - 5} more")
    
    # Check logs
    log_files = list(Path("logs").glob("*.json"))
    print(f"\n✅ Log Files: {len(log_files)} found")
    for l in sorted(log_files)[:5]:
        print(f"  - {l.name}")
    if len(log_files) > 5:
        print(f"  ... and {len(log_files) - 5} more")

def main():
    print_header()
    
    while True:
        print_menu()
        choice = input("Enter your choice (0-7): ").strip()
        
        if choice == "1":
            run_quick_test()
        elif choice == "2":
            run_dqn_demo()
        elif choice == "3":
            run_ppo_demo()
        elif choice == "4":
            run_random_demo()
        elif choice == "5":
            run_full_training()
        elif choice == "6":
            check_analysis()
        elif choice == "7":
            check_logs_models()
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
