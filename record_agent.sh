#!/bin/bash
# 
# Quick Record Script - Run Your Professional RL Agent Demo for Video Recording
# This script activates the virtual environment and runs the enhanced agent visualization
#

cd /home/alice/Documents/ML\ ALU/Summative_jan2026/RL\ agent\ development\ /alice__mukarwema_rl_summative

# Activate virtual environment
source .venv/bin/activate

# Run the professional PPO agent demo for presentation (5 episodes for good video length)
# This will show the enhanced Pygame visualization with smooth colors and animations
python3 demo_professional.py --algorithm ppo --episodes 5
