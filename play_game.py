import pygame
import sys
import numpy as np
from environment.custom_env import SmartCropDiseaseEnv

def play_game():
    pygame.init()
    
    env = SmartCropDiseaseEnv(grid_size=5, episode_length=100, render_mode="human")
    from environment.rendering import FarmRenderer
    renderer = FarmRenderer(grid_size=5, fps=10)
    
    observation, _ = env.reset()
    
    action_names = {
        pygame.K_0: ("Monitor", 0),
        pygame.K_1: ("Fungicide", 1),
        pygame.K_2: ("Neem Oil", 2),
        pygame.K_3: ("Remove", 3),
        pygame.K_4: ("Ventilation", 4),
        pygame.K_5: ("Irrigation", 5),
    }

    print("\n" + "="*70)
    print("🌾 SMART FARM - INTERACTIVE GAME MODE 🌾")
    print("="*70)
    print("Welcome to Smart Farm Disease Management!")
    print("You are now the agent. Save your crops from the disease!")
    print("\nControls:")
    print("  0 : Do Nothing (Monitor Farm)")
    print("  1 : Apply Chemical Fungicide (Fast & effective, but expensive)")
    print("  2 : Apply Neem Oil (Organic, cheaper, but slower)")
    print("  3 : Remove Infected Plants (Stop spread entirely, high crop loss)")
    print("  4 : Improve Farm Ventilation (Lower humidity to prevent fungus)")
    print("  5 : Optimize Irrigation (Prevent overwatering)")
    print("  Q / ESC : Quit Game")
    print("="*70)
    
    episode_reward = 0
    step = 0
    running = True
    
    # Render initial state
    renderer.render(env.get_state_dict(), None, None, None)
    
    while running:
        action_taken = False
        action = None
        
        # Wait for user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                elif event.key in action_names:
                    action_name, action = action_names[event.key]
                    action_taken = True
        
        if action_taken and action is not None:
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            print(f"Step {step} | Action: {action_names[event.key][0]} | Reward: {reward:.2f} | Healthy: {info['healthy_plants']} | Infected: {info['infected_plants']}")
            
            renderer.render(env.get_state_dict(), action, reward, info)
            
            if terminated or truncated:
                print("\n" + "="*70)
                print(f"GAME OVER! Final Score: {episode_reward:.2f}")
                print("="*70)
                print("Press R to restart or Q/ESC to quit.")
                
                # Wait for restart or quit
                waiting = True
                while waiting and running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                                running = False
                                waiting = False
                            elif event.key == pygame.K_r:
                                env.reset()
                                episode_reward = 0
                                step = 0
                                renderer.render(env.get_state_dict(), None, None, None)
                                print("\n--- GAME RESTARTED ---")
                                waiting = False
        
        renderer.clock.tick(30)

    renderer.close()
    env.close()

if __name__ == "__main__":
    play_game()