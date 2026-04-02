#!/usr/bin/env python3
"""
🎮 PROFESSIONAL DEMO - Smart Farm RL Agent
Enhanced visualization for impressive presentation
Shows the agent managing the farm with smooth rendering and real-time feedback
"""

import pygame
import numpy as np
from environment.custom_env import SmartCropDiseaseEnv
from stable_baselines3 import PPO, DQN
import math


class PopupScore:
    """Animated score popup with scaling and fading"""
    def __init__(self, x, y, value, color=(100, 255, 100)):
        self.x = x
        self.y = y
        self.value = value
        self.color = color
        self.lifetime = 0
        self.max_lifetime = 1.2  # 1.2 seconds
        self.alive = True
        
    def update(self, dt):
        self.lifetime += dt
        if self.lifetime >= self.max_lifetime:
            self.alive = False
    
    def draw(self, surface, font):
        if not self.alive:
            return
        
        # Smooth fade using easing function
        progress = self.lifetime / self.max_lifetime
        alpha = int(255 * (1 - progress * progress))  # Ease-out
        
        # Float up with acceleration
        float_distance = 40 * progress
        draw_y = self.y - float_distance
        
        # Scale text - start big, shrink
        scale = 1.2 - (progress * 0.2)
        
        # Draw text with color
        text = font.render(f"{self.value:+.1f}", True, self.color)
        
        # Scale the text
        if scale != 1.0:
            new_size = int(text.get_width() * scale)
            text = pygame.transform.scale(text, (new_size, int(text.get_height() * scale)))
        
        # Create surface with alpha
        text_surf = pygame.Surface((text.get_width(), text.get_height()))
        text_surf.fill((25, 25, 50))
        text_surf.blit(text, (0, 0))
        text_surf.set_alpha(alpha)
        
        surface.blit(text_surf, (int(self.x - text.get_width()/2), int(draw_y)))


class ActionEffect:
    """Visual effect when an action is taken"""
    def __init__(self, x, y, action_type):
        self.x = x
        self.y = y
        self.action_type = action_type  # 0-5
        self.lifetime = 0
        self.max_lifetime = 0.6
        self.alive = True
        
        # Action colors
        self.action_colors = [
            (150, 150, 200),  # Monitor - blue
            (255, 100, 100),  # Fungicide - red
            (100, 200, 100),  # Neem Oil - green
            (200, 100, 100),  # Remove - dark red
            (100, 200, 255),  # Ventilation - light blue
            (100, 180, 255)   # Irrigation - cyan
        ]
        
    def update(self, dt):
        self.lifetime += dt
        if self.lifetime >= self.max_lifetime:
            self.alive = False
    
    def draw(self, surface, cell_size):
        if not self.alive:
            return
        
        progress = self.lifetime / self.max_lifetime
        color = self.action_colors[self.action_type]
        
        # Expanding ring effect
        ring_radius = int(30 * progress)
        alpha = int(200 * (1 - progress))
        
        # Draw expanding circle
        if ring_radius > 0:
            s = pygame.Surface((ring_radius * 2, ring_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*color, alpha), (ring_radius, ring_radius), ring_radius, 3)
            surface.blit(s, (int(self.x - ring_radius), int(self.y - ring_radius)))


class ProfessionalFarmDemo:
    """Professional game-style renderer with animations"""
    
    def __init__(self, grid_size=5, fps=20):
        pygame.init()
        
        self.grid_size = grid_size
        self.fps = fps
        self.cell_size = 90
        self.margin = 30
        self.panel_width = 320
        
        # Dimensions
        self.game_width = grid_size * self.cell_size + 2 * self.margin
        self.game_height = grid_size * self.cell_size + 2 * self.margin
        self.screen_width = self.game_width + self.panel_width + 40
        self.screen_height = max(self.game_height + 40, 650)
        
        # Create window with icon
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("🌾 SMART FARM - RL AGENT DEMO 🎮")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_huge = pygame.font.Font(None, 48)
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Color scheme (professional game-style)
        self.BG_COLOR = (20, 20, 40)
        self.GRID_BG = (30, 30, 60)
        self.HEALTHY = (80, 220, 80)      # Bright green
        self.INFECTED = (255, 140, 60)     # Orange
        self.SEVERE = (220, 40, 40)        # Red
        self.REMOVED = (100, 100, 100)     # Gray
        self.BORDER = (150, 150, 180)
        self.TEXT = (255, 255, 255)
        self.ACCENT = (255, 200, 100)
        
        # Animation state
        self.popups = []
        self.action_effects = []
        self.is_open = True
        self.frame_count = 0
        
    def add_popup(self, x, y, value):
        """Add floating score animation"""
        color = (100, 255, 100) if value > 0 else (255, 100, 100)
        self.popups.append(PopupScore(x, y, value, color))
    
    def add_action_effect(self, x, y, action):
        """Add action effect animation"""
        self.action_effects.append(ActionEffect(x, y, action))
    
    def draw_plant_cell(self, x, y, health, disease, is_removed, pulse_factor=1.0):
        """Draw a single plant with smooth coloring and pulse effect"""
        cell_x = x
        cell_y = y
        cell_w = self.cell_size - 2
        cell_h = self.cell_size - 2
        
        # Determine color with smooth gradients
        if is_removed:
            color = self.REMOVED
        elif disease > 60:
            # Severe infection - dark red with pulsing
            color = self.SEVERE
            # Pulse animation for severe infection
            pulse = int(50 * abs(pulse_factor - 1.0))
            color = (min(255, color[0] + pulse), max(0, color[1] - pulse), max(0, color[2] - pulse))
        elif disease > 30:
            # Infected - orange/red mix
            ratio = (disease - 30) / 30
            r = int(self.INFECTED[0] * ratio + self.SEVERE[0] * (1 - ratio))
            g = int(self.INFECTED[1] * ratio + self.SEVERE[1] * (1 - ratio))
            b = int(self.INFECTED[2] * ratio + self.SEVERE[2] * (1 - ratio))
            color = (r, g, b)
        else:
            # Healthy - green
            color = self.HEALTHY
        
        # Draw main cell with border
        pygame.draw.rect(self.screen, color, (cell_x, cell_y, cell_w, cell_h))
        
        # Smooth border animation
        border_width = 2
        pygame.draw.rect(self.screen, self.BORDER, (cell_x, cell_y, cell_w, cell_h), border_width)
        
        # Draw health/disease percentage with better contrast
        if not is_removed:
            percentage = disease if disease > 0 else health
            text_color = (255, 255, 255) if percentage > 50 else (0, 0, 0)
            
            perc_text = self.font_small.render(f"{percentage:.0f}%", True, text_color)
            text_rect = perc_text.get_rect(center=(cell_x + cell_w // 2, cell_y + cell_h // 2))
            self.screen.blit(perc_text, text_rect)
    
    def draw_grid(self, state_dict, frame_count=0):
        """Draw the farm grid with animation effects"""
        grid_x = self.margin
        grid_y = self.margin
        grid_w = self.grid_size * self.cell_size
        grid_h = self.grid_size * self.cell_size
        
        # Draw grid background
        pygame.draw.rect(self.screen, self.GRID_BG, 
                        (grid_x - 5, grid_y - 5, grid_w + 10, grid_h + 10))
        pygame.draw.rect(self.screen, self.BORDER, 
                        (grid_x - 5, grid_y - 5, grid_w + 10, grid_h + 10), 3)
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            pos = grid_x + i * self.cell_size
            pygame.draw.line(self.screen, (60, 60, 100), 
                            (pos, grid_y), (pos, grid_y + grid_h), 1)
            pygame.draw.line(self.screen, (60, 60, 100), 
                            (grid_x, pos), (grid_x + grid_w, pos), 1)
        
        # Draw plants with pulse animation
        crop_health = state_dict['crop_health']
        disease_severity = state_dict['disease_severity']
        removed_plants = state_dict['removed_plants']
        
        # Pulse animation - cycle every 1 second
        pulse_phase = (frame_count % 20) / 20.0
        pulse_factor = 1.0 + 0.3 * abs(pulse_phase - 0.5) * 2
        
        for idx in range(self.grid_size * self.grid_size):
            row, col = divmod(idx, self.grid_size)
            x = grid_x + col * self.cell_size
            y = grid_y + row * self.cell_size
            
            is_removed = idx in removed_plants
            health = crop_health[idx]
            disease = disease_severity[idx]
            
            # Use pulse for infected plants only
            pulse = pulse_factor if (disease > 30 and not is_removed) else 1.0
            self.draw_plant_cell(x, y, health, disease, is_removed, pulse)
    
    def draw_panel(self, state_dict, action=None, reward=None, step=None):
        """Draw info panel with professional styling"""
        panel_x = self.game_width + 20
        panel_y = 20
        panel_h = self.screen_height - 40
        
        # Panel background with border
        pygame.draw.rect(self.screen, (25, 25, 55),
                        (panel_x - 10, panel_y - 10, self.panel_width, panel_h),
                        border_radius=8)
        pygame.draw.rect(self.screen, self.ACCENT,
                        (panel_x - 10, panel_y - 10, self.panel_width, panel_h), 2)
        
        # Title
        title = self.font_large.render("📊 STATUS", True, self.ACCENT)
        self.screen.blit(title, (panel_x, panel_y))
        
        # Statistics
        crop_health = state_dict['crop_health']
        disease_severity = state_dict['disease_severity']
        removed = state_dict['removed_plants']
        
        healthy = np.sum(crop_health > 70)
        infected = np.sum(disease_severity > 30)
        
        y = panel_y + 50
        line_h = 28
        
        # Color-coded stats
        stats = [
            ("🌱 HEALTHY", f"{healthy}/25", (100, 255, 100)),
            ("🔴 INFECTED", f"{infected}/25", (255, 150, 100)),
            ("❌ REMOVED", f"{len(removed)}", (200, 100, 100)),
            ("💧 MOISTURE", f"{state_dict['soil_moisture'][0]:.0f}%", (100, 200, 255)),
            ("🌡️ TEMP", f"{state_dict['weather'][0]:.0f}°C", (255, 150, 100)),
        ]
        
        for label, value, color in stats:
            label_text = self.font_small.render(label, True, color)
            value_text = self.font_medium.render(value, True, self.TEXT)
            
            self.screen.blit(label_text, (panel_x, y))
            self.screen.blit(value_text, (panel_x + 140, y))
            y += line_h
        
        # Last action
        if action is not None:
            action_names = [
                "🔍 MONITOR",
                "💉 FUNGICIDE",
                "🌿 NEEM OIL",
                "✂️  REMOVE",
                "💨 VENTILATION",
                "💧 IRRIGATION"
            ]
            action_colors = [
                (150, 150, 200),
                (255, 100, 100),
                (100, 200, 100),
                (200, 100, 100),
                (100, 200, 255),
                (100, 180, 255)
            ]
            
            y += 10
            label = self.font_small.render("LAST ACTION", True, (200, 200, 200))
            self.screen.blit(label, (panel_x, y))
            y += 25
            
            action_text = self.font_medium.render(action_names[action], True, action_colors[action])
            self.screen.blit(action_text, (panel_x, y))
        
        # Reward display (prominent)
        if reward is not None:
            y = panel_y + 280
            reward_color = (100, 255, 100) if reward > 0 else (255, 100, 100)
            
            label = self.font_small.render("⭐ REWARD", True, self.ACCENT)
            self.screen.blit(label, (panel_x, y))
            
            reward_display = self.font_huge.render(f"{reward:+.1f}", True, reward_color)
            self.screen.blit(reward_display, (panel_x + 20, y + 30))
        
        # Step counter at bottom
        if step is not None:
            y = panel_y + panel_h - 60
            step_label = self.font_small.render("STEP", True, (200, 200, 200))
            step_display = self.font_large.render(str(step), True, self.TEXT)
            
            self.screen.blit(step_label, (panel_x, y))
            self.screen.blit(step_display, (panel_x + 80, y - 5))
    
    def render(self, state_dict, action=None, reward=None, step=None):
        """Render complete frame with animations"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_open = False
                return
        
        # Clear screen
        self.screen.fill(self.BG_COLOR)
        
        # Draw game with frame count for pulse effects
        self.draw_grid(state_dict, self.frame_count)
        self.draw_panel(state_dict, action, reward, step)
        
        # Draw and update popups
        dt = self.clock.tick(self.fps) / 1000.0
        for popup in self.popups[:]:
            popup.update(dt)
            popup.draw(self.screen, self.font_medium)
            if not popup.alive:
                self.popups.remove(popup)
        
        # Draw and update action effects
        for effect in self.action_effects[:]:
            effect.update(dt)
            effect.draw(self.screen, self.cell_size)
            if not effect.alive:
                self.action_effects.remove(effect)
        
        # Draw FPS counter
        fps_text = pygame.font.Font(None, 12)
        fps_display = fps_text.render(f"FPS: {self.clock.get_fps():.0f}", True, (100, 100, 100))
        self.screen.blit(fps_display, (10, self.screen_height - 20))
        
        pygame.display.flip()
        self.frame_count += 1
    
    def close(self):
        self.is_open = False
        pygame.quit()


def run_professional_demo(algorithm='ppo', episodes=3):
    """Run the professional demo"""
    print("\n" + "=" * 80)
    print("🎮 PROFESSIONAL SMART FARM RL AGENT DEMO")
    print("=" * 80)
    print(f"\n📚 Loading trained {algorithm.upper()} agent...")
    
    # Create environment and renderer
    env = SmartCropDiseaseEnv(grid_size=5, episode_length=100, render_mode="rgb_array")
    renderer = ProfessionalFarmDemo(grid_size=5, fps=20)
    
    # Load model
    try:
        if algorithm.lower() == 'ppo':
            model = PPO.load('models/pg/ppo_exp_08')
            print("✅ PPO Agent Loaded (Best: 328.00 reward)")
        else:
            model = DQN.load('models/dqn/dqn_exp_06')
            print("✅ DQN Agent Loaded (Best: 315.46 reward)")
    except FileNotFoundError:
        print(f"❌ Model not found. Make sure models are in models/ folder")
        return
    
    print(f"\n🌾 Running {episodes} episode(s)...\n")
    print("=" * 80)
    
    episode_rewards = []
    episode_healthy = []
    
    for episode_num in range(episodes):
        print(f"\n🎮 EPISODE {episode_num + 1}/{episodes}")
        print("-" * 80)
        
        observation, _ = env.reset()
        episode_reward = 0
        max_healthy = 0
        
        for step in range(100):
            # Get action
            action, _ = model.predict(observation, deterministic=True)
            action = int(action)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            state_dict = env.get_state_dict()
            healthy = np.sum(state_dict['crop_health'] > 70)
            max_healthy = max(max_healthy, healthy)
            
            # Render with popup and action effect
            if renderer.is_open:
                renderer.render(state_dict, action, reward, step + 1)
                
                # Add popup at grid center for big rewards
                if reward > 5:
                    grid_center_x = renderer.margin + (renderer.grid_size * renderer.cell_size) // 2
                    grid_center_y = renderer.margin + (renderer.grid_size * renderer.cell_size) // 2
                    renderer.add_popup(grid_center_x, grid_center_y, reward)
                
                # Add action effect at random grid cell
                if action is not None:
                    grid_x = renderer.margin + np.random.randint(0, renderer.grid_size) * renderer.cell_size + renderer.cell_size // 2
                    grid_y = renderer.margin + np.random.randint(0, renderer.grid_size) * renderer.cell_size + renderer.cell_size // 2
                    renderer.add_action_effect(grid_x, grid_y, action)
            
            # Print progress
            if (step + 1) % 20 == 0:
                action_names = ["Monitor", "Fungicide", "Neem Oil", "Remove", "Ventilation", "Irrigation"]
                print(f"  Step {step + 1:3d} | {action_names[action]:12s} | "
                      f"Reward: {reward:+6.2f} | 🌱 {healthy:2d}/25 | Total: {episode_reward:7.2f}")
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_healthy.append(max_healthy)
        
        # Episode summary
        print(f"\n✅ Episode {episode_num + 1} Complete!")
        print(f"   💰 Total Reward: {episode_reward:8.2f}")
        print(f"   🌱 Max Healthy:  {max_healthy:8}/25")
        print(f"   💾 Total Cost:   ${state_dict['total_cost']:8.2f}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏆 DEMO COMPLETE!")
    print("=" * 80)
    print(f"📊 Episodes Played:        {episodes}")
    print(f"⭐ Average Reward:         {np.mean(episode_rewards):8.2f}")
    print(f"📈 Best Episode:           {np.max(episode_rewards):8.2f}")
    print(f"🌱 Average Max Healthy:    {np.mean(episode_healthy):8.1f}/25")
    print(f"📉 Standard Deviation:     {np.std(episode_rewards):8.2f}")
    
    if algorithm.lower() == 'ppo':
        print(f"\n✨ PPO Agent Performance:")
        print(f"   Trained on 100,000 timesteps")
        print(f"   Best configuration: LR=0.0005, batch=32, n_steps=128")
        print(f"   Final mean reward: 328.00 ± 15.24")
    else:
        print(f"\n✨ DQN Agent Performance:")
        print(f"   Trained on 100,000 timesteps")
        print(f"   Best configuration: LR=0.0005, batch=16, buffer=10,000")
        print(f"   Final mean reward: 315.46 ± 18.37")
    
    print("\n🎮 Thanks for watching the Smart Farm RL Agent Demo!")
    print("=" * 80 + "\n")
    
    renderer.close()
    env.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional Smart Farm Demo")
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'dqn'])
    parser.add_argument('--episodes', type=int, default=3)
    
    args = parser.parse_args()
    
    run_professional_demo(algorithm=args.algorithm, episodes=args.episodes)


if __name__ == "__main__":
    main()
