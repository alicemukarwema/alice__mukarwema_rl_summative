"""
Visualization module for the Smart Crop Disease Management Environment.
Uses Pygame for 2D rendering and real-time visualization.
"""

import pygame
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum


class PlantState(Enum):
    """Plant health states for visualization"""
    HEALTHY = (34, 177, 76)      # Green
    INFECTED = (244, 67, 54)     # Red
    SEVERELY_INFECTED = (136, 14, 79)  # Dark red
    REMOVED = (158, 158, 158)    # Gray


class FarmRenderer:
    """Pygame-based renderer for the farm environment"""
    
    def __init__(self, grid_size: int = 5, cell_size: int = 80, fps: int = 30):
        """
        Initialize the farm renderer.
        
        Args:
            grid_size: Size of farm grid
            cell_size: Size of each plant cell in pixels
            fps: Frames per second for rendering
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        
        # Calculate window dimensions
        self.info_panel_width = 300
        self.grid_width = grid_size * cell_size
        self.grid_height = grid_size * cell_size
        self.window_width = self.grid_width + self.info_panel_width + 40
        self.window_height = self.grid_height + 40
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Smart Farm - Disease Management System")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_tiny = pygame.font.Font(None, 14)
        self.font_huge = pygame.font.Font(None, 72)
        
        self.is_open = True
    
    def render(self, state_dict: Dict[str, Any], action: Optional[int] = None, 
               reward: Optional[float] = None, info: Optional[Dict] = None):
        """
        Render the farm environment state.
        
        Args:
            state_dict: Environment state dictionary
            action: Current action taken
            reward: Reward received
            info: Additional info dict
        """
        if not self.is_open:
            return
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_open = False
                return
        
        # Clear screen
        self.screen.fill((240, 240, 240))
        
        # Draw farm grid
        self._draw_farm_grid(state_dict)
        
        # Draw info panel
        self._draw_info_panel(state_dict, action, reward, info)

        # Draw big action banner if an action was just taken
        if action is not None:
            self._draw_action_banner(action)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _draw_action_banner(self, action: int):
        """Draw a large clear banner indicating the action taken"""
        action_names = {
            0: ("NO ACTION", (150, 150, 150)),
            1: ("APPLY ORGANIC PESTICIDE", (0, 200, 100)),
            2: ("APPLY CHEMICAL PESTICIDE", (255, 100, 0)),
            3: ("REMOVE INFECTED CROP", (200, 0, 0))
        }
        
        if action in action_names:
            text, color = action_names[action]
            
            # Semi-transparent overlay
            overlay = pygame.Surface((self.window_width, 100))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, self.window_height // 2 - 50))
            
            # Draw text
            text_surface = self.font_huge.render(text, True, color)
            text_rect = text_surface.get_rect(center=(self.window_width // 2, self.window_height // 2))
            
            # Draw outline
            outline_surface = self.font_huge.render(text, True, (255, 255, 255))
            for dx, dy in [(-2,-2), (2,-2), (-2,2), (2,2)]:
                outline_rect = text_surface.get_rect(center=(self.window_width // 2 + dx, self.window_height // 2 + dy))
                self.screen.blit(outline_surface, outline_rect)
                
            self.screen.blit(text_surface, text_rect)
            
            # Force update and small delay to make it visible
            pygame.display.flip()
            pygame.time.delay(100)
            
    def _draw_farm_grid(self, state_dict: Dict[str, Any]):
        """Draw the farm grid with plant health visualization"""
        crop_health = state_dict['crop_health']
        disease_severity = state_dict['disease_severity']
        removed_plants = state_dict['removed_plants']
        grid_size = state_dict['grid_size']
        
        margin = 20
        start_x = margin
        start_y = margin
        
        for idx in range(grid_size * grid_size):
            row, col = divmod(idx, grid_size)
            x = start_x + col * self.cell_size
            y = start_y + row * self.cell_size
            
            # Determine plant color based on health status
            if idx in removed_plants:
                color = PlantState.REMOVED.value
                health = 0
            elif disease_severity[idx] > 60:
                color = PlantState.SEVERELY_INFECTED.value
                health = disease_severity[idx]
            elif disease_severity[idx] > 30:
                color = PlantState.INFECTED.value
                health = disease_severity[idx]
            else:
                color = PlantState.HEALTHY.value
                health = crop_health[idx]
            
            # Draw plant cell
            pygame.draw.rect(self.screen, color, 
                           (x, y, self.cell_size - 2, self.cell_size - 2))
            pygame.draw.rect(self.screen, (0, 0, 0), 
                           (x, y, self.cell_size - 2, self.cell_size - 2), 2)
            
            # Draw health percentage
            if idx not in removed_plants:
                health_text = self.font_tiny.render(f"{health:.0f}%", True, (255, 255, 255))
                text_rect = health_text.get_rect(center=(x + self.cell_size // 2 - 1, 
                                                         y + self.cell_size // 2 - 1))
                self.screen.blit(health_text, text_rect)
    
    def _draw_info_panel(self, state_dict: Dict[str, Any], action: Optional[int] = None, 
                        reward: Optional[float] = None, info: Optional[Dict] = None):
        """Draw the information panel with statistics"""
        panel_x = self.grid_width + 40
        panel_y = 30
        
        # Background for the panel (white rounded rectangle with slight shadow)
        panel_rect = pygame.Rect(panel_x - 10, panel_y - 10, self.info_panel_width - 20, self.window_height - 40)
        pygame.draw.rect(self.screen, (255, 255, 255), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (200, 200, 200), panel_rect, width=2, border_radius=10)

        # Title
        title_font = pygame.font.Font(None, 32)
        title = title_font.render("FARM STATUS", True, (30, 30, 30))
        self.screen.blit(title, (panel_x, panel_y))
        
        # Calculate statistics
        crop_health = state_dict['crop_health']
        disease_severity = state_dict['disease_severity']
        removed_plants = state_dict['removed_plants']
        weather = state_dict['weather']
        
        healthy_count = np.sum(crop_health > 70)
        infected_count = np.sum(disease_severity > 30)
        
        line_y = panel_y + 50
        line_height = 28
        
        # Better formatting and colors for the text
        info_items = [
            ("Step / Time", f"{state_dict['current_step']}", (0, 0, 0)),
            ("Healthy Crops", f"{healthy_count} / {self.grid_size * self.grid_size}", (34, 177, 76)),
            ("Infected Crops", f"{infected_count}", (244, 67, 54) if infected_count > 0 else (0,0,0)),
            ("Removed Crops", f"{len(removed_plants)}", (100, 100, 100)),
            ("Avg Health", f"{np.mean(crop_health):.1f}%", (0, 0, 150)),
            ("Avg Disease", f"{np.mean(disease_severity):.1f}%", (200, 0, 0) if np.mean(disease_severity) > 10 else (0,0,0)),
            ("----------------", "", (150, 150, 150)),
            ("Soil Moisture", f"{state_dict['soil_moisture'][0]:.1f}%", (0, 100, 200)),
            ("Temperature", f"{weather[0]:.1f}°C", (200, 100, 0)),
            ("Humidity", f"{weather[1]:.1f}%", (0, 150, 200)),
            ("Rainfall", f"{weather[2]:.1f}mm", (0, 50, 250)),
            ("----------------", "", (150, 150, 150)),
            ("Total Cost", f"${state_dict['total_cost']:.1f}", (200, 0, 0)),
        ]
        
        # Display information
        for label, val, color in info_items:
            # Render Label
            lbl_surface = self.font_small.render(label, True, (100, 100, 100))
            self.screen.blit(lbl_surface, (panel_x, line_y))
            
            # Render Value right aligned
            val_surface = self.font_large.render(val, True, color)
            val_rect = val_surface.get_rect(right=panel_x + self.info_panel_width - 40, top=line_y-5)
            self.screen.blit(val_surface, val_rect)
            
            line_y += line_height
            
        if action is not None:
            action_names = ["Monitor", "Fungicide", "Neem Oil", "Remove", "Ventilation", "Irrigation"]
            # Fallback if action is out of range
            action_str = action_names[action] if action < len(action_names) else f"Action {action}"
            
            line_y += 10
            action_lbl = self.font_large.render("Current Action:", True, (50, 50, 50))
            self.screen.blit(action_lbl, (panel_x, line_y))
            
            line_y += 25
            action_val = title_font.render(action_str, True, (0, 150, 50))
            self.screen.blit(action_val, (panel_x, line_y))
            
        if reward is not None:
            line_y += 40
            reward_color = (0, 180, 0) if reward >= 0 else (220, 0, 0)
            reward_lbl = self.font_small.render("Latest Reward:", True, (100, 100, 100))
            self.screen.blit(reward_lbl, (panel_x, line_y))
            
            reward_val = title_font.render(f"{reward:+.2f}", True, reward_color)
            self.screen.blit(reward_val, (panel_x + 100, line_y - 5))

    def close(self):
        """Close the renderer"""
        self.is_open = False
        pygame.quit()


class EnvironmentVisualizer:
    """High-level interface for environment visualization"""
    
    def __init__(self, env, grid_size: int = 5, render_fps: int = 30):
        """
        Initialize the visualizer.
        
        Args:
            env: Gymnasium environment
            grid_size: Size of farm grid
            render_fps: FPS for rendering
        """
        self.env = env
        self.renderer = FarmRenderer(grid_size=grid_size, fps=render_fps)
    
    def render_step(self, action: int, reward: float, info: Dict[str, Any]):
        """Render a single environment step"""
        state_dict = self.env.get_state_dict()
        self.renderer.render(state_dict, action, reward, info)
    
    def render_state(self):
        """Render current environment state without action info"""
        state_dict = self.env.get_state_dict()
        self.renderer.render(state_dict)
    
    def close(self):
        """Close the visualizer"""
        self.renderer.close()
