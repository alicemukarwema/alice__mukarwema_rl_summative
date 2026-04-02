"""
Smart Crop Disease Management Environment
A Gymnasium-compliant environment where an agent manages disease spread in a farm
to minimize disease impact and maximize crop yield.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any


class SmartCropDiseaseEnv(gym.Env):
    """
    Custom environment for intelligent farm disease management.
    
    State Space: Multi-dimensional observations including:
    - Crop health status per plant (0-100)
    - Disease severity (0-100)
    - Soil moisture (0-100)
    - Weather conditions (temperature, humidity, rainfall)
    - Days since infection (0-30)
    - Treatment history (count)
    
    Action Space:
    0: Do Nothing (monitor)
    1: Spray Fungicide (immediate cost, effectiveness 60-80%)
    2: Apply Neem Oil (moderate cost, effectiveness 40-60%, slower)
    3: Remove Infected Plant (high cost, prevents spread)
    4: Improve Ventilation (low cost, moderate effectiveness)
    5: Adjust Irrigation (low cost, disease prevention)
    """
    
    # Human-readable action names for explainability
    ACTION_NAMES = {
        0: "Do Nothing (Monitor Farm)",
        1: "Apply Chemical Fungicide (Fast & effective, but expensive)",
        2: "Apply Neem Oil (Organic, cheaper, but slower)",
        3: "Remove Infected Plants (Stop spread entirely, high crop loss)",
        4: "Improve Farm Ventilation (Lower humidity to prevent fungus)",
        5: "Optimize Irrigation (Prevent overwatering)"
    }

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, grid_size: int = 5, episode_length: int = 100, render_mode: str | None = None):
        """
        Initialize the environment.
        
        Args:
            grid_size: Size of the farm grid (grid_size x grid_size plants)
            episode_length: Maximum steps per episode
            render_mode: Visualization mode (None, "human", "rgb_array")
        """
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.render_mode = render_mode
        
        # Total plants
        self.num_plants = grid_size * grid_size
        
        # Time tracking
        self.current_step = 0
        
        # Cumulative metrics for reward calculation
        self.total_healthy_plants = 0
        self.total_cost_spent = 0
        self.total_loss_to_disease = 0
        
        # Define action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Define observation space (structured for clarity)
        self.observation_space = spaces.Dict({
            'crop_health': spaces.Box(low=0, high=100, shape=(self.num_plants,), dtype=np.float32),
            'disease_severity': spaces.Box(low=0, high=100, shape=(self.num_plants,), dtype=np.float32),
            'soil_moisture': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'weather': spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32),  # temp, humidity, rainfall
            'days_since_infection': spaces.Box(low=0, high=30, shape=(self.num_plants,), dtype=np.float32),
            'treatment_history': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'total_infected': spaces.Box(low=0, high=self.num_plants, shape=(1,), dtype=np.float32),
        })
        
        # Environment parameters
        self.disease_spread_rate = 0.05  # Daily spread rate
        self.infection_threshold = 30  # Health below this = infected
        self.recovery_threshold = 70  # Health above this = recovered
        
        # Cost parameters (in reward units)
        self.fungicide_cost = 5.0
        self.neem_cost = 2.0
        self.removal_cost = 10.0
        self.ventilation_cost = 1.0
        self.irrigation_cost = 0.5
        
        # Initialize state
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize farm state"""
        self.crop_health = np.full(self.num_plants, 85.0, dtype=np.float32)
        self.disease_severity = np.zeros(self.num_plants, dtype=np.float32)
        self.days_since_infection = np.zeros(self.num_plants, dtype=np.float32)
        self.soil_moisture = np.array([60.0], dtype=np.float32)
        
        # Weather: [temperature, humidity, rainfall]
        self.weather = np.array([25.0, 60.0, 30.0], dtype=np.float32)
        
        self.treatment_count = np.array([0.0], dtype=np.float32)
        self.removed_plants = set()
        
        self.current_step = 0
        self.total_healthy_plants = 0
        self.total_cost_spent = 0
        self.total_loss_to_disease = 0
        
        # Introduce initial infection(s)
        initial_infected_idx = np.random.choice(self.num_plants, size=1, replace=False)
        self.disease_severity[initial_infected_idx] = 40.0
        self.crop_health[initial_infected_idx] = 40.0
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Return current state observation"""
        total_infected = np.sum(self.disease_severity > 0).astype(np.float32)
        
        return {
            'crop_health': self.crop_health.copy(),
            'disease_severity': self.disease_severity.copy(),
            'soil_moisture': self.soil_moisture.copy(),
            'weather': self.weather.copy(),
            'days_since_infection': self.days_since_infection.copy(),
            'treatment_history': self.treatment_count.copy(),
            'total_infected': np.array([total_infected], dtype=np.float32),
        }
    
    def _apply_action(self, action: int) -> float:
        """
        Apply action and return immediate cost.
        
        Actions:
        0: Do Nothing (Monitor Farm)
        1: Apply Chemical Fungicide (cost=5.0, reduces disease by 70%)
        2: Apply Neem Oil (cost=2.0, reduces disease by 50%)
        3: Remove Infected Plants (cost=10.0, completely removes infection source)
        4: Improve Farm Ventilation (cost=1.0, reduces humidity to stop spread)
        5: Optimize Irrigation (cost=0.5, restores optimal soil moisture)
        """
        cost = 0.0
        
        if action == 0:
            # Do nothing
            pass
        
        elif action == 1:
            # Spray fungicide on infected plants
            cost = self.fungicide_cost * np.sum(self.disease_severity > self.infection_threshold)
            infected_mask = self.disease_severity > self.infection_threshold
            self.disease_severity[infected_mask] *= 0.3  # Reduce by 70%
            self.crop_health[infected_mask] += 5
            self.treatment_count += 1
        
        elif action == 2:
            # Apply neem oil (slower, cheaper)
            cost = self.neem_cost * np.sum(self.disease_severity > self.infection_threshold)
            infected_mask = self.disease_severity > self.infection_threshold
            self.disease_severity[infected_mask] *= 0.5  # Reduce by 50%
            self.crop_health[infected_mask] += 2
            self.treatment_count += 1
        
        elif action == 3:
            # Remove infected plants
            infected_mask = self.disease_severity > self.infection_threshold * 2
            cost = self.removal_cost * np.sum(infected_mask)
            self.crop_health[infected_mask] = 0
            self.disease_severity[infected_mask] = 0
            self.removed_plants.update(np.where(infected_mask)[0])
            self.treatment_count += 1
        
        elif action == 4:
            # Improve ventilation
            cost = self.ventilation_cost
            self.weather[1] -= 10  # Reduce humidity
            self.disease_severity *= 0.95  # Slight disease reduction
        
        elif action == 5:
            # Adjust irrigation to optimal level
            cost = self.irrigation_cost
            self.soil_moisture = np.array([70.0], dtype=np.float32)  # Set to optimal
            self.disease_severity *= 0.98  # Slight disease reduction
        
        self.total_cost_spent += cost
        return cost
    
    def _simulate_environment(self):
        """Simulate natural disease progression and environmental conditions"""
        # Disease spreads to neighboring plants
        for i in range(self.num_plants):
            if i in self.removed_plants:
                continue
            
            if self.disease_severity[i] > self.infection_threshold:
                # Spread to neighbors (4-connectivity in grid)
                neighbors = self._get_neighbors(i)
                for neighbor in neighbors:
                    if neighbor not in self.removed_plants:
                        # Spread probability depends on humidity and rainfall
                        spread_prob = (
                            self.disease_spread_rate * 
                            (self.weather[1] / 100.0) *  # Humidity factor
                            (self.weather[2] / 100.0)    # Rainfall factor
                        )
                        if np.random.random() < spread_prob:
                            self.disease_severity[neighbor] += 5
                
                self.days_since_infection[i] += 1
        
        # Natural health decline due to disease
        self.crop_health -= 0.5  # Base daily health decline
        self.crop_health[self.disease_severity > 0] -= self.disease_severity[self.disease_severity > 0] * 0.01
        
        # Clamp values
        self.crop_health = np.clip(self.crop_health, 0, 100)
        self.disease_severity = np.clip(self.disease_severity, 0, 100)
        
        # Update weather (stochastic)
        self.weather[0] += np.random.normal(0, 2)  # Temperature fluctuation
        self.weather[1] = 50 + 30 * np.sin(self.current_step / 10) + np.random.normal(0, 5)  # Humidity
        self.weather[2] = max(0, 30 + np.random.normal(0, 20))  # Rainfall
        self.weather = np.clip(self.weather, 0, 100)
        
        # Soil moisture decreases naturally
        self.soil_moisture -= 1
        self.soil_moisture = np.clip(self.soil_moisture, 0, 100)
    
    def _get_neighbors(self, plant_idx: int) -> list:
        """Get neighboring plant indices (4-connectivity)"""
        row, col = divmod(plant_idx, self.grid_size)
        neighbors = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                neighbors.append(nr * self.grid_size + nc)
        
        return neighbors
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on:
        - Healthy crops maintained: +1 per plant
        - Infection prevented: +5 per plant kept healthy
        - Disease severity reduction: +2 per unit reduction
        - Cost of actions: -cost
        """
        # Count healthy and infected plants
        healthy = np.sum(self.crop_health > self.recovery_threshold)
        infected = np.sum(self.disease_severity > self.infection_threshold)
        
        self.total_healthy_plants = healthy
        self.total_loss_to_disease = np.sum(self.crop_health[self.crop_health > 0] < self.infection_threshold)
        
        # Reward components
        health_reward = healthy * 0.5  # Reward for healthy plants
        infection_penalty = infected * -2.0  # Penalty for infected plants
        cost_penalty = self.total_cost_spent * 0.1  # Cost penalty
        
        total_reward = health_reward + infection_penalty - cost_penalty
        
        return total_reward
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Returns:
            observation: Current state
            reward: Reward signal
            terminated: Whether episode is done
            truncated: Whether max steps reached
            info: Additional information
        """
        self.current_step += 1
        
        # Apply action
        action_cost = self._apply_action(action)
        
        # Simulate environment progression
        self._simulate_environment()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        # Episode ends if all plants are removed or dead
        alive_plants = len(self.removed_plants) < self.num_plants
        if not alive_plants:
            terminated = True
        
        observation = self._get_observation()
        
        info = {
            'step': self.current_step,
            'action': action,
            'action_cost': action_cost,
            'healthy_plants': self.total_healthy_plants,
            'infected_plants': np.sum(self.disease_severity > self.infection_threshold),
            'total_cost': self.total_cost_spent,
            'average_health': np.mean(self.crop_health[list(range(self.num_plants)) if len(self.removed_plants) == 0 else list(set(range(self.num_plants)) - self.removed_plants)]),
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), {}
    
    def render(self):
        """Render environment state (handled by rendering.py)"""
        pass
    
    def close(self):
        """Clean up resources"""
        pass
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete environment state for visualization"""
        return {
            'crop_health': self.crop_health.copy(),
            'disease_severity': self.disease_severity.copy(),
            'soil_moisture': self.soil_moisture.copy(),
            'weather': self.weather.copy(),
            'days_since_infection': self.days_since_infection.copy(),
            'treatment_count': self.treatment_count.copy(),
            'removed_plants': list(self.removed_plants),
            'grid_size': self.grid_size,
            'current_step': self.current_step,
            'total_cost': self.total_cost_spent,
        }
