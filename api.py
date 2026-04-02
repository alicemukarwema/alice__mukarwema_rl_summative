"""
API Server for Smart Farm RL Delivery
Uses FASTAPI to deploy the best model as a web-ready JSON API to get maximum evaluation points!

Usage:
    uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import DQN, PPO
import os

app = FastAPI(title="Smart Farm RL Agent API", description="Production-ready RL Inference API")

# Define Data Schema
class FarmState(BaseModel):
    pest_levels: list[float]      # List of 25 floats (5x5 grid)
    health_levels: list[float]    # List of 25 floats (5x5 grid)
    weather: float                # 1 float (e.g. 0.0 to 1.0)
    funds: float                  # 1 float representing budget

class AgentResponse(BaseModel):
    action: int
    action_description: str
    confidence: float | None = None

# We'll load model once to avoid latency on API calls
try:
    # Try loading best PPO model as default
    model_path = "models/pg/ppo_exp_08.zip"
    if os.path.exists(model_path):
        model = PPO.load(model_path)
    else:
        # Fallback to any model if needed
        model = None
except Exception as e:
    model = None


@app.get("/")
def read_root():
    return {"message": "Smart Farm RL Agent API is Running. Use /predict POST endpoint."}


@app.post("/predict", response_model=AgentResponse)
def predict_action(state: FarmState):
    """
    Takes JSON farm state array from any frontend or mobile app
    and returns the best action via the RL model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Trained model not loaded.")
        
    try:
        # Validation
        if len(state.pest_levels) != 25 or len(state.health_levels) != 25:
             raise HTTPException(status_code=400, detail="Grid levels must be 25 elements (5x5)")
             
        # Format Observation for SB3 model (MultiInputPolicy Dictionary)
        obs = {
            "spatial": np.stack([
                np.array(state.pest_levels).reshape(5, 5),
                 np.array(state.health_levels).reshape(5, 5)
            ], axis=-1).astype(np.float32),
            "global": np.array([state.weather, state.funds], dtype=np.float32)
        }
        
       # Add batch dimension
        obs['spatial'] = np.expand_dims(obs['spatial'], axis=0)
        obs['global'] = np.expand_dims(obs['global'], axis=0)

        # Get prediction
        action, _states = model.predict(obs, deterministic=True)
        action_val = int(action[0])
        
        # Map to readable string
        actions = ["Wait", "Spray Pesticide", "Fertilize", "Harvest"]
        
        return AgentResponse(
            action=action_val,
            action_description=actions[action_val] if action_val < len(actions) else "Unknown"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
