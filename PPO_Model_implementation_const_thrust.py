import gym
import pybullet
import numpy as np
import time
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# Load the environment
env = HoverAviary(gui=True)
obs = env.reset()

# Extract observation if returned as a tuple
if isinstance(obs, tuple):
    obs = obs[0]

# Set constant action values to hover
HOVER_THRUST = 0.6  # Adjust if needed to balance hover
dummy_action = np.array([[HOVER_THRUST, HOVER_THRUST, HOVER_THRUST, HOVER_THRUST]], dtype=np.float32)

# Run simulation
for i in range(500):
    # Take dummy hover action
    obs, reward, done, info, *_ = env.step(dummy_action)
    
    # Extract observation if returned as a tuple
    if isinstance(obs, tuple):
        obs = obs[0]

    print(f"Step {i}: Action = {dummy_action}, Reward = {reward}")

    env.render()
    time.sleep(0.05)

    # Reset environment if done
    if done:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Ensure correct format

env.close()
