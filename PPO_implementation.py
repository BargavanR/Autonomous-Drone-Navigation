import numpy as np
import time
from stable_baselines3 import PPO
import sys
sys.path.append('/home/bargavan/VIT_submission/gym-pybullet-drones')
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

class DynamicTargetHoverAviary(HoverAviary):
    def __init__(self, gui=True, target_pos=None):
        super().__init__(gui=gui)
        self.target_pos = np.array(target_pos) if target_pos is not None else np.array([1.0, 1.0, 1.0])

    def reset(self):
        obs = super().reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Ensure correct format
        return obs

    def set_target(self, new_target):
        """ Update the target position dynamically. """
        self.target_pos = np.array(new_target)

    def step(self, action):
        """ Take a step in the environment and update the reward based on the target position. """
        result = super().step(action)  # Get environment output
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result  # New Gym API (5 values)
            done = terminated or truncated  # Combine termination conditions
        else:
            obs, reward, done, info = result  # Old Gym API (4 values)

        # Ensure obs is a NumPy array
        obs = np.array(obs)

        # Extract drone position correctly
        if obs.shape[0] == 1 and obs.shape[1] >= 3:
            drone_pos = obs[0, :3]  # If obs is (1, 72), extract the first 3 values
        elif obs.shape[0] >= 3:
            drone_pos = obs[:3]  # If obs is flat (72,), take first 3 elements
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")

        # Compute distance to target
        distance = np.linalg.norm(self.target_pos - drone_pos)

        # Reward function: Higher reward when closer to target
        reward = 1.0 / (1.0 + distance)  # Avoid division by zero

        # Mark as done if close to target
        done = done or (distance < 0.1)  # Stop episode if drone is close

        return obs, reward, done, info

# Load trained PPO model
model = PPO.load("/home/bargavan/VIT_submission/Model/drone_model.zip")

# Create environment
env = DynamicTargetHoverAviary(gui=True)

# List of target positions to reach
target_positions = [
    [1.0, 1.0, 1.0],
    [-1.0, -1.0, 1.5],
    [0.5, -0.5, 0.8],
    [2.0, 2.0, 2.0]
]

for target in target_positions:
    env.set_target(target)  # Update target position
    obs = env.reset()

    print(f"Moving to target: {target}")

    for i in range(500):  # Give it enough steps to reach target
        action, _states = model.predict(obs, deterministic=True)  # Get action from PPO
        obs, reward, done, info = env.step(action)  # Apply action

        print(f"Step {i}: Position = {obs[:3]}, Reward = {reward}")

        env.render()
        time.sleep(0.05)

        if done:
            print(f"Reached target {target}!")
            break  # Move to next target if reached

env.close()
