import gym
import numpy as np
import time
import sys
sys.path.append('/home/bargavan/VIT_submission/gym-pybullet-drones')
from gym_pybullet_drones.envs.HoverAviary import HoverAviary


class PIDController:
    def __init__(self, kp, ki, kd, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.output_limit = output_limit  # Limit for integral windup
        self.last_time = time.time()

    def compute(self, target, current):
        error = target - current
        current_time = time.time()
        dt = current_time - self.last_time if self.last_time else 1.0
        self.last_time = current_time

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        if self.output_limit:
            self.integral = np.clip(self.integral, -self.output_limit, self.output_limit)
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


def run():
    env = HoverAviary(gui=True)
    state, _ = env.reset()
    
    pid_x = PIDController(kp=6, ki=0.02, kd=3, output_limit=5)
    pid_y = PIDController(kp=6, ki=0.02, kd=3, output_limit=5)
    pid_z = PIDController(kp=8, ki=0.02, kd=4, output_limit=5)
    
    x_set, y_set, z_set = int(input("Enter x: ")), int(input("Enter y: ")), int(input("Enter z: "))
    
    for i in range(1000):
        x, y, z = state[0][0], state[0][1], state[0][2]
        
        control_x = np.clip(pid_x.compute(x_set, x), -0.3, 0.3)
        control_y = np.clip(pid_y.compute(y_set, y), -0.3, 0.3)
        control_z = np.clip(pid_z.compute(z_set, z), -0.3, 0.3)
        
        base_thrust = 0.5  # Adjust based on drone weight
        action = np.array([
            base_thrust + control_z - control_x - control_y,
            base_thrust + control_z + control_x + control_y,
            base_thrust + control_z + control_x - control_y,
            base_thrust + control_z - control_x + control_y
        ])
        if np.any(np.isnan(action)):
            break
        
        action = np.clip(action, 0.3, 0.7)
        action = np.reshape(action, (1, 4))
        
        state, _, _, _, _ = env.step(action)
        env.render()
        time.sleep(0.05)
    
    env.close()

run()