import gym
import pybullet as p
import numpy as np
import time
from casadi import *
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
#this model is failure because we dont know the mass of the drone,
#distance bwtn motors .and diagonal moment of inertia of the drone
#cant sble to find the urdf file of the drone for mass
class MPCController:
    def __init__(self):
        self.horizon = 5
        self.dt = 0.1
        self.mass = 0.1
        self.gravity = 9.81
        self.opti = None  # Initialize optimizer instance
        self.setup_optimizer()

    def setup_optimizer(self):
        # Initialize optimization problem
        self.opti = Opti()
        
        # State variables [x, y, z, vx, vy, vz]
        state = self.opti.variable(6, self.horizon+1)
        controls = self.opti.variable(4, self.horizon)
        
        # Parameters
        initial_state = self.opti.parameter(6, 1)
        target_state = self.opti.parameter(6, 1)
        
        # Weights
        Q = diag([10, 10, 10, 1, 1, 1])
        R = diag([0.1, 0.1, 0.1, 0.1])
        
        # Dynamics constraints
        for t in range(self.horizon):
            acceleration_z = (controls[0,t] + controls[1,t] + 
                            controls[2,t] + controls[3,t])/self.mass - self.gravity
            
            self.opti.subject_to(state[0:3, t+1] == state[0:3, t] + state[3:6, t] * self.dt)
            self.opti.subject_to(state[5, t+1] == state[5, t] + acceleration_z * self.dt)
            
            # Cost function using CasADi's mtimes
            state_error = state[:,t] - target_state
            cost = mtimes(mtimes(state_error.T, Q), state_error)
            cost += mtimes(mtimes(controls[:,t].T, R), controls[:,t])
            self.opti.minimize(cost)
        
        # Constraints
        self.opti.subject_to(self.opti.bounded(0.3, controls, 0.7))
        self.opti.subject_to(state[:,0] == initial_state)
        
        # Solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt', opts)
        
        # Save references
        self.state_var = state
        self.controls_var = controls
        self.initial_state_param = initial_state
        self.target_state_param = target_state

    def compute_control(self, current_state, target):
        try:
            # Set parameter values
            self.opti.set_value(self.initial_state_param, current_state)
            self.opti.set_value(self.target_state_param, target)
            
            # Solve optimization
            sol = self.opti.solve()
            return np.clip(sol.value(self.controls_var[:,0]), 0.3, 0.7)
        except RuntimeError as e:
            print(f"MPC failed: {e}")
            return np.array([0.5, 0.5, 0.5, 0.5])  # Fallback to hover

def run():
    env = HoverAviary(gui=True)
    env.reset()
    
    mpc = MPCController()
    target = np.array([2.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    
    try:
        for _ in range(1000):
            state = env._getDroneStateVector(0)
            current_state = np.array([
                state[0], state[1], state[2],  # Position
                state[7], state[8], state[9]    # Velocity
            ])
            
            action = mpc.compute_control(current_state, target)
            _, _, _, _, _ = env.step(action.reshape(1, 4))
            env.render()
            
            if np.linalg.norm(current_state[:3] - target[:3]) < 0.1:
                print("Target reached!")
                break
                
            time.sleep(0.02)
    finally:
        env.close()

if __name__ == "__main__":
    run()