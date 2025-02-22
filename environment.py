import pybullet as p
import numpy as np
import time
import sys
import gymnasium as gyms
sys.path.append('/home/bargavan/VIT_submission/gym-pybullet-drones')
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel

import os
import matplotlib.pyplot as plt

class DroneNavigationAviary(BaseAviary):
    def __init__(
        self,
        drone_model=DroneModel.CF2X,
        num_drones=1,
        gui=True,
        record=False
    ):
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=10,
            pyb_freq=240,
            ctrl_freq=48,
            physics=True,
            gui=gui,
            record=record
        )
        # Environment parameters
        self.initial_height = 1.0
        self.obstacles = []
        self.camera_resolution = (640, 480)
        
        # Define waypoints
        self.waypoints = [
            [0, 0, 1],        # Starting point
            [1, 1, 1.5],      # Waypoint 1
            [-1, 2, 0.8],     # Waypoint 2
            [2, -1, 1.2],     # Waypoint 3
            [0, -2, 1.7],     # Waypoint 4
            [-2, -1, 0.5],    # Waypoint 5
            [1.5, 2.5, 1.3],  # Waypoint 6
            [3, 3, 1]         # Final goal
        ]
        self.current_waypoint = 0
        self.waypoint_threshold = 0.5  
        self.collision_threshold = 0.05
        
        # Create visualization directory
        self.viz_dir = "drone_navigation_viz"
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Create waypoint visualization
        self._createWaypoints()
        self._addObstacles()

    def _actionSpace(self):
        """Returns the action space of the environment."""
        # RPM action space for each rotor, normalized between 0 and 1
        return gyms.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,), 
            dtype=np.float32)
        

    def _observationSpace(self):
        """Returns the observation space of the environment."""
        # [x, y, z, qw, qx, qy, qz, vx, vy, vz]
        return gyms.spaces.Box(
            low=np.array([-np.inf]*10),
            high=np.array([np.inf]*10),
            dtype=np.float32
        )

    def _computeObs(self):
        """Returns the current observation of the environment."""
        state = self._getDroneStateVector(0)
        return np.array([
            *state[0:3],   # position
            *state[3:7],   # quaternion
            *state[10:13]  # linear velocity
        ], dtype=np.float32)

    def _computeInfo(self):
        """Compute the info dictionary with useful debugging information."""
        state = self._getDroneStateVector(0)
        current_pos = state[0:3]
        current_target = np.array(self.waypoints[self.current_waypoint])
        distance_to_target = np.linalg.norm(current_pos - current_target)
        
        return {
            'position': state[0:3],
            'orientation': state[3:7],
            'linear_velocity': state[10:13],
            'angular_velocity': state[13:16],
            'current_waypoint': self.current_waypoint,
            'distance_to_target': distance_to_target,
            'next_waypoint_position': self.waypoints[self.current_waypoint],
            'collision_warnings': self._checkCollisionWarnings()
        }

    def _checkCollisionWarnings(self):
        """Check for near collisions and return warning information."""
        warnings = []
        drone_pos = self._getDroneStateVector(0)[0:3]
        
        for obstacle_id in self.obstacles:
            contact_points = p.getContactPoints(
                bodyA=self.getDroneIds()[0],
                bodyB=obstacle_id
            )
            
            if contact_points:
                for contact in contact_points:
                    distance = contact[8]
                    if distance < self.collision_threshold * 2:  # Warning threshold
                        obstacle_pos = p.getBasePositionAndOrientation(obstacle_id)[0]
                        warnings.append({
                            'obstacle_id': obstacle_id,
                            'distance': distance,
                            'obstacle_position': obstacle_pos,
                            'relative_position': np.array(obstacle_pos) - np.array(drone_pos)
                        })
        
        return warnings

    def _createWaypoints(self):
        """Create visual markers for waypoints."""
        self.waypoint_ids = []
        colors = [
            [0, 1, 0, 0.7],      # Green (start)
            [0, 0, 1, 0.7],      # Blue (intermediate)
            [0, 0, 1, 0.7],
            [0, 0, 1, 0.7],
            [0, 0, 1, 0.7],
            [0, 0, 1, 0.7],
            [0, 0, 1, 0.7],
            [1, 0, 0, 0.7],      # Red (final)
        ]
        
        for i, (pos, color) in enumerate(zip(self.waypoints, colors)):
            visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.1,
                rgbaColor=color
            )
            
            waypoint_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual,
                basePosition=pos
            )
            self.waypoint_ids.append(waypoint_id)

            # Add line to next waypoint if not the last waypoint
            if i < len(self.waypoints) - 1:
                next_pos = self.waypoints[i + 1]
                points = [pos, next_pos]
                line_color = [0.5, 0.5, 0.5]  # Gray color for lines
                p.addUserDebugLine(
                    points[0],
                    points[1],
                    lineColorRGB=line_color,
                    lineWidth=1,
                    lifeTime=0
                )

    def _addObstacles(self):
        """Add static obstacles to create a challenging environment."""
        # Clear existing obstacles
        self.obstacles = []
        
        # Create boundary walls
        wall_positions = [
            {'start': [-3, -3], 'end': [3, -3], 'height': 1.0},  # South wall
            {'start': [3, -3], 'end': [3, 3], 'height': 1.0},    # East wall
            {'start': [3, 3], 'end': [-3, 3], 'height': 1.0},    # North wall
            {'start': [-3, 3], 'end': [-3, -3], 'height': 1.0},  # West wall
        ]
        
        # Create walls
        for wall in wall_positions:
            start, end = wall['start'], wall['end']
            length = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            angle = np.arctan2(end[1]-start[1], end[0]-start[0])
            
            wall_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[length/2, 0.1, wall['height']/2]
            )
            wall_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[length/2, 0.1, wall['height']/2],
                rgbaColor=[0.8, 0.8, 0.8, 0.9]
            )
            
            center_pos = [
                (start[0] + end[0])/2,
                (start[1] + end[1])/2,
                wall['height']/2
            ]
            
            wall_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=wall_shape,
                baseVisualShapeIndex=wall_visual,
                basePosition=center_pos,
                baseOrientation=p.getQuaternionFromEuler([0, 0, angle])
            )
            self.obstacles.append(wall_id)
        
        # Add static obstacles (boxes and cylinders)
        obstacle_configs = [
            {'type': 'box', 'position': [1, 1, 0.5], 'size': [0.3, 0.3, 1.0], 'color': [1, 0, 0, 0.7]},
            {'type': 'box', 'position': [-1, -1, 0.3], 'size': [0.4, 0.4, 0.6], 'color': [0, 1, 0, 0.7]},
            {'type': 'cylinder', 'position': [0, 1.5, 0.4], 'radius': 0.2, 'height': 0.8, 'color': [0, 1,0,0.7]},
            {'type': 'cylinder', 'position': [-1.5, 0, 0.6], 'radius': 0.15, 'height': 1.2, 'color': [1, 1, 0, 0.7]}
        ]
        
        for config in obstacle_configs:
            if config['type'] == 'box':
                shape = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[s/2 for s in config['size']]
                )
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[s/2 for s in config['size']],
                    rgbaColor=config['color']
                )
            else:  # cylinder
                shape = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=config['radius'],
                    height=config['height']
                )
                visual = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=config['radius'],
                    length=config['height'],
                    rgbaColor=config['color']
                )
                
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape,
                baseVisualShapeIndex=visual,
                basePosition=config['position']
            )
            self.obstacles.append(obstacle_id)

    def reset(self,seed = None ,options = None):
        """Reset the environment."""
        self.current_waypoint = 0
        self.obstacles = []
        obs = super().reset()
        
        # Set initial drone position with offset
        init_pos = np.array([-0.5, -0.5, self.initial_height])
        init_quat = np.array([0., 0., 0., 1.])
        
        p.resetBasePositionAndOrientation(
            self.getDroneIds()[0],
            init_pos,
            init_quat
        )
        
        self._addObstacles()
        self._createWaypoints()  # Recreate waypoints
        return self._computeObs(),self._computeInfo

    def _computeReward(self):
        """Compute the current reward value."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        
        # Basic survival reward
        reward = 0.1
        
        # Distance to current waypoint
        target_pos = np.array(self.waypoints[self.current_waypoint])
        distance = np.linalg.norm(pos - target_pos)
        reward += 2.0 / (1.0 + distance**2)
        
        # Check if reached current waypoint
        if distance < self.waypoint_threshold:
            reward += 5.0
            self.current_waypoint = min(self.current_waypoint + 1, len(self.waypoints) - 1)
        
        # Collision penalties
        for obstacle_id in self.obstacles:
            contact_points = p.getContactPoints(
                bodyA=self.getDroneIds()[0],
                bodyB=obstacle_id
            )
            if len(contact_points) > 0:
                for contact in contact_points:
                    contact_distance = contact[8]
                    if contact_distance < self.collision_threshold:
                        reward -= 2.0 * (1.0 - contact_distance/self.collision_threshold)
        
        return reward

    def _computeTerminated(self):
        """Compute the terminated flag."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        
        # More lenient height limits
        if pos[2] < 0.01 or pos[2] > 5.0:  # Increased height range
            print(f"Height violation: {pos[2]:.2f}")
            return True
            
        # More lenient collision check
        for obstacle_id in self.obstacles:
            contact_points = p.getContactPoints(
                bodyA=self.getDroneIds()[0],
                bodyB=obstacle_id
            )
            if len(contact_points) > 0:
                for contact in contact_points:
                    if contact[8] < 0.01:  # Reduced collision sensitivity
                        print(f"Collision detected: {contact[8]:.3f}")
                        return True
        
        # Check if reached final waypoint
        final_waypoint = np.array(self.waypoints[-1])
        distance_to_final = np.linalg.norm(pos - final_waypoint)
        
        if (distance_to_final < self.waypoint_threshold and 
            self.current_waypoint == len(self.waypoints) - 1):
            return True
            
        return False

    def _computeTruncated(self):
        """Compute the truncated flag."""
        if self.step_counter >= self.PYB_FREQ * 60:
            return True
        return False

    def _preprocessAction(self, action):
        """Pre-processes the action for step()."""
        # Scale action from [0,1] to [0,MAX_RPM]
        return action * self.MAX_RPM

    def step(self, action):
        """Applies the action to the environment."""
        action = self._preprocessAction(action)
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update text display for current waypoint and distance
        if self.GUI:  # Using GUI instead of gui
            state = self._getDroneStateVector(0)
            current_pos = state[0:3]
            target_pos = self.waypoints[self.current_waypoint]
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            
            text = f"Current Waypoint: {self.current_waypoint}\n"
            text += f"Distance to target: {distance:.2f}m\n"
            text += f"Height: {current_pos[2]:.2f}m"
            
            # Update or create text
            if hasattr(self, 'text_id'):
                p.removeUserDebugItem(self.text_id)
            self.text_id = p.addUserDebugText(
                text,
                [0, 0, 2],
                textColorRGB=[1, 1, 1],
                textSize=1.5
            )
            
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            self._updateDebugLines()
            
            # Get drone state
            state = self._getDroneStateVector(0)
            drone_pos = state[0:3]
            
            # Update camera to follow drone
            if self.GUI:  # Using GUI instead of gui
                p.resetDebugVisualizerCamera(
                    cameraDistance=3.0,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=drone_pos
                )
        
        return np.array([])

    def _updateDebugLines(self):
        """Update debug visualization lines."""
        if self.GUI:  # Using GUI instead of gui
            # Draw path to next waypoint
            if hasattr(self, 'path_line_id'):
                p.removeUserDebugItem(self.path_line_id)
            
            drone_pos = self._getDroneStateVector(0)[0:3]
            target_pos = self.waypoints[self.current_waypoint]
            
            self.path_line_id = p.addUserDebugLine(
                drone_pos,
                target_pos,
                lineColorRGB=[0, 1, 0],
                lineWidth=2.0,
                lifeTime=0.1
            )

    def close(self):
        """Clean up the environment."""
        # Remove all debug items
        if self.GUI:  # Using GUI instead of gui
            if hasattr(self, 'text_id'):
                p.removeUserDebugItem(self.text_id)
            if hasattr(self, 'path_line_id'):
                p.removeUserDebugItem(self.path_line_id)
        
        # Remove all waypoint visualizations
        for waypoint_id in self.waypoint_ids:
            p.removeBody(waypoint_id)
        
        # Remove all obstacles
        for obstacle_id in self.obstacles:
            p.removeBody(obstacle_id)
        
        super().close()