import numpy as np
from typing import Dict, List

class Vehicle:
    LENGTH: float = 4.0
    MAX_STEERING_ANGLE: float = np.pi / 20  # About 18*2 degrees, adjustable as needed
    MAX_STEERING_CHANGE: float = np.pi / 40  # Maximum steering angle change per step, adjustable as needed
    MAX_SPEED: float = 10.0
    MIN_SPEED: float = -10.0  # Allow reverse
    MAX_ACCELERATION: float = 0.3  # Maximum acceleration
    MAX_STEERING = 0.5  # Consistent with the scaling in the environment

    def __init__(self, position: List[float], heading: float = 0, velocity: float = 0):
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading
        self.velocity = velocity
        
        self.action: Dict[str, float] = {
            "steering": 0,
            "acceleration": 0
        }
        
        self.crashed = False
        self.steering_angle = 0  # 初始化转向角

    def act(self, action: Dict[str, float]) -> None:
        """Store an action, waiting for execution"""
        if action:
            self.action.update(action)

    def step(self, dt: float) -> None:
        """Update vehicle state"""
        self.clip_actions()

        # Calculate the change in steering angle
        steering_change = self.action["steering"] * self.MAX_STEERING_CHANGE

        # Update steering angle and limit it within the maximum steering angle range
        self.steering_angle = np.clip(
            self.steering_angle + steering_change,
            -self.MAX_STEERING_ANGLE,
            self.MAX_STEERING_ANGLE
        )

        # Update speed
        self.velocity += self.action["acceleration"] * dt
        self.velocity = np.clip(self.velocity, self.MIN_SPEED, self.MAX_SPEED)

        # Calculate angular velocity (based on the bicycle model)
        # Ensure to use the tangent function to calculate angular velocity
        if np.cos(self.steering_angle) != 0:  # Prevent division by zero
            angular_velocity = (self.velocity / self.LENGTH) * np.tan(self.steering_angle)
        else:
            angular_velocity = 0  # If heading angle is ±π/2, set angular velocity to 0

        # Update heading
        self.heading += angular_velocity * dt

        # Update position (based on current speed and heading)
        self.position += self.velocity * dt * np.array([np.cos(self.heading), np.sin(self.heading)])

        # Ensure heading is between -π and π
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi


    def clip_actions(self) -> None:
        """Limit action range"""
        if self.crashed:
            self.action["steering"] = 0
            self.action["acceleration"] = 0
        # Only clip, no scaling
        self.action["steering"] = np.clip(self.action["steering"], -self.MAX_STEERING, self.MAX_STEERING)
        self.action["acceleration"] = np.clip(self.action["acceleration"], -self.MAX_ACCELERATION, self.MAX_ACCELERATION)

    def get_state(self) -> Dict[str, float]:
        """Return the current state of the vehicle"""
        return {
            'position': self.position.tolist(),
            'heading': self.heading,
            'velocity': self.velocity,
            'steering': self.action["steering"],
            'acceleration': self.action["acceleration"],
            'crashed': self.crashed,
            'steering_angle': self.steering_angle
        }

class StaticVehicle:
    LENGTH: float = 4.0
    WIDTH: float = 2.0

    def __init__(self, position: List[float], heading: float = 0):
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading

    def get_state(self) -> Dict[str, float]:
        """Return the current state of the static vehicle"""
        return {
            'position': self.position.tolist(),
            'heading': self.heading,
            'length': self.LENGTH,
            'width': self.WIDTH
        }
