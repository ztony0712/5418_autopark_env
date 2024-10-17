import numpy as np
from typing import Dict, List

class Vehicle:
    LENGTH: float = 4.0
    MAX_STEERING_ANGLE: float = np.pi / 4  # About 45 degrees, can be adjusted as needed
    MAX_SPEED: float = 20.0
    MIN_SPEED: float = -10.0  # Allow reverse
    MAX_ACCELERATION: float = 1.0  # New: maximum acceleration

    def __init__(self, position: List[float], heading: float = 0, velocity: float = 0):
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading
        self.velocity = velocity
        
        self.action: Dict[str, float] = {
            "steering": 0,
            "acceleration": 0
        }
        
        self.crashed = False
        self.steering_angle = 0

    def act(self, action: Dict[str, float]) -> None:
        """Store an action to be executed"""
        if action:
            self.action.update(action)

    def step(self, dt: float) -> None:
        """Update vehicle state"""
        self.clip_actions()
        
        # Update steering angle and limit it
        self.steering_angle = np.clip(
            self.action["steering"] * self.MAX_STEERING_ANGLE,
            -self.MAX_STEERING_ANGLE,
            self.MAX_STEERING_ANGLE
        )
        
        # Update velocity
        self.velocity += self.action["acceleration"] * dt
        self.velocity = np.clip(self.velocity, self.MIN_SPEED, self.MAX_SPEED)
        
        # Update heading (bicycle model)
        angular_velocity = self.velocity * np.tan(self.steering_angle) / self.LENGTH
        self.heading += angular_velocity * dt
        
        # Update position
        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        
        # Ensure heading angle is between -π and π
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

    def clip_actions(self) -> None:
        """Limit the range of actions"""
        if self.crashed:
            self.action["steering"] = 0
            self.action["acceleration"] = 0
        self.action["steering"] = np.clip(self.action["steering"], -1, 1)
        self.action["acceleration"] = np.clip(self.action["acceleration"], -1, 1) * self.MAX_ACCELERATION

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
