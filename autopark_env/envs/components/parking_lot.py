import numpy as np
from .obstacle import Obstacle

class ParkingLot:
    """
    Represents a parking lot environment with vehicles, obstacles, and a goal position.
    """
    def __init__(self, width, height):
        """
        Initialize the parking lot with given dimensions.
        """
        self.width = width
        self.height = height
        self.vehicles = []  # List to store movable vehicles
        self.static_vehicles = []  # List to store stationary vehicles
        self.obstacles = []  # List to store obstacles
        self.goal_position = None  # Target parking position

    def add_vehicle(self, vehicle):
        """Add a movable vehicle to the parking lot."""
        self.vehicles.append(vehicle)

    def add_static_vehicle(self, static_vehicle):
        """Add a stationary vehicle to the parking lot."""
        self.static_vehicles.append(static_vehicle)

    def add_obstacle(self, obstacle):
        """Add an obstacle to the parking lot."""
        self.obstacles.append(obstacle)

    def set_goal_position(self, position):
        """Set the target parking position."""
        self.goal_position = position

    def clear(self):
        """Reset the parking lot by clearing all vehicles, obstacles, and goal position."""
        self.vehicles.clear()
        self.static_vehicles.clear()
        self.obstacles.clear()
        self.goal_position = None
