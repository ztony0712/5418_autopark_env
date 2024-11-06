import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
import pygame
import random
import gymnasium as gym

from .components.parking_lot import ParkingLot
from .components.vehicle import StaticVehicle, Vehicle
from .components.obstacle import Obstacle
from .components.graphics import VehicleGraphics

# Global parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
NUM_COLS = 10
NUM_ROWS = 3
LANE_WIDTH = 40
LANE_HEIGHT = 100
VEHICLE_WIDTH = 40
VEHICLE_HEIGHT = 20
GOAL_SIZE = 25
STATIC_VEHICLE_PROBABILITY = 0.3
SAFE_DISTANCE = 20  # Safety threshold for distance to obstacles

# Additional global parameters
INITIAL_VEHICLE_X = (SCREEN_WIDTH - NUM_COLS * LANE_WIDTH) // 2 - 30
INITIAL_VEHICLE_Y = 50
STEERING_ANGLE = 0.1  # Steering angle
STEP_REWARD = -0.1  # Reward per step
FPS = 30
GOAL_THRESHOLD = 10  # Distance threshold to consider goal as reached

class MyNewEnv(gym.Env):
    def __init__(self):
        super(MyNewEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([-800, -600, -np.inf, -np.pi, 0, 0], dtype=np.float32),
            high=np.array([800, 600, np.inf, np.pi, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.state = None
        self.parking_lot = None
        self.vehicle = None
        self.goal_heading = 0  # Initialize goal heading

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Parking Lot Visualization")
        self.clock = pygame.time.Clock()

    def _create_parking_lot(self):
        self.parking_lot = ParkingLot(SCREEN_WIDTH, SCREEN_HEIGHT)

    def _create_vehicle(self):
        self.vehicle = Vehicle([INITIAL_VEHICLE_X, INITIAL_VEHICLE_Y], heading=0)
        self.parking_lot.add_vehicle(self.vehicle)

    def _create_static_vehicles(self):
        self.parking_lot.static_vehicles.clear()
        start_x = (SCREEN_WIDTH - NUM_COLS * LANE_WIDTH) // 2
        start_y = (SCREEN_HEIGHT - NUM_ROWS * LANE_HEIGHT) // 2

        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                vehicle_x = start_x + col * LANE_WIDTH + LANE_WIDTH // 2
                vehicle_y = start_y + row * LANE_HEIGHT + LANE_HEIGHT // 4

                if self.parking_lot.goal_position and (
                    abs(vehicle_x - self.parking_lot.goal_position[0]) < LANE_WIDTH // 2 and
                    abs(vehicle_y - self.parking_lot.goal_position[1]) < LANE_HEIGHT // 2):
                    continue

                if random.random() < STATIC_VEHICLE_PROBABILITY:
                    static_vehicle = StaticVehicle([vehicle_x, vehicle_y], heading=np.pi / 2)
                    self.parking_lot.add_static_vehicle(static_vehicle)

    def _create_goal(self):
        start_x = (SCREEN_WIDTH - NUM_COLS * LANE_WIDTH) // 2
        start_y = (SCREEN_HEIGHT - NUM_ROWS * LANE_HEIGHT) // 2

        random_col = random.randint(0, NUM_COLS - 2)
        random_row = random.randint(0, NUM_ROWS - 1)

        goal_x = start_x + random_col * LANE_WIDTH + (LANE_WIDTH - GOAL_SIZE) // 2 + 5
        goal_y = start_y + random_row * LANE_HEIGHT + LANE_HEIGHT // 4

        self.parking_lot.set_goal_position((goal_x, goal_y))

        # Set the goal heading based on the goal position relative to the vehicle's current position
        self.goal_heading = np.arctan2(goal_y - INITIAL_VEHICLE_Y, goal_x - INITIAL_VEHICLE_X)

    def _create_walls(self):
        wall_thickness = 10  # Wall thickness

        walls = [
            # Top wall
            Obstacle([SCREEN_WIDTH / 2, wall_thickness / 2], [SCREEN_WIDTH, wall_thickness]),
            # Bottom wall
            Obstacle([SCREEN_WIDTH / 2, SCREEN_HEIGHT - wall_thickness / 2], [SCREEN_WIDTH, wall_thickness]),
            # Left wall
            Obstacle([wall_thickness / 2, SCREEN_HEIGHT / 2], [wall_thickness, SCREEN_HEIGHT]),
            # Right wall
            Obstacle([SCREEN_WIDTH - wall_thickness / 2, SCREEN_HEIGHT / 2], [wall_thickness, SCREEN_HEIGHT])
        ]

        for wall in walls:
            self.parking_lot.add_obstacle(wall)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize parking lot, vehicle, goal, and walls
        self._create_parking_lot()
        self._create_vehicle()
        self._create_goal()
        self._create_static_vehicles()
        self._create_walls()

        # Calculate initial distance to goal
        self.initial_distance_to_goal = np.linalg.norm(
            np.array(self.vehicle.position) - np.array(self.parking_lot.goal_position)
        )
        self.prev_distance_to_goal = self.initial_distance_to_goal

        # Initialize state and return
        self.state = self._get_obs()
        return self.state, {}

    def step(self, action):
        # Parse action
        steering = action[0] * 0.1  # Range [-0.1, 0.1]
        acceleration = action[1] * 0.5  # Scale acceleration to a reasonable range

        # Update vehicle state
        self.vehicle.action["steering"] = steering
        self.vehicle.action["acceleration"] = acceleration
        self.vehicle.step(1.0 / FPS)  # Use FPS to calculate time step

        # Check for goal reached
        if self._check_goal_reached():
            print("Goal reached!")
            reward = 100  # Positive reward
            done = True
            self.state = self._get_obs()
            return self.state, reward, done, False, {}

        # Check if the vehicle touches the border or collides
        if self._check_out_of_bounds() or self._check_collision():
            print("Vehicle out of bounds or collision detected.")
            reward = -10  # Negative reward
            done = True
            self.state = self._get_obs()
            return self.state, reward, done, False, {}

        # Update state
        self.state = self._get_obs()
        reward = self._get_reward()
        done = False

        return self.state, reward, done, False, {}

    def _get_obs(self):
        # Get the distance to the nearest obstacle and wall
        nearest_obstacle_distance = self._get_nearest_obstacle_distance()
        nearest_wall_distance = self._get_nearest_wall_distance()

        # Relative position to goal
        delta_x = self.parking_lot.goal_position[0] - self.vehicle.position[0]
        delta_y = self.parking_lot.goal_position[1] - self.vehicle.position[1]
        distance_to_goal = np.linalg.norm([delta_x, delta_y])
        angle_to_goal = np.arctan2(delta_y, delta_x) - self.vehicle.heading

        return np.array([
            delta_x,                 # Relative x position to goal
            delta_y,                 # Relative y position to goal
            self.vehicle.velocity,   # Vehicle speed
            self.vehicle.heading,    # Vehicle heading
            nearest_obstacle_distance,
            nearest_wall_distance
        ], dtype=np.float32)

    def _get_nearest_obstacle_distance(self):
        """
        Calculate the distance from the vehicle to the nearest obstacle
        """
        min_distance = float('inf')  # Initialize to infinity
        vehicle_position = np.array(self.vehicle.position)

        # Iterate through all static obstacles and calculate distances
        for obstacle in self.parking_lot.static_vehicles:
            obstacle_position = np.array(obstacle.position)
            distance = np.linalg.norm(vehicle_position - obstacle_position)
            min_distance = min(min_distance, distance)

        return min_distance

    def _get_nearest_wall_distance(self):
        """
        Calculate the distance from the vehicle to the walls
        """
        vehicle_position = np.array(self.vehicle.position)
        distances = [
            vehicle_position[0],  # Distance to the left wall
            SCREEN_WIDTH - vehicle_position[0],  # Distance to the right wall
            vehicle_position[1],  # Distance to the top wall
            SCREEN_HEIGHT - vehicle_position[1]   # Distance to the bottom wall
        ]
        return min(distances)  # Return the minimum distance

    def _get_reward(self):
        # Calculate the distance to the goal
        delta_x = self.parking_lot.goal_position[0] - self.vehicle.position[0]
        delta_y = self.parking_lot.goal_position[1] - self.vehicle.position[1]
        distance_to_goal = np.linalg.norm([delta_x, delta_y])

        # Calculate the change in distance to the goal
        delta_distance = self.prev_distance_to_goal - distance_to_goal

        # Reward is proportional to the progress towards the goal
        reward = delta_distance * 10  # Scale factor can be adjusted

        # Update previous distance
        self.prev_distance_to_goal = distance_to_goal

        return reward

    def _check_goal_reached(self):
        distance_to_goal = np.linalg.norm(
            np.array(self.vehicle.position) - np.array(self.parking_lot.goal_position)
        )
        if distance_to_goal < GOAL_THRESHOLD:
            return True
        else:
            return False

    def _check_out_of_bounds(self):
        """
        Check if the vehicle touches the border, considering wall thickness
        """
        wall_thickness = 10  # Wall thickness
        vehicle_corners = self._get_vehicle_corners(self.vehicle)
        for corner in vehicle_corners:
            x, y = corner
            if (x < wall_thickness or
                x > SCREEN_WIDTH - wall_thickness or
                y < wall_thickness or
                y > SCREEN_HEIGHT - wall_thickness):
                return True
        return False

    def _check_collision(self):
        """
        Use Separating Axis Theorem to check if the yellow vehicle collides with stationary blue vehicles
        """
        vehicle_corners = self._get_vehicle_corners(self.vehicle)
        for static_vehicle in self.parking_lot.static_vehicles:
            static_vehicle_corners = self._get_vehicle_corners(static_vehicle)
            if self._sat_collision(vehicle_corners, static_vehicle_corners):
                return True
        return False

    def _sat_collision(self, corners1, corners2):
        """
        Use Separating Axis Theorem to check if two convex polygons (defined by vertices) collide
        """
        for shape in [corners1, corners2]:
            for i in range(len(shape)):
                # Calculate the normal vector (axis) of the edge
                axis = np.array([shape[(i+1)%4][1] - shape[i][1], shape[i][0] - shape[(i+1)%4][0]])
                axis = axis / np.linalg.norm(axis)

                # Calculate projections of both shapes onto the axis
                proj1 = [np.dot(corner, axis) for corner in corners1]
                proj2 = [np.dot(corner, axis) for corner in corners2]

                # Check if projections overlap
                if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                    return False  # Found a separating axis, no collision
        return True  # All axes overlap, collision occurs

    def _get_vehicle_corners(self, vehicle):
        x, y = vehicle.position
        heading = vehicle.heading
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        half_width = VEHICLE_WIDTH / 2
        half_height = VEHICLE_HEIGHT / 2

        corners = [
            np.array([x + half_width * cos_h - half_height * sin_h, y + half_width * sin_h + half_height * cos_h]),
            np.array([x + half_width * cos_h + half_height * sin_h, y + half_width * sin_h - half_height * cos_h]),
            np.array([x - half_width * cos_h + half_height * sin_h, y - half_width * sin_h - half_height * cos_h]),
            np.array([x - half_width * cos_h - half_height * sin_h, y - half_width * sin_h + half_height * cos_h]),
        ]
        return corners

    def render(self):
        self.screen.fill((100, 100, 100))  # Gray background

        VehicleGraphics.draw_walls(self.screen, self.parking_lot.obstacles)

        VehicleGraphics.draw_parking_lanes(self.screen, LANE_WIDTH, LANE_HEIGHT, NUM_ROWS, NUM_COLS, SCREEN_WIDTH, SCREEN_HEIGHT)

        if self.parking_lot.goal_position:
            VehicleGraphics.draw_goal(self.screen, self.parking_lot.goal_position[0], self.parking_lot.goal_position[1])

        for static_vehicle in self.parking_lot.static_vehicles:
            VehicleGraphics.display(static_vehicle, self.screen, color=(0, 0, 255))  # Blue stationary vehicles

        for vehicle in self.parking_lot.vehicles:
            VehicleGraphics.display(vehicle, self.screen)

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()
