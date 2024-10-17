import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
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

# Additional global parameters
INITIAL_VEHICLE_X = (SCREEN_WIDTH - NUM_COLS * LANE_WIDTH) // 2 - 30
INITIAL_VEHICLE_Y = 50
STEERING_ANGLE = 0.1  # Steering angle
STEP_REWARD = -0.1  # Reward per step
FPS = 30

class MyNewEnv(gym.Env):
    def __init__(self):
        super(MyNewEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # Update the upper limit of the observation space to the maximum of screen dimensions
        self.observation_space = Box(low=0, high=max(SCREEN_WIDTH, SCREEN_HEIGHT), shape=(2,), dtype=np.float32)
        self.state = None
        self.parking_lot = None
        self.vehicle = None

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

                if self.parking_lot.goal_position and (abs(vehicle_x - self.parking_lot.goal_position[0]) < LANE_WIDTH // 2 and 
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

        goal_x = (start_x + random_col * LANE_WIDTH + (LANE_WIDTH - GOAL_SIZE) // 2) + 5
        goal_y = start_y + random_row * LANE_HEIGHT + LANE_HEIGHT // 4

        self.parking_lot.set_goal_position((goal_x, goal_y))

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

    def reset(self):
        self._create_parking_lot()
        self._create_vehicle()
        self._create_goal()
        self._create_static_vehicles()
        self._create_walls()

        self.state = np.array(self.vehicle.position, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # Parse action
        steering = action[0]*0.1  # Range [-0.1, 0.1]
        acceleration = action[1]  # Range [-1, 1]

        # Update vehicle state
        self.vehicle.action["steering"] = steering
        self.vehicle.action["acceleration"] = acceleration
        self.vehicle.step(1.0)  # Use FPS to calculate time step

        # Check if the vehicle touches the border or collides
        if self._check_out_of_bounds() or self._check_collision():
            print("Vehicle out of bounds or collision detected, resetting environment.")
            return self.reset()[0], -1, False, False, {}  # Return a negative reward and mark done=True

        # Update state
        self.state = self._get_obs()
        reward = self._get_reward()
        # done = self._check_goal()

        return self.state, reward, False, False, {}

    def _get_obs(self):
        # Return observation state, e.g., vehicle position, velocity, heading, etc.
        return np.array([
            *self.vehicle.position,
            self.vehicle.velocity,
            self.vehicle.heading,
            self.vehicle.steering_angle
        ], dtype=np.float32)

    def _get_reward(self):
        # Implement reward function
        # This is just an example, you may need to adjust according to specific requirements
        goal_distance = np.linalg.norm(self.vehicle.position - self.parking_lot.goal_position)
        return -goal_distance  # The closer to the target, the higher the reward

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
        Check if the yellow car collides with stationary blue cars
        """
        vehicle_corners = self._get_vehicle_corners(self.vehicle)
        for static_vehicle in self.parking_lot.static_vehicles:
            static_vehicle_corners = self._get_vehicle_corners(static_vehicle)
            if self._check_rect_collision(vehicle_corners, static_vehicle_corners):
                return True
        return False

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

    def _check_rect_collision(self, corners1, corners2):
        """
        Check if two rectangles (represented by four vertices) collide
        """
        for corner in corners1:
            if self._point_in_rect(corner, corners2):
                return True
        for corner in corners2:
            if self._point_in_rect(corner, corners1):
                return True
        return False

    def _point_in_rect(self, point, rect_corners):
        """
        Check if a point is inside a rectangle defined by four corners
        """
        x, y = point
        rect = pygame.Rect(
            min(corner[0] for corner in rect_corners),
            min(corner[1] for corner in rect_corners),
            max(corner[0] for corner in rect_corners) - min(corner[0] for corner in rect_corners),
            max(corner[1] for corner in rect_corners) - min(corner[1] for corner in rect_corners)
        )
        return rect.collidepoint(x, y)

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
