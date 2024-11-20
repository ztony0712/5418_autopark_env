import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box
import pygame
import random
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from .components.parking_lot import ParkingLot
from .components.vehicle import StaticVehicle, Vehicle
from .components.obstacle import Obstacle
from .components.graphics import VehicleGraphics
from .components.mpc_controller import MPCController

# Global parameters
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
MAX_SPEED = 15
NUM_COLS = 10
NUM_ROWS = 3
LANE_WIDTH = 40
LANE_HEIGHT = 100
VEHICLE_WIDTH = 40
VEHICLE_HEIGHT = 20

GOAL_SIZE = 15
MIN_HEADING_SIMILARITY = 0.9

STATIC_VEHICLE_PROBABILITY = 0.1
SAFE_DISTANCE = 30  # Safety threshold for distance to obstacles

# Additional global parameters
INITIAL_VEHICLE_X = (SCREEN_WIDTH - NUM_COLS * LANE_WIDTH) // 2 - 30
# INITIAL_VEHICLE_X = 400
INITIAL_VEHICLE_Y = 100
MAX_STEPS = 200
FPS = 1


class MyNewEnv(gym.Env):
    def __init__(self, render_mode=None, use_mpc_freq=5):
        super(MyNewEnv, self).__init__()
        self.render_mode = render_mode
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 30,
        }
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Modify the definition of observation_space
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=np.array([0, 0, -1, -1, -1, -1], dtype=np.float32),  # x, y, vx, vy, cos_h, sin_h
                high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
                dtype=np.float32
            ),
            'achieved_goal': gym.spaces.Box(
                low=np.array([0, 0, -1, -1, -1, -1], dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
                dtype=np.float32
            ),
            'desired_goal': gym.spaces.Box(
                low=np.array([0, 0, -1, -1, -1, -1], dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
                dtype=np.float32
            )
        })
        self.state = None
        self.parking_lot = None
        self.vehicle = None
        self.goal_heading = 0  # Initialize goal heading

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Parking Lot Visualization")
        self.clock = pygame.time.Clock()

        # Add step counter
        self.step_count = 0

        # Add collision statistics
        self.episode_count = 0
        self.collision_penalty = 10
        self.max_steps = MAX_STEPS
        self.safe_distance = SAFE_DISTANCE

        # 初始化MPC控制器
        self.mpc_controller = MPCController(self)

        self.use_mpc_freq = use_mpc_freq

    @property
    def render_modes(self):
        """Implement render_modes property"""
        return self.metadata["render_modes"]

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

        goal_x = (start_x + random_col * LANE_WIDTH + (LANE_WIDTH - GOAL_SIZE) // 2) + 5
        goal_y = start_y + random_row * LANE_HEIGHT + LANE_HEIGHT // 4

        # goal_x = 490
        # goal_y = 380

        self.parking_lot.set_goal_position((goal_x, goal_y))

        # Set the goal heading based on the goal position relative to the vehicle's current position
        self.goal_heading = np.pi / 2  # 竖直朝向

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
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Original reset logic
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
        self.step_count += 1
        # 确保动作在 [-1, 1] 范围内
        action = np.clip(action, -1, 1)
        
        # 将动作映射到车辆模型的有效范围
        steering = action[0] * Vehicle.MAX_STEERING_ANGLE
        acceleration = action[1] * Vehicle.MAX_ACCELERATION
        
        # 只在特定步数使用MPC
        if self.step_count % self.use_mpc_freq == 0:
            vehicle_state = self.vehicle.get_state()
            corrected_action = self.mpc_controller.optimize(vehicle_state, [steering, acceleration])
        else:
            corrected_action = [steering, acceleration]
            
        # 更新车辆状态
        self.vehicle.action["steering"] = corrected_action[0]
        self.vehicle.action["acceleration"] = corrected_action[1]
        self.vehicle.step(1.0 / FPS)
        
        # Get new state
        self.state = self._get_obs()
        
        # Calculate weighted p-norm distance
        achieved_state = self.state['achieved_goal']     # shape: (6,)
        desired_state = self.state['desired_goal']       # shape: (6,)

        # Modify to use unwrapped access to compute_reward
        reward = self.unwrapped.compute_reward(achieved_state, desired_state, None)
        
        # Calculate action penalty to discourage spinning in place
        # action_penalty = 0.1 * (steering ** 2 + acceleration ** 2)
        # reward -= action_penalty

        # Check for collision
        collision = self._check_collision() or self._check_out_of_bounds()
        if collision:
            reward -= self.collision_penalty # Collision penalty
            terminated = True
            truncated = False
            info = {'is_success': False}
            return self.state, reward, terminated, truncated, info

        # Check if goal is reached
        success = self.check_goal_reached(self.state['achieved_goal'], self.state['desired_goal'])
        if success:
            terminated = True
            truncated = False
            info = {'is_success': True}
            return self.state, reward, terminated, truncated, info
        
        # Check if maximum steps exceeded
        truncated = self.step_count >= self.max_steps
        terminated = False

        info = {'is_success': success}
        
        return self.state, reward, terminated, truncated, info

    # def choose_action(self, state):
    #     # state contains all the state information as an array
    #     nearest_obstacle_distance = state[4]  # Assuming this is the distance to the nearest obstacle

    #     # Choose actions based on the distance to the obstacle
    #     if nearest_obstacle_distance < SAFE_DISTANCE:  # SAFE_DISTANCE is a threshold
    #         action = np.array([0.1, -1])  # Slow down and slightly steer
    #     else:
    #         action = np.array([0.1, 1])  # Normal acceleration

    #     return action

    def _get_obs(self):
        # Get current vehicle state
        current_pos = self.vehicle.position
        current_heading = self.vehicle.heading
        current_heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)])
        
        # Calculate velocity components
        velocity = self.vehicle.velocity
        vx = velocity * np.cos(current_heading)  # x direction velocity component
        vy = velocity * np.sin(current_heading)  # y direction velocity component
        
        # Goal heading vector
        goal_heading_vec = np.array([np.cos(self.goal_heading), np.sin(self.goal_heading)])
        
        # 统一归一化处理
        normalized_pos = np.array(current_pos) / np.array([SCREEN_WIDTH, SCREEN_HEIGHT])
        normalized_goal_pos = np.array(self.parking_lot.goal_position) / np.array([SCREEN_WIDTH, SCREEN_HEIGHT])
        normalized_vx = vx / MAX_SPEED
        normalized_vy = vy / MAX_SPEED
        
        return {
            'observation': np.concatenate([
                normalized_pos,
                [normalized_vx, normalized_vy],
                current_heading_vec
            ]).astype(np.float32),
            'achieved_goal': np.concatenate([
                normalized_pos,
                [normalized_vx, normalized_vy],
                current_heading_vec
            ]).astype(np.float32),
            'desired_goal': np.concatenate([
                normalized_goal_pos,
                [0.0, 0.0],
                goal_heading_vec
            ]).astype(np.float32)
        }

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
        if self.render_mode is None:
            return

        self.screen.fill((100, 100, 100))  # Gray background

        VehicleGraphics.draw_walls(self.screen, self.parking_lot.obstacles)
        VehicleGraphics.draw_parking_lanes(self.screen, LANE_WIDTH, LANE_HEIGHT, NUM_ROWS, NUM_COLS, SCREEN_WIDTH, SCREEN_HEIGHT)

        if self.parking_lot.goal_position:
            VehicleGraphics.draw_goal(self.screen, self.parking_lot.goal_position[0], self.parking_lot.goal_position[1])

        for static_vehicle in self.parking_lot.static_vehicles:
            VehicleGraphics.display(static_vehicle, self.screen, color=(0, 0, 255))

        for vehicle in self.parking_lot.vehicles:
            VehicleGraphics.display(vehicle, self.screen)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(30)
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))
        
    def update_metadata(self, video_real_time_ratio=2):
        frames_freq = (
            30
            if self._record_video_wrapper
            else 30
        )
        self.metadata["render_fps"] = video_real_time_ratio * frames_freq
        
    def set_record_video_wrapper(self, wrapper: RecordVideo):
        self._record_video_wrapper = wrapper
        self.update_metadata()

    def close(self):
        pygame.quit()

    def check_goal_reached(self, achieved_goal, desired_goal):
        """检查是否到达目标
        """
        # 将归一化的位置转换回实际像素坐标
        achieved_pos = achieved_goal[:2] * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])
        desired_pos = desired_goal[:2] * np.array([SCREEN_WIDTH, SCREEN_HEIGHT])
        
        # 计算位置距离
        pos_distance = np.linalg.norm(achieved_pos - desired_pos)
        
        # 计算航向相似度
        heading_similarity = np.dot(achieved_goal[4:6], desired_goal[4:6])
        
        # 使用更宽松的阈值
        return pos_distance < GOAL_SIZE and heading_similarity > MIN_HEADING_SIMILARITY

    def compute_reward(self, achieved_goal, desired_goal, info):
        """计算奖励值
        Args:
            achieved_goal: 已经归一化的目标 [x, y, vx, vy, cos_h, sin_h]
            desired_goal: 已经归一化的期望目标
        """
        # 直接使用已经归一化的值，不需要再次归一化
        weights = np.array([1, 0.3, 0, 0, 0.02, 0.02])
        
        # 处理批处理情况
        if achieved_goal.ndim == 2:
            distance = np.power(
                np.sum(
                    np.abs(achieved_goal - desired_goal) * weights,
                    axis=1
                ),
                0.5,
            )
        else:
            distance = np.power(
                np.dot(
                    np.abs(achieved_goal - desired_goal),
                    weights,
                ),
                0.5,
            )
        
        # Limit reward to a reasonable range
        reward = -np.clip(distance, 0, 5)
        
        return reward



