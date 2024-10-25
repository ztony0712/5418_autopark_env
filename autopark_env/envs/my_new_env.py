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
MAX_SPEED = 15
NUM_COLS = 10
NUM_ROWS = 3
LANE_WIDTH = 40
LANE_HEIGHT = 100
VEHICLE_WIDTH = 40
VEHICLE_HEIGHT = 20
GOAL_SIZE = 25
STATIC_VEHICLE_PROBABILITY = 0
SAFE_DISTANCE = 30  # Safety threshold for distance to obstacles

# Additional global parameters
# INITIAL_VEHICLE_X = (SCREEN_WIDTH - NUM_COLS * LANE_WIDTH) // 2 - 30
INITIAL_VEHICLE_X = 400
INITIAL_VEHICLE_Y = 300
MAX_STEPS = 300
FPS = 1


class MyNewEnv(gym.Env):
    def __init__(self):
        super(MyNewEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # 修改observation_space的定义
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(
                low=np.array([0, 0, -MAX_SPEED, -MAX_SPEED, -1, -1], dtype=np.float32),  # x, y, vx, vy, cos_h, sin_h
                high=np.array([800, 600, MAX_SPEED, MAX_SPEED, 1, 1], dtype=np.float32),
                dtype=np.float32
            ),
            'achieved_goal': gym.spaces.Box(  # 当前位置和朝向
                low=np.array([0, 0, -MAX_SPEED, -MAX_SPEED, -1, -1], dtype=np.float32),
                high=np.array([800, 600, MAX_SPEED, MAX_SPEED, 1, 1], dtype=np.float32),
                dtype=np.float32
            ),
            'desired_goal': gym.spaces.Box(   # 目标位置和朝向
                low=np.array([0, 0, -MAX_SPEED, -MAX_SPEED, -1, -1], dtype=np.float32),
                high=np.array([800, 600, MAX_SPEED, MAX_SPEED, 1, 1], dtype=np.float32),
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

        # 添加步数计数器
        self.step_count = 0

        # 添加碰撞统计
        self.episode_count = 0
        self.collision_penalty = 5    # 进一步降低碰撞惩罚
        self.max_steps = MAX_STEPS   # 减少最大步数
        self.safe_distance = SAFE_DISTANCE        # 增加安全距离

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
        # start_x = (SCREEN_WIDTH - NUM_COLS * LANE_WIDTH) // 2
        # start_y = (SCREEN_HEIGHT - NUM_ROWS * LANE_HEIGHT) // 2

        # random_col = random.randint(0, NUM_COLS - 2)
        # random_row = random.randint(0, NUM_ROWS - 1)

        # goal_x = (start_x + random_col * LANE_WIDTH + (LANE_WIDTH - GOAL_SIZE) // 2) + 5
        # goal_y = start_y + random_row * LANE_HEIGHT + LANE_HEIGHT // 4

        goal_x = 490
        goal_y = 380

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
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 重置步数计数器
        self.step_count = 0
        
        # 原有的重置逻辑
        self._create_parking_lot()
        self._create_vehicle()
        self._create_goal()
        self._create_static_vehicles()
        self._create_walls()

        # 初始化状态
        self.state = self._get_obs()
        
        # 增加episode计数
        self.episode_count += 1
        
        # 返回初始观察和空的info字典
        return self.state, {}

    def step(self, action):
        # 增加步数计数
        self.step_count += 1
        
        # 解析动作
        steering = action[0]
        acceleration = action[1]

        # 更新车辆状态
        self.vehicle.action["steering"] = steering
        self.vehicle.action["acceleration"] = acceleration
        self.vehicle.step(1/FPS)  # 使用 1.0 秒作为时间步长

        # 获取新的状态
        self.state = self._get_obs()
        
        # 计算加权p范数距离
        achieved_state = self.state['achieved_goal']     # shape: (6,)
        desired_state = self.state['desired_goal']       # shape: (6,)

        # 修改为使用 unwrapped 访问 compute_reward
        reward = self.unwrapped.compute_reward(achieved_state, desired_state, None)
        
        # 检查是否碰撞
        collision = self._check_collision() or self._check_out_of_bounds()
        if collision:
            reward -= self.collision_penalty # 碰撞惩罚
            terminated = True
            truncated = False
            info = {'is_success': False}
            return self.state, reward, terminated, truncated, info

        # 检查是否到达目标
        success = self._is_success(self.state['achieved_goal'], self.state['desired_goal'])
        if success:
            terminated = True
            truncated = False
            info = {'is_success': True}
            return self.state, reward, terminated, truncated, info
        
        # 检查是否超出最大步数
        truncated = self.step_count >= self.max_steps
        terminated = False

        info = {'is_success': False}
        
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
        # 获取当前车辆状态
        current_pos = self.vehicle.position
        current_heading = self.vehicle.heading
        current_heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)])
        
        # 计算速度在x和y方向的分量
        velocity = self.vehicle.velocity
        vx = velocity * np.cos(current_heading)  # x方向速度分量
        vy = velocity * np.sin(current_heading)  # y方向速度分量
        
        # 目标朝向的向量
        goal_heading_vec = np.array([np.cos(self.goal_heading), np.sin(self.goal_heading)])
        
        # 目标速度（假设目标是静止的）
        goal_vx, goal_vy = 0.0, 0.0
        
        return {
            'observation': np.array([
                current_pos[0],           # x coordinate
                current_pos[1],           # y coordinate
                vx,                       # velocity x component
                vy,                       # velocity y component
                current_heading_vec[0],   # heading cos
                current_heading_vec[1],   # heading sin
            ], dtype=np.float32),
            'achieved_goal': np.array([
                current_pos[0],           # x coordinate
                current_pos[1],           # y coordinate
                vx,                       # velocity x component
                vy,                       # velocity y component
                current_heading_vec[0],   # heading cos
                current_heading_vec[1],   # heading sin
            ], dtype=np.float32),
            'desired_goal': np.array([
                self.parking_lot.goal_position[0],  # goal x
                self.parking_lot.goal_position[1],  # goal y
                goal_vx,                           # goal velocity x (0)
                goal_vy,                           # goal velocity y (0)
                goal_heading_vec[0],               # goal heading cos
                goal_heading_vec[1],               # goal heading sin
            ], dtype=np.float32)
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
        self.clock.tick(30)

    def close(self):
        pygame.quit()

    def _is_success(self, achieved_goal, desired_goal):
        """
        判断智能体是否成功到达目标。
        参数:
            achieved_goal: 实际达到的目标，形状为 (4,)
            desired_goal: 期望的目标，形状为 (4,)
        返回:
            bool: 是否成功
        """
        # pos_distance = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        # heading_similarity = np.dot(achieved_goal[2:], desired_goal[2:])
        # return pos_distance < GOAL_SIZE and heading_similarity > 0.95
        return (
            self.compute_reward(achieved_goal, desired_goal, {})
            > -0.12  # 原来是-0.12，稍微放宽一些
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        weights = np.array([1, 0.3, 0, 0, 0.02, 0.02])
        
        # 归一化处理
        achieved_normalized = achieved_goal / np.array([800, 600, 10, 10, 1, 1])  # 注意速度项的归一化值改为10
        desired_normalized = desired_goal / np.array([800, 600, 10, 10, 1, 1])
        
        distance = np.power(
            np.dot(
                np.abs(achieved_normalized - desired_normalized),
                weights,
            ),
            0.5,
        )
        
        # 将奖励限制在合理范围内
        reward = -np.clip(distance, 0, 5)
        
        return reward













