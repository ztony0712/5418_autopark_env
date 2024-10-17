import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import pygame
import random
import gymnasium as gym

from .components.lane import LineType, Lane, StraightLane, CircularLane
from .components.road import RoadNetwork, Road
from .components.vehicle import StaticVehicle, Vehicle
from .components.landmark import Landmark, Obstacle
from .components.graphics import VehicleGraphics

# 全局参数
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

# 新增全局参数
INITIAL_VEHICLE_X = (SCREEN_WIDTH - NUM_COLS * LANE_WIDTH) // 2 - 30
INITIAL_VEHICLE_Y = 50
STEERING_ANGLE = 0.1  # 转向角度
STEP_REWARD = -0.1  # 每步的奖励
FPS = 30  # 帧率

class MyNewEnv(gym.Env):
    def __init__(self):
        super(MyNewEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # 更新观察空间的上限为屏幕尺寸的最大值
        self.observation_space = Box(low=0, high=max(SCREEN_WIDTH, SCREEN_HEIGHT), shape=(2,), dtype=np.float32)
        self.state = None
        self.road = None
        self.vehicle = None
        self.static_vehicles = []  # 静止车辆列表
        self.walls = []
        self.goal_position = None

        # 初始化 pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Parking Lot Visualization")
        self.clock = pygame.time.Clock()

    def _create_road(self):
        net = RoadNetwork()

        # 创建直道
        straight_lane = StraightLane([0, 0], [100, 0], width=4.0)
        net.add_lane("a", "b", straight_lane)

        # 创建弯道
        circular_lane = CircularLane([100, 100], radius=50, start_angle=-np.pi / 2, end_angle=0, clockwise=False)
        net.add_lane("b", "c", circular_lane)

        self.road = Road(network=net)

    def _create_vehicle(self):
        # 使用全局变量设置初始位置
        self.vehicle = Vehicle([INITIAL_VEHICLE_X, INITIAL_VEHICLE_Y], heading=0)
        self.road.vehicles.append(self.vehicle)

    def _create_static_vehicles(self):
        # 清除之前的静止车辆
        self.static_vehicles.clear()
        start_x = (SCREEN_WIDTH - NUM_COLS * LANE_WIDTH) // 2
        start_y = (SCREEN_HEIGHT - NUM_ROWS * LANE_HEIGHT) // 2

        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                vehicle_x = start_x + col * LANE_WIDTH + LANE_WIDTH // 2
                vehicle_y = start_y + row * LANE_HEIGHT + LANE_HEIGHT // 4

                if self.goal_position and (abs(vehicle_x - self.goal_position[0]) < LANE_WIDTH // 2 and 
                                           abs(vehicle_y - self.goal_position[1]) < LANE_HEIGHT // 2):
                    continue

                if random.random() < STATIC_VEHICLE_PROBABILITY: # 以 30% 的概率放置静止车辆
                    static_vehicle = StaticVehicle([vehicle_x, vehicle_y], heading=np.pi / 2) # 竖直方向，中心对齐停车位
                    self.static_vehicles.append(static_vehicle)

    def _create_goal(self):
        start_x = (SCREEN_WIDTH - NUM_COLS * LANE_WIDTH) // 2
        start_y = (SCREEN_HEIGHT - NUM_ROWS * LANE_HEIGHT) // 2

        random_col = random.randint(0, NUM_COLS - 2)
        random_row = random.randint(0, NUM_ROWS - 1)

        goal_x = (start_x + random_col * LANE_WIDTH + (LANE_WIDTH - GOAL_SIZE) // 2) + 5
        goal_y = start_y + random_row * LANE_HEIGHT + LANE_HEIGHT // 4

        self.goal_position = (goal_x, goal_y)

    def _create_walls(self):
        # 创建墙壁并随机放置
        width, height = 70, 10
        self.walls = [
            Obstacle(self.road, [50, 50]),  # 你可以根据需要调整墙壁位置
            Obstacle(self.road, [200, 100]),
            Obstacle(self.road, [400, 200])
        ]

    def reset(self):
        self._create_road()
        self._create_vehicle()
        self._create_goal()
        self._create_static_vehicles()
        self._create_walls()

        self.state = np.array(self.vehicle.position, dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # 解析动作
        steering = action[0]*0.1  # 范围 [-0.1, 0.1]
        acceleration = action[1]  # 范围 [-1, 1]

        # 更新车辆状态
        self.vehicle.action["steering"] = steering
        self.vehicle.action["acceleration"] = acceleration
        self.vehicle.step(1.0)  # 使用 FPS 来计算时间步长

        # 检查车辆是否触碰边框或发生碰撞
        if self._check_out_of_bounds() or self._check_collision():
            print("Vehicle out of bounds or collision detected, resetting environment.")
            return self.reset()[0], -1, False, False, {}  # 返回一个负的奖励，并标记 done=True

        # 更新状态
        self.state = self._get_obs()
        reward = self._get_reward()
        # done = self._check_goal()

        return self.state, reward, False, False, {}

    def _get_obs(self):
        # 返回观察状态，例如车辆位置、速度、朝向等
        return np.array([
            *self.vehicle.position,
            self.vehicle.velocity,
            self.vehicle.heading,
            self.vehicle.steering_angle
        ], dtype=np.float32)

    def _get_reward(self):
        # 实现奖励函数
        # 这里只是一个示例，您可能需要根据具体需求调整
        goal_distance = np.linalg.norm(self.vehicle.position - self.goal_position)
        return -goal_distance  # 距离目标越近，奖励越高

    def _check_out_of_bounds(self):
        """
        检查车辆是否触碰边框
        """
        vehicle_corners = self._get_vehicle_corners(self.vehicle)
        for corner in vehicle_corners:
            x, y = corner
            if x < 0 or x > SCREEN_WIDTH or y < 0 or y > SCREEN_HEIGHT:
                return True
        return False

    def _check_collision(self):
        """
        检查黄车是否与静止的蓝车发生碰撞
        """
        vehicle_corners = self._get_vehicle_corners(self.vehicle)
        for static_vehicle in self.static_vehicles:
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
        检查两个矩形（由四个顶点表示）是否碰撞
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
        检查一个点是否在由四个角定义的矩形内
        """
        x, y = point
        rect = pygame.Rect(
            min(corner[0] for corner in rect_corners),
            min(corner[1] for corner in rect_corners),
            max(corner[0] for corner in rect_corners) - min(corner[0] for corner in rect_corners),
            max(corner[1] for corner in rect_corners) - min(corner[1] for corner in rect_corners)
        )
        return rect.collidepoint(x, y)

    def render(self, mode='human'):
        self.screen.fill((100, 100, 100))  # 灰色背景

        VehicleGraphics.draw_parking_lanes(self.screen, LANE_WIDTH, LANE_HEIGHT, NUM_ROWS, NUM_COLS, SCREEN_WIDTH, SCREEN_HEIGHT)

        if self.goal_position:
            VehicleGraphics.draw_goal(self.screen, self.goal_position[0], self.goal_position[1])

        for static_vehicle in self.static_vehicles:
            VehicleGraphics.display(static_vehicle, self.screen, color=(0, 0, 255))  # 蓝色的静止车辆

        for vehicle in self.road.vehicles:
            VehicleGraphics.display(vehicle, self.screen)

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.quit()
