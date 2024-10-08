import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import pygame
import random

# ----------- 停车位和车道的类 -----------

class LineType:
    CONTINUOUS = 0
    BROKEN = 1

class StraightLane:
    def __init__(self, start, end, width=4.0, line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)):
        self.start = np.array(start)
        self.end = np.array(end)
        self.width = width
        self.line_types = line_types

class CircularLane:
    def __init__(self, center, radius, start_angle, end_angle, clockwise=True, width=4.0):
        self.center = np.array(center)
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.clockwise = clockwise
        self.width = width

    def position(self, angle):
        return self.center + self.radius * np.array([np.cos(angle), np.sin(angle)])

class RoadNetwork:
    def __init__(self):
        self.lanes = []

    def add_lane(self, start, end, lane):
        self.lanes.append((start, end, lane))

    def lanes_list(self):
        return self.lanes

class Road:
    def __init__(self, network):
        self.network = network
        self.vehicles = []
        self.objects = []

class Vehicle:
    def __init__(self, road, position, heading=0, steering_angle=0):
        self.road = road
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading  # 朝向角度
        self.steering_angle = steering_angle  # 转向角度
        self.velocity = 5.0  # 固定速度，沿车辆的前进方向移动

    def step(self, dt):
        # 更新朝向（转向角度影响朝向）
        self.heading += self.steering_angle * dt
        
        # 更新位置（沿着当前朝向前进）
        self.position[0] += self.velocity * dt * np.cos(self.heading)
        self.position[1] += self.velocity * dt * np.sin(self.heading)

class Landmark:
    def __init__(self, road, position, heading=0):
        self.road = road
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading

class Obstacle:
    def __init__(self, road, position, heading=0):
        self.road = road
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading

class VehicleGraphics:
    @staticmethod
    def display(vehicle, screen):
        """在屏幕上显示车辆"""
        vehicle_surface = pygame.Surface((40, 20))  # 创建一个表示车辆的表面，尺寸为 40x20
        vehicle_surface.set_colorkey((0, 0, 0))  # 设置透明色
        vehicle_surface.fill((255, 255, 0))  # 填充黄色
        rotated_vehicle = pygame.transform.rotate(vehicle_surface, -np.degrees(vehicle.heading))  # 旋转车辆表面，使其朝向当前的 heading
        vehicle_rect = rotated_vehicle.get_rect(center=(int(vehicle.position[0]), int(vehicle.position[1])))  # 获取旋转后的位置矩形
        screen.blit(rotated_vehicle, vehicle_rect.topleft)  # 绘制旋转后的车辆

    @staticmethod
    def draw_parking_lanes(screen, lane_width, lane_height, num_rows, num_cols, screen_width, screen_height):
        """绘制停车位的白色线条并确保居中和竖直显示"""
        total_width = num_cols * lane_width  # 总宽度
        total_height = num_rows * lane_height  # 总高度
        
        # 计算居中的起始位置
        start_x = (screen_width - total_width) // 2
        start_y = (screen_height - total_height) // 2

        # 绘制竖直的停车位线条
        for row in range(num_rows):
            for col in range(num_cols):
                x = start_x + col * lane_width  # 每列的 x 坐标
                y_start = start_y + row * lane_height  # 每排的 y 坐标
                y_end = y_start + lane_height // 2  # 缩短线条长度
                pygame.draw.line(screen, (255, 255, 255), (x, y_start), (x, y_end), 2)  # 绘制竖直线

    @staticmethod
    def draw_goal(screen, goal_x, goal_y, goal_width=15, goal_height=15):
        """绘制目标停车点"""
        goal_rect = pygame.Rect(goal_x, goal_y, goal_width, goal_height)
        pygame.draw.rect(screen, (0, 255, 0), goal_rect)  # 绿色的停车目标

    @staticmethod
    def draw_walls(screen, walls):
        """绘制墙壁"""
        for wall in walls:
            wall_rect = pygame.Rect(int(wall.position[0]), int(wall.position[1]), 70, 10)  # 墙的大小
            pygame.draw.rect(screen, (255, 0, 0), wall_rect)  # 红色的墙

# ----------- 定义 MyNewEnv 类 -----------

class MyNewEnv(Env):
    def __init__(self):
        super(MyNewEnv, self).__init__()
        self.action_space = Discrete(3)  # 三个动作：左转、直行、右转
        self.observation_space = Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.state = None
        self.road = None
        self.vehicle = None
        self.walls = []

        # 初始化 pygame
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Parking Lot Visualization")
        self.clock = pygame.time.Clock()

        # 初始化 goal 的位置
        self.goal_position = None

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
        # 固定车辆的初始位置（位于左上角第一根白线的左上方）
        num_cols = 10  # 白线的列数
        lane_width = 40  # 每个车位的宽度
        start_x = (self.screen_width - num_cols * lane_width) // 2  # 左上角第一根白线的x坐标

        # 车辆初始位置位于左上角第一根白线的左上方一点
        initial_x = start_x - 30  # 偏移量使得车辆在白线的左上方
        initial_y = 50  # 一个适当的y位置（根据实际布局调整）

        self.vehicle = Vehicle(self.road, [initial_x, initial_y], heading=0)  # 固定初始位置
        self.road.vehicles.append(self.vehicle)

    def _create_goal(self):
        # 随机选择 goal 的位置
        num_cols = 10
        num_rows = 3
        lane_width = 40
        lane_height = 100
        goal_size = 25  # 确保 goal 的大小为 25x25
        start_x = (self.screen_width - num_cols * lane_width) // 2
        start_y = (self.screen_height - num_rows * lane_height) // 2

        random_col = random.randint(0, num_cols - 2)  # 随机选择一列
        random_row = random.randint(0, num_rows - 1)  # 随机选择一行

        # 计算 goal 的位置，确保其在两条线之间居中
        goal_x = (start_x + random_col * lane_width + (lane_width - goal_size) // 2) + 5  # 偏移，使其在停车位中间
        goal_y = start_y + random_row * lane_height + lane_height // 4  # 偏移，使其在停车位中间

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
        self._create_walls()

        self.state = np.array([self.vehicle.position[0], self.vehicle.position[1]], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        # 根据动作更新车辆的转向角度
        if action == 0:  # 左转
            self.vehicle.steering_angle = -0.1
        elif action == 1:  # 右转
            self.vehicle.steering_angle = 0.1
        elif action == 2:  # 直行
            self.vehicle.steering_angle = 0.0

        # 更新车辆的位置和朝向
        self.vehicle.step(1.0)

        # 获取当前车辆的位置
        self.state = np.array([self.vehicle.position[0], self.vehicle.position[1]], dtype=np.float32)

        # 定义运动范围
        x_min, x_max = 0, self.screen_width
        y_min, y_max = 0, self.screen_height

        # 检查车辆是否超出运动范围
        if self.vehicle.position[0] < x_min or self.vehicle.position[0] > x_max or self.vehicle.position[1] < y_min or self.vehicle.position[1] > y_max:
            print("Vehicle out of bounds, resetting environment.")
            self.reset()  # 车辆超出范围，重置环境
            return self.state, -1, False, False, {}  # 返回一个负的奖励，并标记 `done=True`

        # 车辆到达目标位置的检测
        done = self.state[0] >= 150 and self.state[1] == 50  # 可以根据实际需要调整目标位置的条件
        reward = 1 if done else -0.1
        return self.state, reward, done, False, {}

    def render(self, mode='human'):
        # 绘制背景
        self.screen.fill((100, 100, 100))  # 灰色背景

        # 绘制三排停车位
        VehicleGraphics.draw_parking_lanes(self.screen, lane_width=40, lane_height=100, num_rows=3, num_cols=10, screen_width=self.screen_width, screen_height=self.screen_height)

        # 绘制目标停车位
        if self.goal_position:
            VehicleGraphics.draw_goal(self.screen, self.goal_position[0], self.goal_position[1])

        # # 绘制墙壁
        # VehicleGraphics.draw_walls(self.screen, self.walls)

        # 绘制车辆
        for vehicle in self.road.vehicles:
            VehicleGraphics.display(vehicle, self.screen)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()