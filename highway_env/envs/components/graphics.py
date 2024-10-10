import pygame
import numpy as np

class VehicleGraphics:
    @staticmethod
    def display(vehicle, screen, color=(255, 255, 0)):
        """
        在屏幕上显示车辆
        """
        vehicle_surface = pygame.Surface((40, 20))  # 创建一个表示车辆的表面，尺寸为 40x20
        vehicle_surface.set_colorkey((0, 0, 0))  # 设置透明色
        vehicle_surface.fill(color)  # 填充指定颜色
        rotated_vehicle = pygame.transform.rotate(vehicle_surface, -np.degrees(vehicle.heading))  # 旋转车辆表面，使其朝向当前的 heading
        vehicle_rect = rotated_vehicle.get_rect(center=(int(vehicle.position[0]), int(vehicle.position[1])))  # 获取旋转后的位置矩形
        screen.blit(rotated_vehicle, vehicle_rect.topleft)  # 绘制旋转后的车辆

    @staticmethod
    def draw_parking_lanes(screen, lane_width, lane_height, num_rows, num_cols, screen_width, screen_height):
        """
        绘制停车位的白色线条并确保居中和竖直显示
        """
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
        """
        绘制目标停车点
        """
        goal_rect = pygame.Rect(goal_x, goal_y, goal_width, goal_height)
        pygame.draw.rect(screen, (0, 255, 0), goal_rect)  # 绿色的停车目标

    @staticmethod
    def draw_walls(screen, walls):
        """
        绘制墙壁
        """
        for wall in walls:
            wall_rect = pygame.Rect(int(wall.position[0]), int(wall.position[1]), 70, 10)  # 墙的大小
            pygame.draw.rect(screen, (255, 0, 0), wall_rect)  # 红色的墙
