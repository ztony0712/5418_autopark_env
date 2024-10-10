import numpy as np

class Vehicle:
    def __init__(self, road, position, heading=0, steering_angle=0, velocity=5.0):
        self.road = road
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading  # 朝向角度
        self.steering_angle = steering_angle  # 转向角度
        self.velocity = velocity  # 固定速度，沿车辆的前进方向移动

    def step(self, dt):
        # 更新朝向（转向角度影响朝向）
        self.heading += self.steering_angle * dt
        
        # 更新位置（沿着当前朝向前进）
        self.position[0] += self.velocity * dt * np.cos(self.heading)
        self.position[1] += self.velocity * dt * np.sin(self.heading)
