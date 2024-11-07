import numpy as np
from typing import Dict, List

class Vehicle:
    LENGTH: float = 4.0
    MAX_STEERING_ANGLE: float = np.pi / 20  # 约18*2度，可根据需要调整
    MAX_STEERING_CHANGE: float = np.pi / 60  # 每步最大转向角变化，可根据需要调整
    MAX_SPEED: float = 10.0
    MIN_SPEED: float = -10.0  # 允许倒车
    MAX_ACCELERATION: float = 0.3  # 最大加速度
    MAX_STEERING = 0.1  # 与环境中的缩放一致

    def __init__(self, position: List[float], heading: float = 0, velocity: float = 0):
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading
        self.velocity = velocity
        
        self.action: Dict[str, float] = {
            "steering": 0,
            "acceleration": 0
        }
        
        self.crashed = False
        self.steering_angle = 0  # 初始化转向角

    def act(self, action: Dict[str, float]) -> None:
        """存储一个动作，等待执行"""
        if action:
            self.action.update(action)

    def step(self, dt: float) -> None:
        """更新车辆状态"""
        self.clip_actions()

        # 计算转向角的变化
        steering_change = self.action["steering"] * self.MAX_STEERING_CHANGE

        # 更新转向角，并限制在最大转向角范围内
        self.steering_angle = np.clip(
            self.steering_angle + steering_change,
            -self.MAX_STEERING_ANGLE,
            self.MAX_STEERING_ANGLE
        )

        # 更新速度
        self.velocity += self.action["acceleration"] * dt
        self.velocity = np.clip(self.velocity, self.MIN_SPEED, self.MAX_SPEED)

        # 计算角速度���基于自行车模型）
        # 确保使用正切函数计算角速度
        if np.cos(self.steering_angle) != 0:  # 防止除以零
            angular_velocity = (self.velocity / self.LENGTH) * np.tan(self.steering_angle)
        else:
            angular_velocity = 0  # 如果方向角为±π/2，设置角速度为0

        # 更新朝向
        self.heading += angular_velocity * dt

        # 更新位置（基于当前速度和朝向）
        self.position += self.velocity * dt * np.array([np.cos(self.heading), np.sin(self.heading)])

        # 确保朝向角在 -π 到 π 之间
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi


    def clip_actions(self) -> None:
        """限制动作范围"""
        if self.crashed:
            self.action["steering"] = 0
            self.action["acceleration"] = 0
        # 仅进行裁剪，不再缩放
        self.action["steering"] = np.clip(self.action["steering"], -self.MAX_STEERING, self.MAX_STEERING)
        self.action["acceleration"] = np.clip(self.action["acceleration"], -self.MAX_ACCELERATION, self.MAX_ACCELERATION)

    def get_state(self) -> Dict[str, float]:
        """返回车辆当前状态"""
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
        """返回静态车辆当前状态"""
        return {
            'position': self.position.tolist(),
            'heading': self.heading,
            'length': self.LENGTH,
            'width': self.WIDTH
        }
