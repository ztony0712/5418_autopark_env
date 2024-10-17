import numpy as np
from typing import Dict, List

class Vehicle:
    LENGTH: float = 4.0
    MAX_STEERING_ANGLE: float = np.pi / 4  # 约30度，可以根据需要调整
    MAX_SPEED: float = 20.0
    MIN_SPEED: float = -10.0  # 允许倒车
    MAX_ACCELERATION: float = 1.0  # 新增：最大加速度

    def __init__(self, position: List[float], heading: float = 0, velocity: float = 0):
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading
        self.velocity = velocity
        
        self.action: Dict[str, float] = {
            "steering": 0,
            "acceleration": 0
        }
        
        self.crashed = False
        self.steering_angle = 0

    def act(self, action: Dict[str, float]) -> None:
        """存储一个将要执行的动作"""
        if action:
            self.action.update(action)

    def step(self, dt: float) -> None:
        """更新车辆状态"""
        self.clip_actions()
        
        # 更新转向角度，并进行限制
        self.steering_angle = np.clip(
            self.action["steering"] * self.MAX_STEERING_ANGLE,
            -self.MAX_STEERING_ANGLE,
            self.MAX_STEERING_ANGLE
        )
        
        # 更新速度
        self.velocity += self.action["acceleration"] * dt
        self.velocity = np.clip(self.velocity, self.MIN_SPEED, self.MAX_SPEED)
        
        # 更新朝向
        angular_velocity = self.velocity * np.tan(self.steering_angle) / self.LENGTH
        self.heading += angular_velocity * dt
        
        # 更新位置
        v = self.velocity * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += v * dt
        
        # 确保朝向角度在 -π 到 π 之间
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

    def clip_actions(self) -> None:
        """限制动作的范围"""
        if self.crashed:
            self.action["steering"] = 0
            self.action["acceleration"] = 0
        self.action["steering"] = np.clip(self.action["steering"], -1, 1)
        self.action["acceleration"] = np.clip(self.action["acceleration"], -1, 1) * self.MAX_ACCELERATION

    def get_state(self) -> Dict[str, float]:
        """返回车辆的当前状态"""
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
        """返回静态车辆的当前状态"""
        return {
            'position': self.position.tolist(),
            'heading': self.heading,
            'length': self.LENGTH,
            'width': self.WIDTH
        }
