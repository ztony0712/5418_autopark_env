import numpy as np
from scipy.optimize import minimize
from .vehicle import Vehicle

class MPCController:
    def __init__(self, env, horizon=5, dt=0.1):
        self.env = env          # 保存环境引用
        self.horizon = horizon
        self.dt = dt
        # 添加缓存来存储上一次的优化结果
        self.last_state = None
        self.last_action = None
        self.last_optimized_action = None
        
        # 预先计算边界条件
        self.bounds = [(-Vehicle.MAX_STEERING_ANGLE, Vehicle.MAX_STEERING_ANGLE),
                      (-Vehicle.MAX_ACCELERATION, Vehicle.MAX_ACCELERATION)] * horizon

    def optimize(self, state_dict, action):
        # 添加状态检查，如果状态和动作变化不大，直接返回上次的结果
        current_state = np.array([
            state_dict['position'][0],
            state_dict['position'][1],
            state_dict['velocity'],
            state_dict['heading'],
            state_dict['steering_angle']
        ])
        
        if (self.last_state is not None and 
            self.last_action is not None and 
            np.allclose(current_state, self.last_state, rtol=1e-2) and 
            np.allclose(action, self.last_action, rtol=1e-2)):
            return self.last_optimized_action
            
        # 将字典格式的状态转换为数组格式
        state = current_state

        # 使用更简单的初始猜测
        u0 = np.tile(action, (self.horizon, 1)).flatten()

        # 优化时使用更宽松的收敛条件
        res = minimize(self._objective_function, 
                      u0, 
                      args=(state,), 
                      bounds=self.bounds,  # 预先定义边界
                      method='SLSQP',
                      options={'maxiter': 50,  # 限制最大迭代次数
                              'ftol': 1e-3})   # 使用更宽松的收敛容差

        # 缓存结果
        self.last_state = current_state
        self.last_action = action
        self.last_optimized_action = res.x[:2]
        
        return self.last_optimized_action

    def _objective_function(self, u, state):
        """
        定义MPC的目标函数。
        Args:
            u: 动作序列，shape: (2*horizon,)
            state: 当前状态

        Returns:
            目标函数值
        """
        cost = 0
        x = state.copy()
        for i in range(self.horizon):
            steering, acceleration = u[2*i], u[2*i+1]
            # 状态更新（这里需要车辆的运动学模型）
            x = self._vehicle_model(x, [steering, acceleration])
            # 计算与目标的偏差
            cost += self._stage_cost(x)
        return cost

    def _vehicle_model(self, x, u):
        """
        车辆运动学模型，用于预测未来状态。
        Args:
            x: 当前状态 [x, y, v, heading, steering_angle]
            u: 当前动作 [steering, acceleration]

        Returns:
            下一时刻状态
        """
        x_next = x.copy()
        dt = self.dt
        
        # 更新位置
        x_next[0] += x[2] * np.cos(x[3]) * dt  # x position
        x_next[1] += x[2] * np.sin(x[3]) * dt  # y position
        
        # 更新速度
        x_next[2] = np.clip(x[2] + u[1] * dt, Vehicle.MIN_SPEED, Vehicle.MAX_SPEED)
        
        # 更新航向角
        if np.cos(x[4]) != 0:  # 防止除零
            angular_velocity = (x[2] / Vehicle.LENGTH) * np.tan(x[4])
            x_next[3] += angular_velocity * dt
        
        # 更新转向角
        steering_change = u[0] * Vehicle.MAX_STEERING_CHANGE * dt
        x_next[4] = np.clip(x[4] + steering_change, 
                           -Vehicle.MAX_STEERING_ANGLE, 
                           Vehicle.MAX_STEERING_ANGLE)
        
        return x_next

    def _stage_cost(self, x):
        """简化的成本函数"""
        goal_pos = np.array(self.env.parking_lot.goal_position)
        goal_heading = self.env.goal_heading
        
        # 使用更简单的距离计算
        position_error = ((x[0] - goal_pos[0])**2 + (x[1] - goal_pos[1])**2) ** 0.5
        heading_error = abs(x[3] - goal_heading)  # 简化的航向误差
        
        # 减少惩罚项
        return position_error + 0.1 * heading_error
