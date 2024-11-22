import numpy as np
from scipy.optimize import minimize
from .vehicle import Vehicle

class MPCController:
    def __init__(self, env, horizon=5, dt=0.1):
        self.env = env          # Save environment reference
        self.horizon = horizon
        self.dt = dt
        # Add cache to store the last optimization result
        self.last_state = None
        self.last_action = None
        self.last_optimized_action = None
        
        # Pre-calculate boundary conditions
        self.bounds = [(-Vehicle.MAX_STEERING_ANGLE, Vehicle.MAX_STEERING_ANGLE),
                      (-Vehicle.MAX_ACCELERATION, Vehicle.MAX_ACCELERATION)] * horizon

    def optimize(self, state_dict, action):
        # Add state check; if the state and action do not change much, return the last result directly
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
            
        # Convert dictionary format state to array format
        state = current_state

        # Use a simpler initial guess
        u0 = np.tile(action, (self.horizon, 1)).flatten()

        # Use more relaxed convergence conditions during optimization
        res = minimize(self._objective_function, 
                      u0, 
                      args=(state,), 
                      bounds=self.bounds,  # Pre-defined boundaries
                      method='SLSQP',
                      options={'maxiter': 50,  # Limit maximum iterations
                              'ftol': 1e-3})   # Use more relaxed convergence tolerance

        # Cache results
        self.last_state = current_state
        self.last_action = action
        self.last_optimized_action = res.x[:2]
        
        return self.last_optimized_action

    def _objective_function(self, u, state):
        """
        Define the objective function for MPC.
        Args:
            u: Action sequence, shape: (2*horizon,)
            state: Current state

        Returns:
            Objective function value
        """
        cost = 0
        x = state.copy()
        for i in range(self.horizon):
            steering, acceleration = u[2*i], u[2*i+1]
            # State update (requires vehicle kinematic model)
            x = self._vehicle_model(x, [steering, acceleration])
            # Calculate deviation from the target
            cost += self._stage_cost(x)
        return cost

    def _vehicle_model(self, x, u):
        """
        Vehicle kinematic model for predicting future states.
        Args:
            x: Current state [x, y, v, heading, steering_angle]
            u: Current action [steering, acceleration]

        Returns:
            Next state
        """
        x_next = x.copy()
        dt = self.dt
        
        # Update position
        x_next[0] += x[2] * np.cos(x[3]) * dt  # x position
        x_next[1] += x[2] * np.sin(x[3]) * dt  # y position
        
        # Update speed
        x_next[2] = np.clip(x[2] + u[1] * dt, Vehicle.MIN_SPEED, Vehicle.MAX_SPEED)
        
        # Update heading angle
        if np.cos(x[4]) != 0:  # Prevent division by zero
            angular_velocity = (x[2] / Vehicle.LENGTH) * np.tan(x[4])
            x_next[3] += angular_velocity * dt
        
        # Update steering angle
        steering_change = u[0] * Vehicle.MAX_STEERING_CHANGE * dt
        x_next[4] = np.clip(x[4] + steering_change, 
                           -Vehicle.MAX_STEERING_ANGLE, 
                           Vehicle.MAX_STEERING_ANGLE)
        
        return x_next

    def _stage_cost(self, x):
        """Simplified cost function"""
        goal_pos = np.array(self.env.parking_lot.goal_position)
        goal_heading = self.env.goal_heading
        
        # Use a simpler distance calculation
        position_error = ((x[0] - goal_pos[0])**2 + (x[1] - goal_pos[1])**2) ** 0.5
        heading_error = abs(x[3] - goal_heading)  # Simplified heading error
        
        # Reduce penalty terms
        return position_error + 0.1 * heading_error
