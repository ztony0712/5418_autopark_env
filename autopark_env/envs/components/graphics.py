import pygame
import numpy as np

class VehicleGraphics:
    @staticmethod
    def display(vehicle, screen, color=(255, 255, 0)):
        """
        Display the vehicle on the screen
        """
        vehicle_surface = pygame.Surface((40, 20))  # Create a surface representing the vehicle, size 40x20
        vehicle_surface.set_colorkey((0, 0, 0))  # Set transparent color
        vehicle_surface.fill(color)  # Fill with specified color
        rotated_vehicle = pygame.transform.rotate(vehicle_surface, -np.degrees(vehicle.heading))  # Rotate vehicle surface to face current heading
        vehicle_rect = rotated_vehicle.get_rect(center=(int(vehicle.position[0]), int(vehicle.position[1])))  # Get position rectangle after rotation
        screen.blit(rotated_vehicle, vehicle_rect.topleft)  # Draw the rotated vehicle

    @staticmethod
    def draw_parking_lanes(screen, lane_width, lane_height, num_rows, num_cols, screen_width, screen_height):
        """
        Draw white lines for parking spaces, ensuring they are centered and displayed vertically
        """
        total_width = num_cols * lane_width  # Total width
        total_height = num_rows * lane_height  # Total height
        
        # Calculate starting position for centering
        start_x = (screen_width - total_width) // 2
        start_y = (screen_height - total_height) // 2

        # Draw vertical parking space lines
        for row in range(num_rows):
            for col in range(num_cols):
                x = start_x + col * lane_width  # x-coordinate for each column
                y_start = start_y + row * lane_height  # y-coordinate for each row
                y_end = y_start + lane_height // 2  # Shorten line length
                pygame.draw.line(screen, (255, 255, 255), (x, y_start), (x, y_end), 2)  # Draw vertical line

    @staticmethod
    def draw_goal(screen, goal_x, goal_y, goal_width=15, goal_height=15):
        """
        Draw the target parking spot
        """
        goal_rect = pygame.Rect(goal_x, goal_y, goal_width, goal_height)
        pygame.draw.rect(screen, (0, 255, 0), goal_rect)  # Green parking target

    @staticmethod
    def draw_walls(screen, walls):
        """
        Draw walls
        """
        for wall in walls:
            wall_rect = pygame.Rect(*wall.get_rect())  # Use the get_rect method of Obstacle
            pygame.draw.rect(screen, (255, 0, 0), wall_rect)  # Red walls
