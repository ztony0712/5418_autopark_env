import numpy as np

class LineType:
    CONTINUOUS = 0
    BROKEN = 1

class Lane:
    def __init__(self, width=4.0):
        self.width = width

class StraightLane(Lane):
    def __init__(self, start, end, width=4.0, line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS)):
        super().__init__(width)
        self.start = np.array(start)
        self.end = np.array(end)
        self.line_types = line_types

class CircularLane(Lane):
    def __init__(self, center, radius, start_angle, end_angle, clockwise=True, width=4.0):
        super().__init__(width)
        self.center = np.array(center)
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.clockwise = clockwise

    def position(self, angle):
        return self.center + self.radius * np.array([np.cos(angle), np.sin(angle)])
