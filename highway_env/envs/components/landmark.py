import numpy as np

class Landmark:
    def __init__(self, road, position, heading=0):
        self.road = road
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading

class Obstacle(Landmark):
    def __init__(self, road, position, heading=0, width=70, height=10):
        super().__init__(road, position, heading)
        self.width = width
        self.height = height

    def get_rect(self):
        return (int(self.position[0]), int(self.position[1]), self.width, self.height)
