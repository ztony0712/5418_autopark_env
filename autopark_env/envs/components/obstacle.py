import numpy as np

class Obstacle:
    def __init__(self, position, size, heading=0):
        self.position = np.array(position, dtype=np.float32)
        self.heading = heading
        self.width, self.height = size

    def get_rect(self):
        # Return top-left corner coordinates and size
        left = int(self.position[0] - self.width / 2)
        top = int(self.position[1] - self.height / 2)
        return (left, top, self.width, self.height)
