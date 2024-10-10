import numpy as np

class RoadNetwork:
    def __init__(self):
        self.lanes = []

    def add_lane(self, start, end, lane):
        self.lanes.append((start, end, lane))

    def lanes_list(self):
        return self.lanes

class Road:
    def __init__(self, network):
        self.network = network
        self.vehicles = []
        self.objects = []
