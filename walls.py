import numpy as np

class Wall:
    def __init__(self, cfg):
        self.type = cfg["type"]

        if self.type == "horizontal":
            self.y = cfg["y"]
            self.normal = np.array(cfg["normal"], dtype=float)

        elif self.type == "vertical":
            self.x = cfg["x"]
            self.normal = np.array(cfg["normal"], dtype=float)

        else:
            raise ValueError("Unknown wall type")

    def distance_and_normal(self, pos):
        if self.type == "horizontal":
            return abs(pos[1] - self.y), self.normal
        if self.type == "vertical":
            return abs(pos[0] - self.x), self.normal
