import numpy as np

class Wall:
    def __init__(self, cfg):
        self.type = cfg["type"]

        if self.type == "horizontal":
            self.y = cfg["y"]
            self.normal = np.array(cfg["normal"], dtype=float)
            self.x_range = cfg.get("x_range", [-float('inf'), float('inf')])

        elif self.type == "vertical":
            self.x = cfg["x"]
            self.normal = np.array(cfg["normal"], dtype=float)
            self.y_range = cfg.get("y_range", [-float('inf'), float('inf')])

        else:
            raise ValueError("Unknown wall type")

    def distance_and_normal(self, pos):
        if self.type == "horizontal":
            if self.x_range[0] <= pos[0] <= self.x_range[1]:
                return abs(pos[1] - self.y), self.normal
            d_left = np.sqrt((pos[0] - self.x_range[0])**2 + (pos[1] - self.y)**2)
            d_right = np.sqrt((pos[0] - self.x_range[1])**2 + (pos[1] - self.y)**2)
            
            if d_left < d_right:
                n = pos - np.array([self.x_range[0], self.y])
                dist = np.linalg.norm(n)
                return dist, n / (dist + 1e-8)
            else:
                n = pos - np.array([self.x_range[1], self.y])
                dist = np.linalg.norm(n)
                return dist, n / (dist + 1e-8)

        if self.type == "vertical":
            if self.y_range[0] <= pos[1] <= self.y_range[1]:
                return abs(pos[0] - self.x), self.normal
            d_bottom = np.sqrt((pos[0] - self.x)**2 + (pos[1] - self.y_range[0])**2)
            d_top = np.sqrt((pos[0] - self.x)**2 + (pos[1] - self.y_range[1])**2)
            
            if d_bottom < d_top:
                n = pos - np.array([self.x, self.y_range[0]])
                dist = np.linalg.norm(n)
                return dist, n / (dist + 1e-8)
            else:
                n = pos - np.array([self.x, self.y_range[1]])
                dist = np.linalg.norm(n)
                return dist, n / (dist + 1e-8)
