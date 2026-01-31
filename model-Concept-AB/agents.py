import numpy as np

class Agent:
    def __init__(self, id, pos, target_x, cfg):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.radius = cfg["agent"]["radius"]
        self.target_x = target_x
        self.max_speed = cfg["agent"]["max_speed"]
        self.desired_speed = cfg["agent"]["desired_speed"]
        
        # Determine direction based on target (Left or Right)

        direction_x = 1.0 if target_x > pos[0] else -1.0
        self.direction = np.array([direction_x, 0.0])

    def corridor_gradient(self):
        # Returns the normalized direction vector (i cap)
        return self.direction
