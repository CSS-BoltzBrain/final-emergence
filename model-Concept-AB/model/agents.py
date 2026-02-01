import numpy as np

class Agent:
    def __init__(self, id, pos, target_pos, cfg):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.radius = cfg["agent"]["radius"]
        self.target_pos = np.array(target_pos, dtype=float)
        self.max_speed = cfg["agent"]["max_speed"]
        self.desired_speed = cfg["agent"]["desired_speed"]
        self.stuck_timer = 0.0

