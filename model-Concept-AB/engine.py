import numpy as np
from agents import Agent
from update import update_all_agents
from walls import Wall

class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg["simulation"]["dt"]
        self.agents = []
        self.time = 0.0
        self.next_id = 0
        self.walls = [Wall(w) for w in cfg.get("walls", [])]
        self.spawn_rate = cfg["spawn"]["rate"]

        # Force spawn initial agents so you see them immediately
        print("Engine Initialized. Spawning starter agents...")
        self.force_spawn_safe("left")
        self.force_spawn_safe("right")

    def check_spawn_clearance(self, pos):
        # Default to 0.5m clearance if not specified
        min_dist = self.cfg["spawn"].get("min_dist", 0.5) 
        for agent in self.agents:
            if np.linalg.norm(agent.pos - pos) < min_dist:
                return False
        return True

    def force_spawn_safe(self, side):
        spawn_cfg = self.cfg["spawn"]
        # Try 10 random positions to find a clear spot
        for _ in range(10):
            if side == "left":
                x = spawn_cfg["left"]["x"]
                y = np.random.uniform(*spawn_cfg["left"]["y_range"])
                
                # --- BUG FIX WAS HERE ---
                # OLD: target_x = spawn_cfg["right"]["target_x"] (resulted in 0.5)
                # NEW: Use the target defined in the LEFT config (19.5)
                target_x = spawn_cfg["left"]["target_x"]
            else:
                x = spawn_cfg["right"]["x"]
                y = np.random.uniform(*spawn_cfg["right"]["y_range"])
                
                # NEW: Use the target defined in the RIGHT config (0.5)
                target_x = spawn_cfg["right"]["target_x"]

            pos = np.array([x, y])
            if self.check_spawn_clearance(pos):
                self._create_agent(pos, target_x)
                return

    def _create_agent(self, pos, target_x):
        agent = Agent(
            agent_id=self.next_id,
            pos=pos,
            target_x=target_x,
            cfg=self.cfg
        )
        self.agents.append(agent)
        self.next_id += 1

    def spawn_agent(self, side):
        self.force_spawn_safe(side)

    def remove_finished_agents(self):
        tol = self.cfg["simulation"]["target_tolerance"]
        # Only keep agents that are far from their target
        self.agents = [a for a in self.agents if abs(a.pos[0] - a.target_x) > tol]

    def step(self):
        # --- Spawn Logic ---
        expected_spawn = self.spawn_rate * self.dt
        n_spawn = int(expected_spawn)
        if np.random.rand() < (expected_spawn - n_spawn):
            n_spawn += 1
            
        for _ in range(n_spawn):
            if np.random.rand() < 0.5:
                self.spawn_agent("left")
            else:
                self.spawn_agent("right")

        # --- Update Physics ---
        update_all_agents(self.agents, self.walls, self.cfg, self.dt)

        # --- Cleanup ---
        self.remove_finished_agents()
        self.time += self.dt