import numpy as np

# Import fallback for package/standalone usage.
try:
    from .update import update_physics, check_exits
except ImportError:  # pragma: no cover
    from update import update_physics, check_exits

try:
    from .walls import Wall
except ImportError:  # pragma: no cover
    from walls import Wall


class Engine:
    def __init__(self, cfg, nav_field=None):
        self.cfg = cfg
        self.dt = cfg["simulation"]["dt"]
        self.time = 0.0

        # Fixed-capacity state buffers.
        self.CAPACITY = 30
        self.active = np.zeros(self.CAPACITY, dtype=bool)
        self.pos = np.zeros((self.CAPACITY, 2), dtype=np.float32)
        self.vel = np.zeros((self.CAPACITY, 2), dtype=np.float32)
        self.target = np.zeros((self.CAPACITY, 2), dtype=np.float32)
        self.ids = np.arange(self.CAPACITY, dtype=np.int32)

        self.walls = [Wall(w) for w in cfg.get("walls", [])]
        self.spawn_rate = cfg["spawn"]["rate"]

        self.max_agents = cfg["spawn"].get("max_agents", 1000)
        self.total_spawned = 0

        # Stored for optional external use.
        self.nav_field = nav_field

    @property
    def agents(self):
        """Compatibility view for visualization utilities."""
        active_indices = np.where(self.active)[0]
        return [AgentView(i, self) for i in active_indices]

    def step(self):
        self._spawn()
        if np.any(self.active):
            update_physics(self.pos, self.vel, self.target, self.active, self.walls, self.cfg, self.dt)
            exited = check_exits(self.pos, self.target, self.active)
            if len(exited) > 0:
                self.active[exited] = False
        self.time += self.dt

    def _spawn(self):
        if self.total_spawned >= self.max_agents:
            return

        expected = int(self.spawn_rate * self.time)
        to_spawn = expected - self.total_spawned
        if to_spawn <= 0:
            return

        free_slots = np.where(~self.active)[0]
        count = min(to_spawn, len(free_slots))
        indices = free_slots[:count]

        for idx in indices:
            side = "left" if np.random.rand() < 0.5 else "right"
            scfg = self.cfg["spawn"][side]
            y_range = scfg["y_range"]

            self.pos[idx] = [scfg["x"], np.random.uniform(y_range[0], y_range[1])]
            self.vel[idx] = [0.0, 0.0]
            self.target[idx] = scfg["target"]
            self.active[idx] = True
            self.total_spawned += 1


class AgentView:
    """
    Mimics an Agent object for backward compatibility.
    """
    def __init__(self, idx, engine):
        self.idx = idx
        self.engine = engine
        self.id = engine.ids[idx]

    @property
    def pos(self):
        return self.engine.pos[self.idx].copy()

    @property
    def vel(self):
        return self.engine.vel[self.idx].copy()

    @property
    def radius(self):
        return self.engine.cfg["agent"]["radius"]

    @property
    def target_pos(self):
        return self.engine.target[self.idx].copy()

    @property
    def max_speed(self):
        return self.engine.cfg["agent"]["max_speed"]
