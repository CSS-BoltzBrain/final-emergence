import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from agents import Agent
from update import update_all_agents

# --- 1. "GUARANTEED JAM" CONFIGURATION ---
CONFIG = {
    "simulation": {
        "dt": 0.03,
        "target_tolerance": 0.5
    },
    # [FIX 1] Narrow Domain (3.0m high instead of 6.0m) forces interaction
    "domain": {"xmin": 0.0, "xmax": 12.0, "ymin": 0.0, "ymax": 3.0},
    "agent": {
        "radius": 0.25,
        "desired_speed": 1.4,
        "max_speed": 2.2,
        "perception_radius": 2.5,
        "crowd_decay": 1.0
    },
    "forces": {
        "gradient":   {"strength": 25.0}, 
        
        # Stronger repulsion to create "pressure" in the crowd
        "repulsive":  {"strength": 20.0, "decay": 0.5, "zone": 0.7, "anisotropy": 0.5},
        
        "wall_strength": 500.0,
        "resistance": {"gamma": 0.5},
        
        # [FIX 2] High Friction (Kappa) + Moderate Stiffness
        # Kappa=500 makes them "sticky" when they slide past each other
        "contact":    {"k": 500.0, "kappa": 500.0}, 
        
        "random":     {"probability": 0.2, "strength": 8.0} # More noise to break lanes
    },
    "spawn": {
        # High injection rate relative to the narrow 3m width
        "rate": 6.0, 
        "left":  {"x": 0.5,  "y_range": [0.5, 2.5], "target_x": 11.5},
        "right": {"x": 11.5, "y_range": [0.5, 2.5], "target_x": 0.5}
    }
}

WIDTH, HEIGHT = 12.0, 3.0 # Must match domain config

# --- 2. SETUP ---
class Wall:
    def __init__(self, p1, p2, normal):
        self.p1, self.p2, self.normal = np.array(p1), np.array(p2), np.array(normal)
        self.v = self.p2 - self.p1
        self.len_sq = np.dot(self.v, self.v)
    def distance_and_normal(self, pos):
        op = pos - self.p1
        t = max(0, min(1, np.dot(op, self.v) / self.len_sq))
        closest = self.p1 + t * self.v
        return np.linalg.norm(pos - closest), self.normal

walls = [
    Wall([0, 0], [WIDTH, 0], [0, 1]),         # Floor
    Wall([0, HEIGHT], [WIDTH, HEIGHT], [0, -1]) # Ceiling
]

# --- 3. LOGIC ---
def apply_navigation(agents):
    for ag in agents:
        target = np.array([ag.target_x, ag.pos[1]])
        direction = target - ag.pos
        dist = np.linalg.norm(direction)
        # Add slight wobble to desired direction to prevent "perfect" lines
        ag.desired_direction = direction / dist if dist > 0.01 else np.zeros(2)

def custom_gradient(self): return getattr(self, 'desired_direction', np.array([1,0]))
Agent.corridor_gradient = custom_gradient

def apply_physics_step(agents, walls, cfg, dt):
    update_all_agents(agents, [], cfg, dt)
    wall_str = cfg["forces"]["wall_strength"]
    
    for ag in agents:
        total_wall = np.zeros(2)
        for w in walls:
            d, n = w.distance_and_normal(ag.pos)
            if d < ag.radius + 0.5:
                f = wall_str * np.exp((ag.radius - d)/0.1)
                total_wall += n * f
                if d < ag.radius: ag.pos += n * (ag.radius - d) * 0.5
        
        ag.vel += total_wall * dt
        # Hard Clamp
        ag.pos[1] = max(ag.radius, min(HEIGHT - ag.radius, ag.pos[1]))
        ag.pos[0] = max(ag.radius, min(WIDTH - ag.radius, ag.pos[0]))

# --- 4. EXPERIMENT ENGINE ---
def run_experiment(max_agents=600):
    agents = []
    agent_id = 0
    spawned = 0
    t = 0.0
    last_spawn = 0.0
    dt = CONFIG["simulation"]["dt"]
    rate = CONFIG["spawn"]["rate"]
    
    active_jams = {} 
    jam_durations = []
    
    print(f"Running NARROW Corridor Experiment (N={max_agents})... ", end="")
    
    while True:
        if spawned % 50 == 0: print(".", end="", flush=True)
        
        # 1. Spawn
        if spawned < max_agents and (t - last_spawn >= 1.0/rate):
            # Left
            ag = Agent(agent_id, [0,0], 0, CONFIG)
            # Randomize speed slightly to break symmetry
            ag.desired_speed = np.random.normal(1.4, 0.2) 
            ag.pos = np.array([0.5, np.random.uniform(0.5, HEIGHT-0.5)])
            ag.target_x = 11.5
            agents.append(ag); agent_id += 1; spawned += 1
            
            # Right
            if spawned < max_agents:
                ag = Agent(agent_id, [0,0], 0, CONFIG)
                ag.desired_speed = np.random.normal(1.4, 0.2)
                ag.pos = np.array([11.5, np.random.uniform(0.5, HEIGHT-0.5)])
                ag.target_x = 0.5
                agents.append(ag); agent_id += 1; spawned += 1
            last_spawn = t
        
        if spawned >= max_agents and len(agents) == 0: break
            
        # 2. Physics
        apply_navigation(agents)
        apply_physics_step(agents, walls, CONFIG, dt)
        t += dt
        
        # 3. Detection
        # Strict jam definition: Speed < 0.15 inside the conflict zone
        mid_min, mid_max = 3.0, 9.0
        current_stuck = set()
        
        for ag in agents:
            if mid_min < ag.pos[0] < mid_max:
                if np.linalg.norm(ag.vel) < 0.15: # Almost dead stop
                    current_stuck.add(ag.id)
                    active_jams[ag.id] = active_jams.get(ag.id, 0.0) + dt
        
        for pid in list(active_jams.keys()):
            if pid not in current_stuck:
                dur = active_jams.pop(pid)
                if dur > 0.5: jam_durations.append(dur)

        agents = [ag for ag in agents if abs(ag.pos[0] - ag.target_x) > 0.5]
        if t > 300: break # Timeout

    print(f"\nExperiment Complete. Jams Captured: {len(jam_durations)}")
    return jam_durations

# --- 5. VISUALIZATION ---
if __name__ == "__main__":
    data = run_experiment(max_agents=600)
    
    if len(data) > 20:
        data = np.array(data)
        data.sort()
        y = np.arange(len(data), 0, -1) / len(data)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.loglog(data, y, 'bo', markersize=3, alpha=0.3, label="Stop Events")
        
        # Tail Fit (Looking for Heavy Tail)
        cutoff = 0.6
        mask = data >= cutoff
        if np.sum(mask) > 10:
            x_tail, y_tail = data[mask], y[mask]
            log_x, log_y = np.log10(x_tail), np.log10(y_tail)
            slope, intercept, r, _, _ = stats.linregress(log_x, log_y)
            
            plt.plot(x_tail, (10**intercept)*(x_tail**slope), 'r--', lw=2, 
                     label=f"Power Law Fit ($\\alpha={-slope:.2f}, R^2={r**2:.2f}$)")
            
            plt.title(f"Crowd Turbulence (Bi-Directional)\nPower Law Exponent $\\alpha \\approx {-slope:.2f}$")
        else:
            plt.title("Crowd Turbulence (Insufficient Tail for Power Law)")
            
        plt.xlabel("Duration Stuck (s)")
        plt.ylabel("P(X $\ge$ x)")
        plt.grid(True, which="both", alpha=0.2)
        plt.legend()
        plt.savefig("narrow_corridor_powerlaw.png")
        print("Saved plot to narrow_corridor_powerlaw.png")
        plt.show()
    else:
        print("Still low jam count. Try increasing 'spawn rate' further.")