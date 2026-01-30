import numpy as np
import matplotlib.pyplot as plt
from agents import Agent
from update import update_all_agents

# --- CONFIGURATION TEMPLATE ---
# We will modify 'desired_speed' and 'max_speed' dynamically in the loop
BASE_CONFIG = {
    "simulation": {
        "dt": 0.02,
        "target_tolerance": 0.3
    },
    "domain": {"xmin": 0.0, "xmax": 20.0, "ymin": 0.0, "ymax": 4.0},
    "agent": {
        "radius": 0.25,
        "perception_radius": 2.5,
        "crowd_decay": 1.1  
    },
    "forces": {
        "gradient":   {"strength": 30.0}, 
        "repulsive":  {"strength": 30.0, "decay": 9.0},
        "resistance": {"gamma": 1.8}, 
        "contact":    {"k": 0.0, "kappa": 0.0},  # Preserved your settings (No Contact Force)
        "random":     {"probability": 0.1, "strength": 1.0}
    },
    "spawn": {
        "rate": 6.0,          # High injection rate
        "left":  {"x": 0.5,   "y_range": [0.5, 3.5], "target_x": 19.5},
        "right": {"x": 19.5,  "y_range": [0.5, 3.5], "target_x": 0.5}
    }
}

WIDTH, HEIGHT = 20.0, 4.0

# --- HELPER FUNCTIONS ---
def custom_gradient(self): 
    return getattr(self, 'desired_direction', np.array([1,0]))
Agent.corridor_gradient = custom_gradient

def apply_navigation(agents):
    for ag in agents:
        target = np.array([ag.target_x, ag.pos[1]])
        direction = target - ag.pos
        dist = np.linalg.norm(direction)
        ag.desired_direction = direction / dist if dist > 0.01 else np.zeros(2)

def enforce_boundaries(agents):
    for ag in agents:
        # Floor/Ceiling Clamp
        if ag.pos[1] < ag.radius:
            ag.pos[1] = ag.radius; ag.vel[1] = 0.0
        elif ag.pos[1] > HEIGHT - ag.radius:
            ag.pos[1] = HEIGHT - ag.radius; ag.vel[1] = 0.0
        # Walls
        ag.pos[0] = max(ag.radius, min(WIDTH - ag.radius, ag.pos[0]))

def run_experiment(speed):
    """Runs the simulation for a specific desired speed."""
    
    # 1. Setup Configuration for this run
    cfg = BASE_CONFIG.copy()
    cfg["agent"]["desired_speed"] = speed
    # Ensure max_speed is slightly higher to allow the desired speed
    cfg["agent"]["max_speed"] = speed * 1.5 
    
    # 2. Reset State
    agents = []
    agent_counter = 0
    sim_time = 0.0
    last_spawn = 0.0
    spawn_times = {} # ID -> Spawn Time
    exit_durations = []
    
    # Stop condition: Collect data for 150 agents or Timeout
    MAX_SAMPLES = 150
    TIMEOUT = 90.0 # seconds of sim time
    
    print(f"Testing Speed: {speed} m/s ... ", end="", flush=True)
    
    while len(exit_durations) < MAX_SAMPLES and sim_time < TIMEOUT:
        dt = cfg["simulation"]["dt"]
        sim_time += dt
        
        # --- SPAWN ---
        if sim_time - last_spawn > 1.0 / cfg["spawn"]["rate"]:
            # Left Spawn
            y_l = np.random.uniform(*cfg["spawn"]["left"]["y_range"])
            agents.append(Agent(agent_counter, [0.5, y_l], cfg["spawn"]["left"]["target_x"], cfg))
            spawn_times[agent_counter] = sim_time
            agent_counter += 1
            
            # Right Spawn
            y_r = np.random.uniform(*cfg["spawn"]["right"]["y_range"])
            agents.append(Agent(agent_counter, [19.5, y_r], cfg["spawn"]["right"]["target_x"], cfg))
            spawn_times[agent_counter] = sim_time
            agent_counter += 1
            
            last_spawn = sim_time
            
        # --- PHYSICS ---
        apply_navigation(agents)
        update_all_agents(agents, [], cfg, dt)
        enforce_boundaries(agents)
        
        # --- DESPAWN & MEASURE ---
        active_agents = []
        for ag in agents:
            dist_to_target = abs(ag.pos[0] - ag.target_x)
            
            # Use same tolerance as your visualization
            if dist_to_target < 0.5:
                # Agent Reached Target
                duration = sim_time - spawn_times[ag.id]
                exit_durations.append(duration)
            else:
                active_agents.append(ag)
        agents = active_agents
        
    avg_time = np.mean(exit_durations) if exit_durations else 0
    print(f"Avg Exit Time: {avg_time:.2f}s (Samples: {len(exit_durations)})")
    return avg_time

# --- MAIN EXECUTION ---
speeds = [1.0, 2.0, 3.0, 4.0, 5.0]
results = []

print("=== Starting 'Faster is Slower' Sweep ===")
print(f"Spawn Rate: {BASE_CONFIG['spawn']['rate']} ag/s | Contact Force: {BASE_CONFIG['forces']['contact']['k']}")
print("-" * 50)

for s in speeds:
    avg_t = run_experiment(s)
    results.append(avg_t)

# --- PLOTTING ---
plt.figure(figsize=(10, 6))
plt.plot(speeds, results, marker='o', linewidth=2, color='#d32f2f', label='Measured Time')

# Theoretical curve (Distance / Speed)
# Distance is roughly 19.0 meters (start x=0.5 to target x=19.5)
theoretical = [19.0 / s for s in speeds]
plt.plot(speeds, theoretical, '--', color='gray', label='Theoretical (No Traffic)')

plt.title("Crowd Efficiency vs Desired Speed")
plt.xlabel("Desired Speed (m/s)")
plt.ylabel("Average Time to Cross Corridor (s)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()