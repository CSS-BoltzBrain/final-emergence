import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from agents import Agent
from update import update_all_agents

# --- CONFIGURATION ---
CONFIG = {
    "simulation": {
        "dt": 0.02,
        "steps_per_frame": 10,  # SPEED RUN: Calculate 10 physics steps for every 1 drawn frame
        "target_tolerance": 0.3
    },
    "domain": {"xmin": 0.0, "xmax": 20.0, "ymin": 0.0, "ymax": 4.0},
    "agent": {
        "radius": 0.25,
        "desired_speed": 1.3,   # Requested
        "max_speed": 1.5,       # Requested
        "perception_radius": 2.5,
        "crowd_decay": 1.1  
    },
    "forces": {
        "gradient":   {"strength": 30.0}, 
        "repulsive":  {"strength": 30.0, "decay": 9.0},
        "resistance": {"gamma": 1.8}, 
        "contact":    {"k": 0.0, "kappa": 0.0}, # Keeping your k=0 setting
        "random":     {"probability": 0.1, "strength": 1.0}
    },
    "spawn": {
        "rate": 3.9,            # "Edge of jams" rate
        "max_agents": 600,      # Increased buffer for high density
        "left":  {"x": 0.5,   "y_range": [0.5, 3.5], "target_x": 19.5},
        "right": {"x": 19.5,  "y_range": [0.5, 3.5], "target_x": 0.5}
    }
}

# --- SETUP ---
WIDTH, HEIGHT = 20.0, 4.0
walls = [] 
agents = []
agent_counter = 0
spawned_total = 0
sim_time = 0.0
last_spawn = 0.0

# Visualization Setup
fig, ax = plt.subplots(figsize=(15, 3))
ax.set_xlim(0, WIDTH); ax.set_ylim(0, HEIGHT); ax.set_aspect('equal')
ax.axhline(0, color='k', lw=3); ax.axhline(HEIGHT, color='k', lw=3)
ax.set_xticks([]); ax.set_yticks([]) # Clean look

# Agents drawn as circles
scat = ax.scatter([], [], s=100, edgecolors='black', linewidth=0.5, alpha=0.9)
time_text = ax.text(0.01, 0.9, '', transform=ax.transAxes, fontsize=10, fontweight='bold')
# Throughput text
flow_text = ax.text(0.85, 0.9, '', transform=ax.transAxes, fontsize=10, color='darkgreen', fontweight='bold')

# Track exits for throughput calculation
exits_window = [] 

def apply_navigation(agents):
    for ag in agents:
        target = np.array([ag.target_x, ag.pos[1]])
        direction = target - ag.pos
        dist = np.linalg.norm(direction)
        ag.desired_direction = direction / dist if dist > 0.01 else np.zeros(2)

def custom_gradient(self): return getattr(self, 'desired_direction', np.array([1,0]))
Agent.corridor_gradient = custom_gradient

def update(frame):
    global agents, agent_counter, spawned_total, last_spawn, sim_time, exits_window
    dt = CONFIG["simulation"]["dt"]
    steps = CONFIG["simulation"]["steps_per_frame"]
    max_agents = CONFIG["spawn"]["max_agents"]
    spawn_rate = CONFIG["spawn"]["rate"]
    
    # Physics Loop (High speed steps)
    for _ in range(steps):
        sim_time += dt
        
        # --- Spawning ---
        if spawned_total < max_agents:
            if sim_time - last_spawn > 1.0 / spawn_rate:
                # Spawn Left
                y_l = np.random.uniform(*CONFIG["spawn"]["left"]["y_range"])
                agents.append(Agent(agent_counter, [0.5, y_l], CONFIG["spawn"]["left"]["target_x"], CONFIG))
                agent_counter += 1; spawned_total += 1
                
                # Spawn Right
                if spawned_total < max_agents:
                    y_r = np.random.uniform(*CONFIG["spawn"]["right"]["y_range"])
                    agents.append(Agent(agent_counter, [19.5, y_r], CONFIG["spawn"]["right"]["target_x"], CONFIG))
                    agent_counter += 1; spawned_total += 1
                
                last_spawn = sim_time
        
        # --- Updates ---
        apply_navigation(agents)
        update_all_agents(agents, walls, CONFIG, dt)
        
        # Wall Clamps
        for ag in agents:
            if ag.pos[1] < ag.radius: ag.pos[1] = ag.radius; ag.vel[1] = 0
            elif ag.pos[1] > HEIGHT - ag.radius: ag.pos[1] = HEIGHT - ag.radius; ag.vel[1] = 0
            ag.pos[0] = max(ag.radius, min(WIDTH - ag.radius, ag.pos[0]))
        
        # --- Despawn & Flow Measure ---
        active = []
        for ag in agents:
            if abs(ag.pos[0] - ag.target_x) < 0.5:
                # Agent exited: Record time
                exits_window.append(sim_time)
            else:
                active.append(ag)
        agents = active

    # Filter exits to last 3 seconds for instantaneous flow rate
    exits_window = [t for t in exits_window if t > sim_time - 3.0]
    flow_rate = len(exits_window) / 3.0 if sim_time > 3.0 else 0

    # --- Drawing ---
    if agents:
        pos = np.array([ag.pos for ag in agents])
        colors = ['#d32f2f' if ag.target_x > 10 else '#1976d2' for ag in agents]
        scat.set_offsets(pos)
        scat.set_facecolors(colors)
        
        # Dynamic Status
        status = "FREE FLOW"
        if len(agents) > 100 and flow_rate < 2.0: status = "CONGESTED"
        if len(agents) > 150 and flow_rate < 1.0: status = "JAMMED"
        
        time_text.set_text(f"Time: {sim_time:.1f}s | Count: {len(agents)} | Status: {status}")
        flow_text.set_text(f"Throughput: {flow_rate:.1f} ag/s")
    else:
        scat.set_offsets(np.empty((0, 2)))
        
    return scat, time_text, flow_text

# --- RENDER CONFIG ---
# We want to see roughly 60 seconds of simulation.
# Each frame covers 10 steps * 0.02s = 0.2s
# Frames needed = 60 / 0.2 = 300 frames
frames = 300

print(f"Simulating & Recording {frames} frames (Speed run)...")
ani = FuncAnimation(fig, update, frames=frames, blit=True)

# Save at 30 FPS. 
# 300 frames / 30 fps = 10 second GIF duration.
# Realtime factor: 60s sim / 10s video = 6x speed.
ani.save("jam_speedrun.gif", writer=PillowWriter(fps=30))
print("Saved 'jam_speedrun.gif'")
plt.close()