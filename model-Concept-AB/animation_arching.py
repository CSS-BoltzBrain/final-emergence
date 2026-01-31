import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ============================================================
# 1. CONFIGURATION (Stop-and-Go Waves)
# ============================================================
CONFIG = {
    "dt": 0.02,
    "domain": {"width": 20.0, "height": 4.0},
    "forces": {
        "gradient":   {"strength": 30.0}, 
        "repulsive":  {"strength": 30.0, "decay": 9.0},
        "resistance": {"gamma": 1.8}, 
    },
    "agent": {
        "crowd_decay": 1.1, 
        "radius": 0.25
    },
    "barrier": {
        "x": 10.0, "thickness": 0.4, "gap_width": 1.2, "y_center": 2.0
    },
    # Ensure enough agents exist at t=90s
    "spawn_rate": 3.0,     
    "max_agents": 100     
}

# Derived Geometry
gap_half = CONFIG["barrier"]["gap_width"] / 2.0
wall_y_bottom = CONFIG["barrier"]["y_center"] - gap_half
wall_y_top    = CONFIG["barrier"]["y_center"] + gap_half
b_x_min = CONFIG["barrier"]["x"] - CONFIG["barrier"]["thickness"]/2
b_x_max = CONFIG["barrier"]["x"] + CONFIG["barrier"]["thickness"]/2

# ============================================================
# 2. PHYSICS ENGINE (Optimized)
# ============================================================
class Agent:
    def __init__(self, uid, pos, target_x):
        self.id = uid
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.target_x = target_x
        self.jitter = np.random.uniform(-0.2, 0.2) 

    def get_direction(self):
        # Nav Logic
        bx = CONFIG["barrier"]["x"]
        to_right = self.target_x > 10.0
        x, y = self.pos
        must_cross = (to_right and x < bx) or (not to_right and x > bx)
        
        if must_cross:
            target = np.array([bx, CONFIG["barrier"]["y_center"] + self.jitter])
        else:
            target = np.array([self.target_x, y])
            
        d = target - self.pos
        dist = np.linalg.norm(d)
        return d / dist if dist > 1e-8 else np.zeros(2)

def update_physics(agents):
    # Neighbors
    pos_arr = np.array([a.pos for a in agents])
    
    active = []
    
    for i, ag in enumerate(agents):
        # 1. Force Calculation
        # Simple dist check
        d_vec = ag.pos - pos_arr
        d_sq = np.sum(d_vec**2, axis=1)
        # Filter neighbors < 2.0m
        mask = (d_sq < 4.0) & (d_sq > 0)
        neighbors_vec = d_vec[mask]
        neighbors_dsq = d_sq[mask]
        
        # Drive
        m = len(neighbors_vec)
        decay = CONFIG["agent"]["crowd_decay"] ** -m
        F = CONFIG["forces"]["gradient"]["strength"] * ag.get_direction() * decay
        
        # Repulsion
        A, B = CONFIG["forces"]["repulsive"]["strength"], CONFIG["forces"]["repulsive"]["decay"]
        if m > 0:
            rep = A * np.exp(-B * neighbors_dsq)[:,None] * 2.0 * neighbors_vec
            F += np.sum(rep, axis=0)
            
        # Walls
        x, y = ag.pos
        if y < 1.0: F[1] += A * np.exp(-B*y**2) * 2.0 * y
        dy = y - 4.0
        if dy > -1.0: F[1] += A * np.exp(-B*dy**2) * 2.0 * dy
        
        # Barrier
        dx = x - CONFIG["barrier"]["x"]
        if abs(dx) < 1.0:
            if y < wall_y_bottom or y > wall_y_top:
                F[0] += np.sign(dx) * A * np.exp(-B*dx**2) * 2.0

        # Drag
        F -= CONFIG["forces"]["resistance"]["gamma"] * ag.vel
        
        # Cap
        if np.linalg.norm(F) > 50.0: F = 50.0 * F / np.linalg.norm(F)
        
        # Integration
        ag.vel += F * CONFIG["dt"]
        s = np.linalg.norm(ag.vel)
        if s > 1.5: ag.vel *= (1.5/s)
        
        old_x = ag.pos[0]
        ag.pos += ag.vel * CONFIG["dt"]
        
        # Collisions (Sliding)
        r = 0.25
        # Domain Y
        if ag.pos[1] < r: 
            ag.pos[1] = r; 
            if ag.vel[1] < 0: ag.vel[1] = 0
        elif ag.pos[1] > 4.0-r: 
            ag.pos[1] = 4.0-r; 
            if ag.vel[1] > 0: ag.vel[1] = 0
        
        # Barrier X
        if b_x_min < ag.pos[0] < b_x_max:
            if ag.pos[1] < wall_y_bottom or ag.pos[1] > wall_y_top:
                if old_x <= b_x_min: 
                    ag.pos[0] = b_x_min; 
                    if ag.vel[0] > 0: ag.vel[0] = 0
                elif old_x >= b_x_max: 
                    ag.pos[0] = b_x_max; 
                    if ag.vel[0] < 0: ag.vel[0] = 0
        
        # Exit Check
        if abs(ag.pos[0] - ag.target_x) > 0.5:
            active.append(ag)
            
    # Solve Overlaps
    active.sort(key=lambda a: a.pos[0])
    pos = np.array([a.pos for a in active])
    for i in range(len(active)):
        for j in range(i+1, len(active)):
            if pos[j,0] - pos[i,0] > 0.5: break
            d = pos[j] - pos[i]
            d2 = np.dot(d,d)
            if d2 < 0.25 and d2 > 1e-8:
                dist = np.sqrt(d2)
                shift = 0.5*(0.5-dist)*(d/dist)
                active[i].pos -= shift
                active[j].pos += shift
                pos[i] -= shift; pos[j] += shift
                
    return active

# ============================================================
# 3. GENERATOR & SAVE LOGIC
# ============================================================
# Global State
agents = []
uid = 0
spawned = 0
sim_time = 0.0
last_spawn = 0.0

# Setup Plots
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(0, 20); ax.set_ylim(0, 4); ax.set_aspect('equal')
ax.axhline(0, c='k', lw=3); ax.axhline(4, c='k', lw=3)
bx = CONFIG["barrier"]["x"]
ax.plot([bx, bx], [0, wall_y_bottom], 'k-', lw=5)
ax.plot([bx, bx], [wall_y_top, 4.0], 'k-', lw=5)
scat = ax.scatter([], [], s=80, edgecolors='k', alpha=0.9)
txt = ax.text(0.02, 0.9, 'Initializing...', transform=ax.transAxes, fontweight='bold')

def frame_generator():
    global sim_time, agents, spawned, uid, last_spawn
    
    print("--- 1. SPEED RUN: Pre-rolling to 90s (Headless) ---")
    while sim_time < 90.0:
        sim_time += CONFIG["dt"]
        
        # Spawn Logic
        if spawned < CONFIG["max_agents"]:
            if sim_time - last_spawn > (1.0/CONFIG["spawn_rate"]):
                agents.append(Agent(uid, [0.5, np.random.uniform(0.5, 3.5)], 19.5))
                uid+=1; spawned+=1
                if spawned < CONFIG["max_agents"]:
                    agents.append(Agent(uid, [19.5, np.random.uniform(0.5, 3.5)], 0.5))
                    uid+=1; spawned+=1
                last_spawn = sim_time
        
        agents = update_physics(agents)
        
        # Progress Indicator
        if int(sim_time) % 10 == 0 and abs(sim_time - int(sim_time)) < 0.02:
            print(f"    Simulating... T={int(sim_time)}s | Agents: {len(agents)}")

    print(f"--- 2. RECORDING: T=90s to Finish ---")
    
    # Recording Loop
    while len(agents) > 0:
        # Speed Run: Process 5 steps per frame (5x speed)
        for _ in range(5):
            sim_time += CONFIG["dt"]
            
            # Continue spawning if not done (unlikely at 90s but safe)
            if spawned < CONFIG["max_agents"]:
                if sim_time - last_spawn > (1.0/CONFIG["spawn_rate"]):
                    agents.append(Agent(uid, [0.5, np.random.uniform(0.5, 3.5)], 19.5))
                    uid+=1; spawned+=1
                    if spawned < CONFIG["max_agents"]:
                        agents.append(Agent(uid, [19.5, np.random.uniform(0.5, 3.5)], 0.5))
                        uid+=1; spawned+=1
                    last_spawn = sim_time
            
            agents = update_physics(agents)
            
        yield agents # Yield to plot

def update_plot(active_agents):
    if not active_agents:
        scat.set_offsets(np.empty((0,2)))
        return scat, txt
        
    pos = np.array([a.pos for a in active_agents])
    scat.set_offsets(pos)
    colors = ['#d32f2f' if a.target_x > 10 else '#1976d2' for a in active_agents]
    scat.set_facecolors(colors)
    txt.set_text(f"TIME: {sim_time:.1f}s | REMAINING: {len(active_agents)}")
    return scat, txt

# Create Animation
ani = FuncAnimation(fig, update_plot, frames=frame_generator, blit=True, 
                    save_count=2000) # Max frames to prevent overflow

print("Saving GIF (this may take a moment)...")
ani.save('simulation_speedrun.gif', writer='pillow', fps=30)
print("Done! Saved as 'simulation_speedrun.gif'")