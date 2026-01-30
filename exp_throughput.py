import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION (STRICT CLONE OF ANIMATION.PY PHYSICS)
# ============================================================
CONF = {
    "dt": 0.02,
    "F_grad": 30.0,
    "F_rep": 30.0,
    "F_decay": 9.0,
    "gamma": 1.8,
    "crowd_decay": 1.1,      # Crucial for hexagonal jamming
    "wall_friction": True,   
    "random_force": False    # Deterministic (matches animation.py)
}

# ============================================================
# 2. CORE PHYSICS ENGINE
# ============================================================
def get_forces(agent, agents_pos, agent_ids):
    pos = agent["pos"]
    dx = pos[0] - agents_pos[:, 0]
    mask = np.abs(dx) < 2.0 
    
    neighbor_count = 0
    F_rep = np.zeros(2)
    
    # 1. Neighbor Repulsion
    if np.any(mask):
        local_pos = agents_pos[mask]
        local_ids = agent_ids[mask]
        valid_mask = local_ids != agent["id"]
        local_pos = local_pos[valid_mask]
        
        if len(local_pos) > 0:
            r_vec = pos - local_pos
            r_sq = np.sum(r_vec**2, axis=1)
            interact_mask = r_sq < 4.0 # 2.0m radius
            
            if np.any(interact_mask):
                neighbor_count = np.sum(interact_mask)
                r_vec = r_vec[interact_mask]
                r_sq = r_sq[interact_mask]
                # Gaussian Repulsion: A * exp(-B * r^2) * 2r
                mag = CONF["F_rep"] * np.exp(-CONF["F_decay"] * r_sq) * 2.0
                F_rep += np.sum(r_vec * mag[:, np.newaxis], axis=0)

    # 2. Driving Force with Crowd Decay
    # Drive reduces as neighbors increase (1.1^-N)
    target_dir = 1.0 if agent["target_x"] > 10 else -1.0
    drive_scale = CONF["crowd_decay"] ** (-neighbor_count)
    F_drive = np.array([target_dir * CONF["F_grad"] * drive_scale, 0.0])

    # 3. Wall Repulsion (Gaussian cushion)
    y = pos[1]
    if y < 1.0: 
        F_rep[1] += CONF["F_rep"] * np.exp(-CONF["F_decay"] * y**2) * 2.0 * y
    dy = y - 4.0
    if dy > -1.0: 
        F_rep[1] += CONF["F_rep"] * np.exp(-CONF["F_decay"] * dy**2) * 2.0 * dy

    # 4. Drag
    F_drag = -CONF["gamma"] * agent["vel"]

    return F_drive + F_rep + F_drag

def resolve_collisions(agents):
    # Hard Sphere Projection (Position Based Dynamics)
    agents.sort(key=lambda a: a["pos"][0])
    pos_arr = np.array([a["pos"] for a in agents])
    
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if pos_arr[j,0] - pos_arr[i,0] > 0.5: break 
            
            d_vec = pos_arr[j] - pos_arr[i]
            d_sq = np.dot(d_vec, d_vec)
            
            if d_sq < 0.25 and d_sq > 1e-8:
                dist = np.sqrt(d_sq)
                shift = 0.5 * (0.5 - dist) * (d_vec / dist)
                
                agents[i]["pos"] -= shift
                agents[j]["pos"] += shift
                pos_arr[i] -= shift
                pos_arr[j] += shift

# ============================================================
# 3. SIMULATION LOOP
# ============================================================
def run_trial(rate):
    WIDTH, HEIGHT = 20.0, 4.0
    MAX_AGENTS = 500
    spawn_interval = 1.0 / rate
    # Time to spawn all agents = (500 agents / 2 per step) * interval
    spawn_end_time = (MAX_AGENTS / 2) * spawn_interval
    max_sim_time = spawn_end_time + 15.0
    
    agents = []
    exit_timestamps = []
    sim_time = 0.0
    last_spawn = 0.0
    agent_counter = 0
    spawned_total = 0
    
    while sim_time < max_sim_time:
        sim_time += CONF["dt"]
        
        # --- SPAWN ---
        if spawned_total < MAX_AGENTS and (sim_time - last_spawn > spawn_interval):
            # Bidirectional Spawn (Left & Right)
            for tx in [19.5, 0.5]: 
                if spawned_total < MAX_AGENTS:
                    start_x = 0.5 if tx > 10 else 19.5
                    agents.append({
                        "id": agent_counter, 
                        "pos": np.array([start_x, np.random.uniform(0.5, 3.5)]),
                        "vel": np.zeros(2), 
                        "target_x": tx
                    })
                    agent_counter += 1; spawned_total += 1
            last_spawn = sim_time
        
        # --- UPDATE ---
        if agents:
            p_arr = np.array([a["pos"] for a in agents])
            id_arr = np.array([a["id"] for a in agents])
            forces = [get_forces(a, p_arr, id_arr) for a in agents]
            
            active = []
            for i, ag in enumerate(agents):
                # Integration
                ag["vel"] += forces[i] * CONF["dt"]
                
                # Speed Cap (1.5 m/s)
                s = np.linalg.norm(ag["vel"])
                if s > 1.5: ag["vel"] *= (1.5/s)
                
                ag["pos"] += ag["vel"] * CONF["dt"]
                
                # --- WALL STICKINESS (MATCHING ANIMATION.PY) ---
                # This kills Y-velocity if hitting the wall, causing friction
                if ag["pos"][1] < 0.25:
                    ag["pos"][1] = 0.25
                    ag["vel"][1] = 0.0
                elif ag["pos"][1] > 3.75:
                    ag["pos"][1] = 3.75
                    ag["vel"][1] = 0.0
                
                # X-Boundaries
                ag["pos"][0] = np.clip(ag["pos"][0], 0.25, 19.75)
                
                # Exit Check
                if abs(ag["pos"][0] - ag["target_x"]) < 0.5:
                    exit_timestamps.append(sim_time)
                else:
                    active.append(ag)
            
            agents = active
            resolve_collisions(agents)

    # ============================================================
    # 4. METRICS CALCULATION
    # ============================================================
    exits = np.array(exit_timestamps)
    
    # Window 1: Load Phase (30s to 40s)
    j_load = np.sum((exits >= 30.0) & (exits < 40.0)) / 10.0
    
    # Window 2: Residual Phase (Spawn End to +10s)
    j_resid = np.sum((exits >= spawn_end_time) & (exits < spawn_end_time + 10.0)) / 10.0
    
    return j_load, j_resid

# ============================================================
# 5. EXECUTION AND PLOTTING
# ============================================================
rates = [1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0]
res_load, res_resid = [], []

print(f"{'Rate':<6} | {'J (30-40s)':<12} | {'J (Resid)':<12}")
print("-" * 36)

for r in rates:
    jl, jr = run_trial(r)
    res_load.append(jl)
    res_resid.append(jr)
    print(f"{r:<6.1f} | {jl:<12.4f} | {jr:<12.4f}")

plt.figure(figsize=(8, 6), dpi=120)

# Load Phase (Blue) - Using raw string r"..." to fix SyntaxWarning
plt.plot(rates, res_load, 'o-', color='#1976D2', lw=2.5, markersize=8, 
         label=r'Load Phase ($t \in [30, 40]$)')

# Residual Phase (Red)
plt.plot(rates, res_resid, 's--', color='#D32F2F', lw=2.5, markersize=8, 
         label=r'Residual Phase ($t > T_{spawn}$)')

plt.title("Crowd Throughput Analysis", fontsize=14, fontweight='bold', pad=15)
plt.xlabel(r"Injection Rate $\lambda$ [agents/s]", fontsize=12)
plt.ylabel(r"Measured Throughput $J$ [agents/s]", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(frameon=True, shadow=True)

plt.tight_layout()
plt.show()