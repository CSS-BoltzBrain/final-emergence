import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# --- CONFIGURATION (Start with User Config) ---
CONFIG = {
    "simulation": {"dt": 0.02, "steps": 1000}, # 20 seconds
    "domain": {"xmin": 0.0, "xmax": 20.0, "ymin": 0.0, "ymax": 4.0},
    "agent": {
        "radius": 0.25,
        "desired_speed": 1.3,
        "max_speed": 1.5,
        "perception_radius": 2.5,
        "crowd_decay": 1.1  
    },
    "forces": {
        "gradient":   {"strength": 30.0}, 
        "repulsive":  {"strength": 30.0, "decay": 9.0},
        "resistance": {"gamma": 1.8}, 
        # DISABLE RANDOMNESS for Lyapunov Analysis
        "random":     {"probability": 0.0, "strength": 0.0}
    }
}

# --- PHYSICS ENGINE (Replicating your logic) ---

class Agent:
    def __init__(self, id, pos, target_x, cfg):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.radius = cfg["agent"]["radius"]
        self.target_x = target_x
        self.max_speed = cfg["agent"]["max_speed"]
        self.direction = np.array([1.0 if target_x > pos[0] else -1.0, 0.0])

    def corridor_gradient(self):
        return self.direction

def unit(vec, eps=1e-8):
    n = np.linalg.norm(vec)
    return vec / n if n > eps else np.zeros_like(vec)

def calculate_forces(agent, agents, cfg):
    # 1. Neighbors
    neighbors = []
    for other in agents:
        if agent.id != other.id:
            if np.linalg.norm(agent.pos - other.pos) < 2.5:
                neighbors.append(other)

    # 2. Gradient Force
    F_grad = cfg["forces"]["gradient"]["strength"] * agent.corridor_gradient()

    # 3. Repulsive Force (Exponential)
    F_rep = np.zeros(2)
    decay = cfg["forces"]["repulsive"]["decay"]
    dia = 2.0 * agent.radius
    
    # Agent Repulsion
    for other in neighbors:
        r_vec = agent.pos - other.pos
        dist = np.linalg.norm(r_vec)
        if dist > 1e-6:
            val = np.exp(-decay * (dist - dia))
            F_rep += val * (r_vec / dist)

    # Wall Repulsion (Top/Bottom)
    y = agent.pos[1]
    if y < dia * 2: F_rep += np.exp(-decay * (y - agent.radius)) * np.array([0, 1])
    if (4.0 - y) < dia * 2: F_rep += np.exp(-decay * ((4.0 - y) - agent.radius)) * np.array([0, -1])

    # 4. State Machine Logic
    # Check alignment
    progress = np.dot(agent.vel, agent.corridor_gradient())
    speed = np.linalg.norm(agent.vel)
    
    # "Free Flow" if moving forward OR moving slowly (stuck fix)
    if progress > -1e-4 or speed < 0.5:
        return F_grad
    else:
        # "Obstructed"
        F_res = np.zeros(2)
        rep_norm = np.linalg.norm(F_rep)
        if rep_norm > 1e-6:
            # Resistance opposes Repulsion
            gamma = cfg["forces"]["resistance"]["gamma"]
            F_res = -1.0 * (F_rep / rep_norm) * (rep_norm * np.exp(-gamma))
        
        w_rep = cfg["forces"]["repulsive"]["strength"]
        return (w_rep * F_rep) + F_res

def run_simulation_scenario(perturbation=0.0):
    """
    Runs a head-on collision scenario.
    Agent 0 is perturbed by `perturbation`.
    """
    # Scenario: Head-on collision to trigger the "Obstructed" state logic
    a1 = Agent(0, [0.0, 2.0 + perturbation], 20.0, CONFIG)
    a2 = Agent(1, [20.0, 2.01], 0.0, CONFIG) # Slightly offset y to ensure sliding, not perfect blocking
    agents = [a1, a2]
    
    dt = CONFIG["simulation"]["dt"]
    trajectory = []

    for _ in range(CONFIG["simulation"]["steps"]):
        # 1. Calc Forces
        forces = [calculate_forces(ag, agents, CONFIG) for ag in agents]
        
        # 2. Update Physics
        for i, ag in enumerate(agents):
            ag.vel += forces[i] * dt
            
            # Speed Cap
            s = np.linalg.norm(ag.vel)
            if s > ag.max_speed: ag.vel = (ag.vel / s) * ag.max_speed
            
            ag.pos += ag.vel * dt
        
        # 3. Simple Hard Shell Resolution (Crucial for divergence)
        dist = np.linalg.norm(agents[0].pos - agents[1].pos)
        min_dist = agents[0].radius + agents[1].radius
        if dist < min_dist:
            overlap = min_dist - dist
            n = (agents[0].pos - agents[1].pos) / dist
            agents[0].pos += 0.5 * overlap * n
            agents[1].pos -= 0.5 * overlap * n

        trajectory.append(agents[0].pos.copy()) # Track Agent 0
        
    return np.array(trajectory)

# --- EXECUTION ---

# Run Reference and Perturbed
print("Running Reference Simulation...")
traj_ref = run_simulation_scenario(perturbation=0.0)

print("Running Perturbed Simulation (delta = 1e-8)...")
traj_pert = run_simulation_scenario(perturbation=1e-8)

# Calculate Divergence (Euclidean distance at each step)
divergence = np.linalg.norm(traj_ref - traj_pert, axis=1)

# Handle log(0)
divergence[divergence == 0] = 1e-16
log_div = np.log(divergence)
time_axis = np.arange(len(log_div)) * CONFIG["simulation"]["dt"]

# --- PLOTTING ---

plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'sans-serif'

# Plot Data
plt.plot(time_axis, log_div, color='#2c3e50', linewidth=2, label=r'$\ln||\delta \mathbf{x}(t)||$')

# Fit Lyapunov Exponent (slope) on the rising edge (interaction phase)
# Agents meet approx around t=6s to t=10s
interaction_mask = (time_axis > 7.0) & (time_axis < 9.0)
if np.any(interaction_mask):
    slope, intercept = np.polyfit(time_axis[interaction_mask], log_div[interaction_mask], 1)
    
    # Plot Regression Line
    plt.plot(time_axis[interaction_mask], slope * time_axis[interaction_mask] + intercept, 
             color='#e74c3c', linestyle='--', linewidth=2, label=f'Fit $\lambda \\approx {slope:.2f}$')

plt.title("Lyapunov Divergence: Sensitivity to Initial Conditions", fontsize=14, pad=15)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel(r"Log Separation Distance ($\ln \delta$)", fontsize=12)

plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(frameon=True, fontsize=10, loc='lower right')
plt.tight_layout()

# Save
plt.savefig('lyapunov_divergence.png', dpi=300)
print("Plot saved as 'lyapunov_divergence.png'")
plt.show()