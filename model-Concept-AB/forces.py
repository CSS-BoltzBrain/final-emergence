import numpy as np

# ABSOLUTE SAFETY LIMIT
MAX_FORCE = 50.0 

def unit(vec, eps=1e-8):
    n = np.linalg.norm(vec)
    if n < eps: return np.zeros_like(vec)
    return vec / n

def get_neighbors(agent, agents, radius):
    neighbors = []
    r_sq = radius**2
    for other in agents:
        if agent.id == other.id: continue
        
        # Fast bounding box check
        if abs(agent.pos[0] - other.pos[0]) > radius: continue
        if abs(agent.pos[1] - other.pos[1]) > radius: continue
        
        # Precise distance check
        dist_sq = np.sum((agent.pos - other.pos)**2)
        if dist_sq < r_sq:
            neighbors.append(other)
    return neighbors

# --- 1. GRADIENT COMPONENT ---
def gradient_force(agent, cfg):
    """
    Driving force towards the target.
    Logic: Strength * Desired Direction
    """
    strength = cfg["forces"]["gradient"]["strength"]
    return strength * agent.corridor_gradient()

# --- 2. REPULSIVE COMPONENT ---
def repulsive_force(agent, neighbors, walls, cfg):
    """
    Exponential Surface-Distance Repulsion.
    Formula: n_vec * exp(-decay * (distance - diameter))
    """
    F_rep = np.zeros(2)
    decay = cfg["forces"]["repulsive"]["decay"]
    
    # Diameter = 2 * Radius
    agent_dia = 2.0 * agent.radius
    
    # A. Agent Repulsion
    for other in neighbors:
        r_vec = agent.pos - other.pos # Vector pointing AWAY from neighbor
        dist = np.linalg.norm(r_vec)
        
        if dist < 1e-6: continue 
        
        # Normalized Direction
        n_vec = r_vec / dist
        
        # Surface Distance: (Center-to-Center) - (Diameter)
        surface_dist = dist - agent_dia
        
        # Formula: unit_vec * e^(-decay * surface_dist)
        val = np.exp(-decay * surface_dist)
        F_rep += val * n_vec

    # B. Wall Repulsion
    # Bottom Wall (y=0)
    dist_y_bot = agent.pos[1]
    if dist_y_bot < agent_dia * 2:
        surface_dist = dist_y_bot - agent.radius
        val = np.exp(-decay * surface_dist)
        F_rep += val * np.array([0.0, 1.0]) # Push UP

    # Top Wall (y=4)
    dist_y_top = 4.0 - agent.pos[1]
    if dist_y_top < agent_dia * 2:
        surface_dist = dist_y_top - agent.radius
        val = np.exp(-decay * surface_dist)
        F_rep += val * np.array([0.0, -1.0]) # Push DOWN
        
    return F_rep

# --- 3. RESISTANCE COMPONENT ---
def resistance_force(agent, f_repul_vec, cfg):
    """
    Resistance dependent on Repulsion Magnitude.
    Formula: ||F_rep|| * exp(-gamma)
    Direction: Opposite to REPULSION.
    """
    gamma = cfg["forces"]["resistance"]["gamma"]
    

    repul_norm = np.linalg.norm(f_repul_vec)
    magnitude = repul_norm * np.exp(-gamma)
    
    
    if repul_norm > 1e-6:
        direction = -f_repul_vec / repul_norm
    else:
        direction = np.zeros(2)
    
    return magnitude * direction

# --- MAIN FORCE LOGIC (WEIGHTING) ---

def calculate_state_forces(agent, agents, walls, cfg):
    """
    Applies logic switching based on agent state.
    """
    
    # 1. Random Fluctuation (Probability P)
    prob = cfg["forces"]["random"]["probability"]
    if np.random.rand() < prob:
        rand_strength = cfg["forces"]["random"]["strength"]
        return unit(np.random.randn(2)) * rand_strength

    # Pre-calculate components
    neighbors = get_neighbors(agent, agents, 2.0)
    
    # Calculate Repulsion (Needed for Resistance calculation)
    F_rep = repulsive_force(agent, neighbors, walls, cfg)
    
    # --- CHECK STATE ---
    desired_dir = agent.corridor_gradient()
    
    # Progress: Positive = Moving forward, Negative = Pushed back
    progress = np.dot(agent.vel, desired_dir)
    speed = np.linalg.norm(agent.vel)
    
    if progress > -1e-4 or speed < 0.5: 
        # --- SCENARIO: Free Flow / Driving ---
        # Gradient = 1, Repulsion = 0, Resistance = 0
        total_force = gradient_force(agent, cfg)
    else:
        # --- SCENARIO: Obstructed / Pushed Back ---
        # Gradient = 0, Repulsion = 1, Resistance = 1
        
        # Calculate Resistance (Opposite to Repulsion now)
        F_res = resistance_force(agent, F_rep, cfg)
        
        # Apply weighting
        w_rep = cfg["forces"]["repulsive"]["strength"]
        
        # Total = Weighted Repulsion + Resistance
        total_force = (w_rep * F_rep) + F_res
    
    # Cap maximum force for stability
    f_mag = np.linalg.norm(total_force)
    if f_mag > MAX_FORCE:
        total_force = (total_force / f_mag) * MAX_FORCE
        
    return total_force