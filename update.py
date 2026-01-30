import numpy as np
from forces import calculate_state_forces

def resolve_agent_collisions(agents):
    """
    Hard-shell collision resolution.
    If agents overlap, push them apart to exactly radius+radius distance.
    Uses 'Sweep and Prune' (Sort by X) to keep performance high with 600+ agents.
    """
    # 1. Sort agents by X-axis to optimize checks (O(N log N))
    # This allows us to break the inner loop early, avoiding N^2 lag.
    agents.sort(key=lambda ag: ag.pos[0])
    
    for i in range(len(agents)):
        a1 = agents[i]
        
        # Only check neighbors to the right in the sorted list
        for j in range(i + 1, len(agents)):
            a2 = agents[j]
            
            # X-Axis Pruning
            dx = a2.pos[0] - a1.pos[0]
            radii_sum = a1.radius + a2.radius
            
            # If horizontal distance is already too big, no subsequent agent can touch a1
            if dx > radii_sum:
                break
            
            # Y-Axis Pruning (Quick check)
            dy = a2.pos[1] - a1.pos[1]
            if abs(dy) > radii_sum:
                continue

            # Precise Circle Check
            dist_sq = dx*dx + dy*dy
            min_dist_sq = radii_sum * radii_sum
            
            if dist_sq < min_dist_sq:
                dist = np.sqrt(dist_sq)
                if dist < 1e-6: continue # Prevent division by zero
                
                # Calculate Overlap
                overlap = radii_sum - dist
                
                # Normal vector pointing from a1 to a2
                nx = dx / dist
                ny = dy / dist
                
                # Displace both agents equally to resolve overlap
                # Each moves half the overlap distance away from the other
                correction = 0.5 * overlap
                
                a1.pos[0] -= correction * nx
                a1.pos[1] -= correction * ny
                
                a2.pos[0] += correction * nx
                a2.pos[1] += correction * ny

def update_all_agents(agents, walls, cfg, dt):
    # 1. Snapshot Phase: Calculate all forces
    forces = []
    for agent in agents:
        F = calculate_state_forces(agent, agents, walls, cfg)
        forces.append(F)

    # 2. Integration Phase: Apply forces to Velocity/Position
    for i, agent in enumerate(agents):
        F = forces[i]
        
        # Euler Integration
        agent.vel += F * dt

        # Hard Speed Limit (Safety)
        speed = np.linalg.norm(agent.vel)
        if speed > agent.max_speed:
            agent.vel = (agent.vel / speed) * agent.max_speed

        # Position Update
        agent.pos += agent.vel * dt
        
    # 3. Collision Resolution Phase (Hard Shell)
    # Corrects positions to ensure minimal body compression
    resolve_agent_collisions(agents)