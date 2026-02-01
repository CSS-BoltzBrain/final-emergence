import numpy as np

def calculate_forces(pos, vel, target, walls, cfg):
    """Return per-agent force vectors (same shape as pos)."""
    n = len(pos)
    if n == 0: return np.zeros((0, 2))
    
    forces = np.zeros((n, 2), dtype=np.float32)
    
    p_grad = cfg["forces"]["gradient"]["strength"]
    p_rep = cfg["forces"]["repulsive"]["strength"]
    p_decay = cfg["forces"]["repulsive"]["decay"]
    p_gamma = cfg["forces"]["resistance"]["gamma"]
    p_rand = cfg["forces"]["random"]["probability"]
    s_rand = cfg["forces"]["random"]["strength"]
    
    diff = target - pos
    dists = np.linalg.norm(diff, axis=1)
    
    mask_move = dists > 1e-6
    dirs = np.zeros_like(diff)
    dirs[mask_move] = diff[mask_move] / dists[mask_move, None]
    
    forces += dirs * p_grad
    
    r_vec = pos[:, None, :] - pos[None, :, :]
    d2 = np.sum(r_vec**2, axis=2)
    np.fill_diagonal(d2, np.inf)
    
    perception_sq = cfg["agent"]["perception_radius"] ** 2
    mask_interact = d2 < perception_sq
    
    if np.any(mask_interact):
        coeffs = p_rep * np.exp(-p_decay * d2) * 2.0
        coeffs[~mask_interact] = 0.0
        
        f_rep = np.sum(r_vec * coeffs[:, :, None], axis=1)
        forces += f_rep
        
        f_res = -f_rep * np.exp(-p_gamma)
        forces += f_res

    if p_rand > 0.0 and s_rand != 0.0:
        rand_mask = np.random.rand(n) < p_rand
        if np.any(rand_mask):
            theta = np.random.uniform(0.0, 2.0 * np.pi, size=rand_mask.sum())
            rand_vec = np.column_stack((np.cos(theta), np.sin(theta))) * s_rand
            forces[rand_mask] += rand_vec

    return forces
