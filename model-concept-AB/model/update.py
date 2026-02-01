import numpy as np

# Import compatibility (package vs standalone)
try:
    from .forces import calculate_forces
except ImportError:  # pragma: no cover
    from forces import calculate_forces


def update_physics(pos, vel, target, active_mask, walls, cfg, dt):
    """Advance one time step in-place for active agents."""
    active_idx = np.where(active_mask)[0]
    if len(active_idx) == 0:
        return

    p = pos[active_idx]
    v = vel[active_idx]
    t = target[active_idx]

    forces = calculate_forces(p, v, t, walls, cfg)

    v += forces * dt

    max_speed = cfg["agent"]["max_speed"]
    speeds = np.linalg.norm(v, axis=1)
    high_speed = speeds > max_speed
    if np.any(high_speed):
        scale = max_speed / speeds[high_speed]
        v[high_speed] *= scale[:, None]

    p += v * dt

    radius = cfg["agent"]["radius"]
    for wall in walls:
        if wall.type == "horizontal":
            dy = p[:, 1] - wall.y
            mask_hit = np.abs(dy) < radius

            if hasattr(wall, "x_range") and wall.x_range is not None:
                mask_hit &= (p[:, 0] >= wall.x_range[0]) & (p[:, 0] <= wall.x_range[1])

            if np.any(mask_hit):
                signs = np.sign(dy[mask_hit])
                p[mask_hit, 1] = wall.y + signs * radius
                v[mask_hit, 1] = 0.0

        elif wall.type == "vertical":
            dx = p[:, 0] - wall.x
            mask_hit = np.abs(dx) < radius

            if hasattr(wall, "y_range") and wall.y_range is not None:
                mask_hit &= (p[:, 1] >= wall.y_range[0]) & (p[:, 1] <= wall.y_range[1])

            if np.any(mask_hit):
                signs = np.sign(dx[mask_hit])
                p[mask_hit, 0] = wall.x + signs * radius
                v[mask_hit, 0] = 0.0

    p = resolve_collisions(p, radius)

    pos[active_idx] = p
    vel[active_idx] = v


def resolve_collisions(pos, radius):
    n = len(pos)
    if n < 2:
        return pos

    delta = pos[:, None, :] - pos[None, :, :]
    d2 = np.sum(delta**2, axis=2)
    np.fill_diagonal(d2, np.inf)

    min_dist = 2 * radius
    mask = d2 < (min_dist**2)

    i_idx, j_idx = np.where(np.triu(mask))
    if len(i_idx) == 0:
        return pos

    p_i = pos[i_idx]
    p_j = pos[j_idx]
    vec = p_i - p_j
    dist = np.linalg.norm(vec, axis=1)
    dist[dist < 1e-6] = 1e-6

    overlap = min_dist - dist
    correction = (vec / dist[:, None]) * (0.5 * overlap[:, None])

    np.add.at(pos, i_idx, correction)
    np.add.at(pos, j_idx, -correction)
    return pos


def check_exits(pos, target, active_mask):
    idx = np.where(active_mask)[0]
    if len(idx) == 0:
        return []

    p = pos[idx]
    t = target[idx]

    is_right = t[:, 0] > 10.0
    exited_right = is_right & (p[:, 0] > (t[:, 0] - 0.2))
    exited_left = (~is_right) & (p[:, 0] < (t[:, 0] + 0.2))

    return idx[exited_right | exited_left]
