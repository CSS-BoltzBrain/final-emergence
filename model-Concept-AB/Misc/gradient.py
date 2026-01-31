import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
WIDTH = 12.0
HEIGHT = 6.0
WALL_X = 6.0
GAP_CENTER_Y = 3.0
GAP_WIDTH = 1.1

# Grid Resolution
res = 0.05
x = np.arange(0, WIDTH + res, res)
y = np.arange(0, HEIGHT + res, res)
X, Y = np.meshgrid(x, y)

# Target Point (End of the Red Agents' journey)
TARGET_X = 11.5
TARGET_Y = 3.0

# Bottleneck Point (The "Waypoint")
GAP_X = WALL_X
GAP_Y = GAP_CENTER_Y

# --- 2. CALCULATE "WALKING DISTANCE" POTENTIAL ---
# This creates a "Height Map" where the Target is the lowest point (0).
# Agents want to roll downhill.

def get_distance(x, y):
    # Distance to the target directly
    dist_direct = np.sqrt((x - TARGET_X)**2 + (y - TARGET_Y)**2)
    
    # Distance via the Gap (Waypoint)
    dist_to_gap = np.sqrt((x - GAP_X)**2 + (y - GAP_Y)**2)
    dist_gap_to_target = np.sqrt((GAP_X - TARGET_X)**2 + (GAP_Y - TARGET_Y)**2)
    path_via_gap = dist_to_gap + dist_gap_to_target
    
    # LOGIC:
    # If we are in the Right Room (x > WALL_X), we just go to target.
    # If we are in the Left Room (x < WALL_X), we MUST go through the gap.
    # Note: We add a small buffer to WALL_X to handle the gap crossing smoothly.
    
    is_past_wall = x > WALL_X
    
    # Vectorized 'where' (like an if-statement for grids)
    # If past wall: Direct Distance. Else: Path via Gap.
    return np.where(is_past_wall, dist_direct, path_via_gap)

Z = get_distance(X, Y)

# --- 3. CALCULATE GRADIENT (VELOCITY VECTORS) ---
# The desired velocity is the negative gradient of the distance.
# "Which way is downhill?"
dy, dx = np.gradient(-Z) # Negative because we want to decrease distance

# Normalize vectors (we care about direction, not magnitude here)
speed = np.sqrt(dx**2 + dy**2)
# Avoid division by zero
speed[speed == 0] = 1.0 
U = dx / speed
V = dy / speed

# --- 4. VISUALIZATION ---
plt.figure(figsize=(12, 6))

# A. Plot the Heatmap (Potential Field)
# Darker Blue = Closer to Target (Low Potential)
plt.imshow(Z, extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='Blues_r', alpha=0.5)
plt.colorbar(label="Walking Distance to Target (m)")

# B. Plot Streamlines (The Flow)
# This connects the arrows to show the 'path' agents will take
strm = plt.streamplot(X, Y, U, V, color='red', linewidth=1.5, density=1.5, arrowsize=1.5)

# C. Draw the Physical Walls
# Bottom Wall
plt.plot([WALL_X, WALL_X], [0.0, GAP_CENTER_Y - GAP_WIDTH/2], 'k-', linewidth=8)
# Top Wall
plt.plot([WALL_X, WALL_X], [GAP_CENTER_Y + GAP_WIDTH/2, HEIGHT], 'k-', linewidth=8)
# Borders
plt.axhline(0, color='k', linewidth=4)
plt.axhline(HEIGHT, color='k', linewidth=4)



plt.title("Smart Navigation Field: Pathfinding via Bottleneck")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.xlim(0, WIDTH)
plt.ylim(0, HEIGHT)
plt.gca().set_aspect('equal')

filename = "smart_gradient_map.png"
plt.savefig(filename, dpi=150)
print(f"Smart gradient map saved to {filename}")
plt.show()