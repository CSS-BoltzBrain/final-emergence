import numpy as np
import matplotlib.pyplot as plt
import random

WIDTH = 100
HEIGHT = 2
TIMESTEPS = 200
SAMPLES = 60

def traffic_rule_step(grid):
    curr = grid.copy()
    after_lane_change = curr.copy()
    
    # We use a set to keep track of where cars ARE moving to during this phase
    # to prevent two cars from moving into the same spot.
    occupied_next = set()

    for j in range(HEIGHT):
        for i in range(WIDTH):
            if curr[j, i] == 1:
                forward = (i + 1) % WIDTH
                other_lane = 1 - j # Switches 0 to 1 and 1 to 0
                
                # CASE: IN THE LEFT LANE (j=0) -> Move Right if possible
                if j == 0:
                    # Move right if the spot is clear AND nobody is there in the next state
                    if curr[other_lane, i] == 0 and curr[other_lane, forward] == 0:
                        after_lane_change[j, i] = 0
                        after_lane_change[other_lane, i] = 1
                
                # CASE: IN THE RIGHT LANE (j=1) -> Move Left only if blocked
                elif j == 1:
                    if curr[j, forward] == 1: # Blocked in front
                        # Move left if side is clear
                        if curr[other_lane, i] == 0 and curr[other_lane, forward] == 0:
                            after_lane_change[j, i] = 0
                            after_lane_change[other_lane, i] = 1

    # Phase 2: Forward Movement
    # We check 'after_lane_change' for positions, but update 'final_grid'
    final_grid = after_lane_change.copy()
    moves = 0
    for j in range(HEIGHT):
        for i in range(WIDTH):
            forward = (i + 1) % WIDTH
            # Standard movement: if car exists and spot in front is empty
            if after_lane_change[j, i] == 1 and after_lane_change[j, forward] == 0:
                final_grid[j, i] = 0
                final_grid[j, forward] = 1
                moves += 1
                
    return final_grid, moves

def calculate_throughput(density):
    # Fix: Explicitly create a 2D zeros array
    grid = np.zeros((HEIGHT, WIDTH))
    
    # Fix: Ensure we fill it correctly as a 2D structure
    num_cars = int(WIDTH * HEIGHT * density)
    indices = random.sample(range(WIDTH * HEIGHT), num_cars)
    for idx in indices:
        row = idx // WIDTH
        col = idx % WIDTH
        grid[row, col] = 1
        
    total_moves = 0
    # Stabilization period
    for _ in range(50):
        grid, _ = traffic_rule_step(grid)
        
    # Measurement period
    for _ in range(TIMESTEPS):
        grid, moves = traffic_rule_step(grid)
        total_moves += moves
        
    return total_moves / (WIDTH * HEIGHT * TIMESTEPS)

# --- Execution ---
densities = np.linspace(0, 1, SAMPLES)
throughputs = [calculate_throughput(d) for d in densities]

plt.figure(figsize=(10, 6))
plt.plot(densities, throughputs, 'o-', color='indigo', markersize=4)
plt.title("2-Lane Fundamental Diagram ")
plt.xlabel("Density")
plt.ylabel("Throughput")
plt.grid(True, alpha=0.3)
plt.show()