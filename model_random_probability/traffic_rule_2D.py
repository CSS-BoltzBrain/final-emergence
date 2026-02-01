import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

WIDTH = 20
HEIGHT = 2
TARGET_DENSITY_RIGHT = 0.7
TARGET_DENSITY_LEFT = 0.5
TIMESTEPS = 100
SAMPLES =50

starting_array = np.array([[1 if random.random() < TARGET_DENSITY_LEFT else 0 for _ in range(WIDTH)], 
                           [1 if random.random() < TARGET_DENSITY_RIGHT else 0 for _ in range(WIDTH)]])

def traffic_rule_step(grid):
    """Moves forward only if the space ahead is empty."""
    curr = grid.copy()
    after_lane_change = curr.copy()

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

    # Forward Movement
    final_grid = after_lane_change.copy()
    for j in range(HEIGHT):
        for i in range(WIDTH):
            forward = (i + 1) % WIDTH
            if after_lane_change[j, i] == 1 and after_lane_change[j, forward] == 0:
                final_grid[j, i] = 0
                final_grid[j, forward] = 1
                
    return final_grid

def traffic_rule(starting_array, timesteps):
    """
    Executes the traffic rule step for the chosen number of timesteps,
    remembers and returns all configurations
    """
    current_array = starting_array
    configurations = [current_array]
    for _ in range(timesteps):
        next_array = traffic_rule_step(current_array)
        current_array = next_array
        configurations.append(current_array)
    return configurations

configurations = traffic_rule(starting_array, TIMESTEPS)

# Visualization
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(-0.5, WIDTH - 0.5)
ax.set_ylim(-0.5, HEIGHT - 0.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Lane 0', 'Lane 1'])
ax.axhline(0.5, color='black', lw=1, alpha=0.2) # Lane divider

dots, = ax.plot([], [], 'o', markersize=12, color='royalblue')

def init():
    dots.set_data([], [])
    return dots,

def update(frame):
    """Updates the global state for each animation frame"""
    grid = configurations[frame]
    y_coords, x_coords = np.where(grid == 1)
    dots.set_data(x_coords, y_coords)
    return dots,

anim = FuncAnimation(fig, update, frames=len(configurations), 
                    init_func=init, blit=True, interval=150, repeat=True)

plt.show()
