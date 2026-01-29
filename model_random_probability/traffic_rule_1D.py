import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

WIDTH = 200
HEIGHT = 2
TARGET_DENSITY = 0.6  # Adjust this to see different traffic flows
TIMESTEPS = 100
SAMPLES = 50

# Initialize starting array 
starting_array = [1 if random.random() < 0.7 else 0 for _ in range(WIDTH)]

def traffic_rule_step(array):
    """Moves forward only if the space ahead is empty."""
    next_array = array.copy()
    
    for i in range(WIDTH):
        # Check current cell and the next cell
        target_idx = (i + 1) % WIDTH
        
        # Move car only if space is empty
        if array[i] == 1 and array[target_idx] == 0:
            next_array[i] = 0
            next_array[target_idx] = 1
            
    return next_array

def traffic_rule(starting_array, timesteps):
    current_array = starting_array
    configurations = [current_array]
    for i in range(timesteps):
        next_array = traffic_rule_step(current_array)
        current_array = next_array
        configurations.append(current_array)
    return configurations

configurations = traffic_rule(starting_array, TIMESTEPS)

# Visualization
fig, ax = plt.subplots(figsize=(10, 2))
ax.set_xlim(-1, WIDTH)
ax.set_ylim(-0.5, 1.5)
ax.axis('off')

dots, = ax.plot([], [], 'o', markersize=12, color='royalblue')

def init():
    dots.set_data([], [])
    return dots,

def update(frame):
    array = configurations[frame]
    x = [i for i, val in enumerate(array) if val == 1]
    y = [0.5] * len(x)
    dots.set_data(x, y)
    return dots,

anim = FuncAnimation(fig, update, frames=len(configurations), 
                    init_func=init, blit=True, interval=200, repeat=True)

plt.show()

anim.save("model_random_probability/results/traffic_rule.gif", writer='pillow', fps=10)

# def get_throughput(density):
#     # Initialize road
#     num_cars = int(WIDTH * density)
#     array = np.zeros(WIDTH)
#     indices = np.random.choice(range(WIDTH), num_cars, replace=False)
#     array[indices] = 1
    
#     total_moves = 0
    
#     # Run simulation
#     for _ in range(TIMESTEPS):
#         next_array = array.copy()
#         for i in range(WIDTH):
#             target = (i + 1) % WIDTH
#             if array[i] == 1 and array[target] == 0:
#                 next_array[i] = 0
#                 next_array[target] = 1
#                 total_moves += 1 # Count every successful move
#         array = next_array
        
#     # Throughput = total moves / (road length * time)
#     return total_moves / (WIDTH * TIMESTEPS)

# # Collect data
# densities = np.linspace(0, 1, SAMPLES)
# throughputs = [get_throughput(d) for d in densities]

# # Plotting
# plt.figure(figsize=(8, 5))
# plt.plot(densities, throughputs, 'o-', color='crimson', markersize=4)
# plt.title("Fundamental Diagram of Traffic Flow")
# plt.xlabel("Density")
# plt.ylabel("Throughput (Flow Rate)")
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()
