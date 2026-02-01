import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import powerlaw
import argparse

parser = argparse.ArgumentParser(description="Crowd Simulation Parameters")
parser.add_argument("--width", type=int, default=100, help="Grid width")
parser.add_argument("--height", type=int, default=100, help="Grid height")
parser.add_argument("--people", type=int, default=6000, help="Number of people")
parser.add_argument("--steps", type=int, default=400, help="Max timesteps")
parser.add_argument("--runs", type=int, default=20, help="Number of runs")
parser.add_argument("--p_change", type=float, default=0.005, help="Probability to change directions")
parser.add_argument("--p_range", type=list[float], default=[0.005, 0.01, 0.05, 0.1, 0.2], help="Probability to change directions list")
parser.add_argument("--pop_range", type=list[int], default=[1000, 3000, 5000, 6000, 7000, 8000], help="Populations list")
parser.add_argument("--heatmap", type=bool, default=False, help="Plot heatmap?")
parser.add_argument("--powerlaw", type=bool, default=False, help="Plot powerlaw?")
parser.add_argument("--save", type=bool, default=True, help="Save animation?")

args = parser.parse_args()

WIDTH, HEIGHT = args.width, args.height
PEOPLE = args.people
TIMESTEPS = args.steps
NUM_RUNS = args.runs
P_CHANGE = args.p_change
DIRECTIONS = [(1,0), (-1,0), (0,1), (0,-1)]
p_range = [0.005, 0.01, 0.05, 0.1, 0.2]
pop_range = [1000, 3000, 5000, 6000, 7000, 8000]
save = args.save
heatmap = args.heatmap
power = args.powerlaw

obstacles = set()
# No obstacles for now

class Person:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.dx, self.dy = random.choice(DIRECTIONS)

def find_cluster_sizes(people):
    occupied = {(p.x, p.y) for p in people}
    visited = set()
    sizes = []
    for cell in occupied:
        if cell in visited: continue
        stack, count = [cell], 0
        visited.add(cell)
        while stack:
            curr = stack.pop()
            count += 1
            for dx, dy in DIRECTIONS:
                nb = (curr[0]+dx, curr[1]+dy)
                if nb in occupied and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        sizes.append(count)
    return sizes

def collect_distribution_data(p_change):
    all_sizes = []
    for run in range(NUM_RUNS):
        people = []
        occ = set()
        while len(people) < PEOPLE:
            x, y = random.randrange(WIDTH), random.randrange(HEIGHT)
            if (x, y) not in obstacles and (x, y) not in occ:
                people.append(Person(x, y))
                occ.add((x, y))
        
        for t in range(TIMESTEPS):
            blocked = {(p.x, p.y) for p in people} | obstacles
            for person in people:
                if random.random() < p_change:
                    person.dx, person.dy = random.choice(DIRECTIONS)
                nx, ny = person.x + person.dx, person.y + person.dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and (nx, ny) not in blocked:
                    blocked.discard((person.x, person.y))
                    person.x, person.y = nx, ny
                    blocked.add((nx, ny))
            
            # Sample only at the end to ensure we are in steady-state
            if t > TIMESTEPS - 50:
                all_sizes.extend(find_cluster_sizes(people))
    return all_sizes

if power:
    # Execution and Plotting with powerlaw package 
    plt.figure(figsize=(12, 7))

    p = P_CHANGE
    distribution = collect_distribution_data(p)

    fit = powerlaw.Fit(distribution, discrete=True, verbose=False)

    # 2. Plot the Empirical PDF (Probability Density Function)
    fit.plot_pdf(label=f'p = {p} (α={fit.alpha:.2f})', marker='o', linestyle='', alpha=0.8)

    # 3. Comparison
    fit.power_law.plot_pdf(label = f'powerlaw fit, alpha = {fit.alpha:.2f}', linestyle='--', ax=plt.gca(),)
    R, p_val = fit.distribution_compare('power_law', 'lognormal')
    print(f"p={p}: Alpha={fit.alpha:.2f}, xmin={fit.xmin}, R={R:.2f}, p-value={p_val:.4f}, R = {R}")

    plt.title(f"Cluster Size Distribution: p-value={p_val:.4f}, R={R:.2f}")
    plt.xlabel("Cluster Size (s)")
    plt.ylabel("P(s)")
    plt.legend()

    if save:
        plt.savefig('model_random_probability/results/powerlaw.png')
    else:
        plt.show()


if heatmap:
    # Initialize a matrix to store R-values
    r_matrix = np.zeros((len(p_range), len(pop_range)))


    for i, p_val in enumerate(p_range):
        for j, pop_val in enumerate(pop_range):
            PEOPLE = pop_val 
            distribution = collect_distribution_data(p_val)
            
            # Fit and compare
            fit = powerlaw.Fit(distribution, discrete=True, verbose=False)
            R, _ = fit.distribution_compare('power_law', 'lognormal')
            
            r_matrix[i, j] = R
            print(f"Tested p={p_val}, Pop={pop_val} -> R={R:.2f}")

    # Save the matrix to a CSV in case you want to plot it later without re-running
    np.savetxt("phase_data_R.csv", r_matrix, delimiter=",")

    plt.figure(figsize=(10, 8))

    # Create the heatmap
    plt.imshow(r_matrix, origin='lower', aspect='auto', cmap='RdBu_r', 
            extent=[min(pop_range), max(pop_range), 0, len(p_range)-1])

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label('Log-Likelihood Ratio (R)')

    # Set labels
    plt.xticks(pop_range)
    plt.yticks(range(len(p_range)), p_range)
    plt.xlabel('Population size')
    plt.ylabel('p')
    plt.title('Phase Diagram')

    if save:
        plt.savefig('model_random_probability/results/heatmap.png')