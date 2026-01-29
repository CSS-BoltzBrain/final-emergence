import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import powerlaw

# Parameters 
WIDTH, HEIGHT = 100, 100
PEOPLE =6000
TIMESTEPS = 400
NUM_RUNS = 20
P_CHANGE = 0.05
DIRECTIONS = [(1,0), (-1,0), (0,1), (0,-1)]

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
plt.show()

# parameters
p_range = [0.005, 0.01, 0.05, 0.1, 0.2]
pop_range = [1000, 3000, 5000, 6000, 7000, 8000]

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


plt.savefig('model_random_probability/results/my_phase_diagram.png')