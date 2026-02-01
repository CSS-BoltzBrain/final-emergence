"""
Each person moves in an initially random chosen direction for as long as possible.
At each move, with probability p the person changes direction.
A person only attempts to move when the target cell is not occupied and not an obstacle
If the move is blocked, the person stays in place.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import random
import argparse

parser = argparse.ArgumentParser(description="Crowd Simulation Parameters")
parser.add_argument("--width", type=int, default=30, help="Grid width")
parser.add_argument("--height", type=int, default=30, help="Grid height")
parser.add_argument("--people", type=int, default=300, help="Number of people")
parser.add_argument("--steps", type=int, default=400, help="Max timesteps")
parser.add_argument("--runs", type=int, default=20, help="Number of runs")
parser.add_argument("--p_change", type=float, default=0.005, help="Probability to change directions")
parser.add_argument("--save", type=bool, default=True, help="Save animation?")

args = parser.parse_args()

WIDTH = args.width
HEIGHT = args.height
PEOPLE = args.people
TIMESTEPS = args.steps
P_CHANGE_DIR = args.p_change
save = args.save

# horizontal or vertical
directions = [(1,0), (-1,0), (0,1), (0,-1)]

obstacles = set()
# No obstacles for now 

class Person:
    """Agent that moves in a randomly chosen fixed direction"""
    def __init__(self, pid, x, y):
        self.id = pid
        self.x = x
        self.y = y
        self.dx, self.dy = random.choice(directions)
        self.jam_time = 0

# Spawn people (allowing overlap initially)
people = []
occupied = set()
for i in range(PEOPLE):
    while True:
        x = random.randrange(WIDTH)
        y = random.randrange(HEIGHT)

        # reject if obstacle or already occupied
        if (x, y) in obstacles or (x, y) in occupied:
            continue

        people.append(Person(i, x, y))
        occupied.add((x, y))
        break

def move(person, blocked, p):
    """
    Executes a persistent random walk step.
    
    Args:
        person: The agent attempting movement.
        blocked: Set of currently occupied coordinates.
        p: Probability of changing direction (turning).
    """
    if random.random() < p:
        person.dx, person.dy = random.choice(directions)

    nx = person.x + person.dx
    ny = person.y + person.dy

    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT and (nx, ny) not in blocked:
        person.x = nx
        person.y = ny
        return True
    else:
        person.jam_time += 1
        return False

            
def get_blocked(people, obstacles):
    """Returns a set of all impassable coordinates."""
    return {(p.x, p.y) for p in people} | obstacles


def find_clusters(people):
    """
    Performs a cluster analysis using Depth-First Search.
    A cluster is defined as a group of agents in adjacent cells.
    Used for calculating the Power Law distribution of crowd sizes.
    """
    occupied = {(p.x, p.y) for p in people}
    visited = set()
    clusters = []

    for cell in occupied:
        if cell in visited:
            continue

        stack = [cell]
        cluster = set()
        visited.add(cell)

        while stack:
            x, y = stack.pop()
            cluster.add((x, y))

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nb = (x + dx, y + dy)
                if nb in occupied and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        clusters.append(cluster)

    return clusters


# Setup figure
fig, ax = plt.subplots()
ax.set_xlim(-0.5, WIDTH-0.5)
ax.set_ylim(-0.5, HEIGHT-0.5)
xs = [float(p.x) for p in people]
ys = [float(p.y) for p in people]
colors = ["blue"] * PEOPLE
# colors[0] = "red"  # person 0 is getting tracked
scat = ax.scatter(xs, ys, c=colors, s=50)

cluster_counts = []
mean_cluster_sizes = []
all_cluster_sizes = []

            
def update(frame):
    """Updates the global state for each animation frame"""
    blocked = get_blocked(people, obstacles)

    for person in people:
        old_pos = (person.x, person.y)

        # Temporarily remove current position so person can move into it
        blocked.discard(old_pos)
        move(person, blocked, P_CHANGE_DIR)
        blocked.add((person.x, person.y))

    clusters = find_clusters(people)
    sizes = [len(c) for c in clusters]
    if frame > TIMESTEPS // 2:
        all_cluster_sizes.extend(sizes)

    if sizes:
        mean_cluster_size = sum(s*s for s in sizes) / sum(sizes)
    else:
        mean_cluster_size = 0


    mean_cluster_sizes.append(mean_cluster_size)

    # update scatter plot
    scat.set_offsets([(p.x, p.y) for p in people])
    ax.set_title(f"t = {frame}, p = {P_CHANGE_DIR}")
    return scat,

anim = FuncAnimation(fig, update, frames=TIMESTEPS, interval=100, blit=False)

# Save the animation or plot
if save:
    anim.save("model_random_probability/results/clusters.gif", writer='pillow', fps=10)
else: 
    plt.show()