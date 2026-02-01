import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

parser = argparse.ArgumentParser(description="Crowd Simulation Parameters")
parser.add_argument("--width", type=int, default=30, help="Grid width")
parser.add_argument("--height", type=int, default=30, help="Grid height")
parser.add_argument("--people", type=int, default=300, help="Number of people")
parser.add_argument("--steps", type=int, default=400, help="Max timesteps")
parser.add_argument("--save", type=bool, default=False, help="Save animation?")

args = parser.parse_args()

WIDTH = args.width
HEIGHT = args.height
PEOPLE = args.people
TIMESTEPS = args.steps
save = args.save


directions = [(1,0), (-1,0), (0,1), (0,-1)]

# Obstacle is a wall with two gaps
obstacles = set()
wall_x = 20
gap_width = 0
gap_top = HEIGHT // 3
gap_bottom = 2 * HEIGHT // 3
for y in range(HEIGHT):
    if not (gap_top - gap_width <= y <= gap_top + gap_width or
            gap_bottom - gap_width <= y <= gap_bottom + gap_width):
        obstacles.add((wall_x, y))

class Person:
    """Represents an individual pedestrian agent in the grid."""
    def __init__(self, pid, x, y):
        self.id = pid
        self.x = x
        self.y = y
        self.dx = 1
        self.dy = 0 # initial direction is to the right
        self.active = True  # becomes False when leaving the screen
        self.jam_time = 0
        self.wait = False

people = []
occupied = set()

for i in range(PEOPLE):
    while True:
        # spawn to the left of the wall
        x = random.randrange(wall_x-2)
        y = random.randrange(HEIGHT)
        if (x, y) not in obstacles and (x, y) not in occupied:
            p = Person(i, x, y)
            people.append(p)
            occupied.add((x, y))
            break

def forward_blocked(person, obstacles, occupied):
    """Return true if the agent is blocked"""
    nx = person.x + person.dx
    ny = person.y + person.dy
    return (
        nx < 0 or nx >= WIDTH or
        ny < 0 or ny >= HEIGHT or
        (nx, ny) in obstacles or
        (nx, ny) in occupied
    )

def near_wall(person, wall_x, dist=7):
    return abs(person.x - (wall_x - 1)) <= dist


def choose_vertical_direction(person, gap_top, gap_bottom):
    """
    Choose a vertical direction that leads to the nearest gap
    """
    # Compute vertical distance to closest gap
    mid_top = gap_top
    mid_bottom = gap_bottom
    dist_top = mid_top - person.y
    dist_bottom = mid_bottom - person.y

    # Choose the closer gap
    if abs(dist_top) < abs(dist_bottom):
        dy = 1 if dist_top > 0 else -1
    else:
        dy = 1 if dist_bottom > 0 else -1

    person.dx, person.dy = (0, dy)

def aligned_with_opening(person, wall_x, gap_top, gap_bottom, x_tol=1):
    """
    Returns True if the agent is vertically aligned with a gap.
    """
    in_top_gap = gap_top - gap_width <= person.y <= gap_top + gap_width
    in_bottom_gap = gap_bottom - gap_width <= person.y <= gap_bottom + gap_width
    return (in_top_gap or in_bottom_gap)


def free_space_right(person, obstacles, occupied):
    """Checks if the cell immediately to the right is available"""
    nx = person.x + 1
    ny = person.y
    return (
        0 <= nx < WIDTH and
        0 <= ny < HEIGHT and
        (nx, ny) not in obstacles and
        (nx, ny) not in occupied
    )


def move(person, obstacles, occupied, wall_x):
    """
    Movement for agent: navigation and collision avoidance.
    
    The decision hierarchy follows:
    1. Goal Seeking: Attempt to move right if the path is clear or aligned with a gap.
    2. Obstacle Avoidance: If blocked by the wall, search for the nearest vertical exit.
    3. Conflict Resolution: If the target cell is occupied by another agent, wait (jam).
    """
    if not person.active:
        return
    
    # Exit system
    if person.x >= WIDTH - 1:
        person.active = False
        occupied.discard((person.x, person.y))
        return
    
    # Sit this one out if blocked last time
    if person.wait:
        person.wait = False
        person.jam_time += 1
        return
    
    # Decide direction
    if aligned_with_opening(person, wall_x, gap_top, gap_bottom):
        person.dx, person.dy = (1, 0)
    
    if free_space_right(person, obstacles, occupied):
        person.dx, person.dy = (1, 0)
        
    elif near_wall(person, wall_x) and forward_blocked(person, obstacles, occupied):
        choose_vertical_direction(person, gap_top, gap_bottom)
        person.wait = True

    # Attempt movement
    if not person.wait:
        nx = person.x + person.dx
        ny = person.y + person.dy

        if (0 <= nx < WIDTH and 0 <= ny < HEIGHT and
            (nx, ny) not in obstacles and
            (nx, ny) not in occupied):

            occupied.discard((person.x, person.y))
            person.x, person.y = nx, ny
            occupied.add((nx, ny))
        else:
            person.jam_time += 1
    else:
        person.jam_time += 1


# Plot setup
fig, ax = plt.subplots()
ax.set_xlim(-0.5, WIDTH-0.5)
ax.set_ylim(-0.5, HEIGHT-0.5)
ax.set_xticks([])
ax.set_yticks([])

ox, oy = zip(*obstacles)
ax.scatter(ox, oy, c="black", s=80, marker="s")

scat = ax.scatter([p.x for p in people],
                  [p.y for p in people],
                  c="blue", s=30)

# observables
total_jam = []
throughput = []

def update(frame):
    """Updates the global state for each animation frame"""
    occupied = {(p.x, p.y) for p in people if p.active}

    random.shuffle(people)

    for p in people:
        move(p, obstacles, occupied, wall_x)

    # Stop animation if everyone is gone
        if not any(p.active for p in people):
            # Only try to stop the timer if it actually exists (interactive mode)
            if ani.event_source is not None:
                ani.event_source.stop()
                
            ax.set_title("All agents exited")
            return scat,
    total_jam.append(sum(p.jam_time for p in people))
    passed = sum(
        1 for p in people
        if p.active and p.x == wall_x and p.dx == 1
    )
    throughput.append(passed)

    scat.set_offsets([(p.x, p.y) for p in people if p.active])
    ax.set_title(f"t = {frame}")

    return scat,


# Animation
ani = FuncAnimation(fig, update,
                    frames=TIMESTEPS,
                    interval=100,
                    blit=False)

if save:
    ani.save("model_random_probability/results/bottleneck.gif", writer='pillow', fps=10)
else:
    plt.show()