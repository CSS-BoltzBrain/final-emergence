import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# Parameters
WIDTH = 40
HEIGHT = 30
TIMESTEPS = 300
NUM_RUNS = 20

# Wall with two gaps
wall_x = 20
gap_width = 2  # number of cells in a gap
gap_top = HEIGHT // 3
gap_bottom = 2 * HEIGHT // 3
obstacles = set()
for y in range(HEIGHT):
    if not (gap_top - gap_width <= y <= gap_top + gap_width or
            gap_bottom - gap_width <= y <= gap_bottom + gap_width):
        obstacles.add((wall_x, y))

class Person:
    def __init__(self, pid, x, y):
        self.id = pid
        self.x = x
        self.y = y
        self.dx = 1
        self.dy = 0
        self.active = True
        self.jam_time = 0
        self.wait = False

def spawn_people(n_people):
    people = []
    occupied = set()
    for i in range(n_people):
        while True:
            x = random.randrange(wall_x)
            y = random.randrange(HEIGHT)
            if (x, y) not in obstacles and (x, y) not in occupied:
                p = Person(i, x, y)
                people.append(p)
                occupied.add((x, y))
                break
    return people

def forward_blocked(person, obstacles, occupied):
    nx, ny = person.x + person.dx, person.y + person.dy
    return (nx < 0 or nx >= WIDTH or ny < 0 or ny >= HEIGHT or
            (nx, ny) in obstacles or (nx, ny) in occupied)

def near_wall(person, wall_x, dist=7):
    return abs(person.x - (wall_x - 1)) <= dist

def choose_vertical_direction(person):
    dist_top = gap_top - person.y
    dist_bottom = gap_bottom - person.y
    if abs(dist_top) < abs(dist_bottom):
        dy = 1 if dist_top > 0 else -1
    else:
        dy = 1 if dist_bottom > 0 else -1
    person.dx, person.dy = 0, dy

def aligned_with_opening(person):
    in_top_gap = gap_top - gap_width <= person.y <= gap_top + gap_width
    in_bottom_gap = gap_bottom - gap_width <= person.y <= gap_bottom + gap_width
    return in_top_gap or in_bottom_gap

def free_space_right(person, obstacles, occupied):
    nx = person.x + 1
    ny = person.y
    return (
        0 <= nx < WIDTH and
        0 <= ny < HEIGHT and
        (nx, ny) not in obstacles and
        (nx, ny) not in occupied
    )

def move(person, obstacles, occupied):
    if not person.active:
        return
    if person.wait:
        person.wait = False
        person.jam_time += 1
        return

    if aligned_with_opening(person):
        person.dx, person.dy = 1, 0
    if free_space_right(person, obstacles, occupied):
        person.dx, person.dy = (1, 0)
    elif near_wall(person, wall_x) and forward_blocked(person, obstacles, occupied):
        choose_vertical_direction(person)
        person.wait = True

    nx, ny = person.x + person.dx, person.y + person.dy
    if (0 <= nx < WIDTH and 0 <= ny < HEIGHT and
        (nx, ny) not in obstacles and (nx, ny) not in occupied):
        occupied.discard((person.x, person.y))
        person.x, person.y = nx, ny
        occupied.add((nx, ny))
    else:
        person.jam_time += 1


def run_simulation(n_people):
    people = spawn_people(n_people)
    throughput = []
    for t in range(TIMESTEPS):
        occupied = {(p.x, p.y) for p in people if p.active}
        passed = 0
        for p in people:
            if p.active and p.x == wall_x and p.dx == 1:
                passed += 1
            move(p, obstacles, occupied)
        throughput.append(passed)
        if not any(p.active for p in people):
            break
    avg_throughput = sum(throughput[10:]) / max(1, len(throughput[10:]))
    return avg_throughput

# Analysis
n_people = list(range(20, 601, 5))
results = []
for n in n_people:
    sum_avg = 0
    for _ in range(NUM_RUNS):
        avg_run = run_simulation(n)
        sum_avg += avg_run
    avg = sum_avg/NUM_RUNS
    results.append(avg)
density = [n / (HEIGHT*wall_x) for n in n_people]
plt.plot(density, results, marker='o')
plt.xlabel("Density")
plt.ylabel("Average throughput per timestep")
plt.title("Fundamental diagram")
plt.show()
