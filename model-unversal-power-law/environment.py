"""
Environment module for the TTC pedestrian simulation.

Defines the geometry in which agents move: boundary conditions, wrapping
rules, wall forces, and agent initialization.  Each environment is a
self-contained object consumed by TTCSimulation (in ttc_engine.py).

Supported environments:
  - TorusEnvironment:    flat 2D torus (periodic in x and y)
  - CorridorEnvironment: rectangular corridor (periodic x, walled y)

Environments can be instantiated directly or loaded from a YAML
configuration file via ``load_config(yaml_path)``.
"""

import random as rnd
from math import cos, sin, exp

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Environment:
    """
    Abstract base for simulation geometries.

    Subclasses must implement every method listed below.  TTCSimulation
    calls these methods to handle all geometry-dependent operations.
    """

    @property
    def num(self):
        """Total number of agents."""
        raise NotImplementedError

    def wrap_relative(self, d):
        """
        Wrap a relative displacement vector *d* (2D, modified in-place)
        so that it represents the shortest displacement in this geometry.
        Returns *d* for convenience.
        """
        raise NotImplementedError

    def apply_boundary(self, pos, rad):
        """
        Apply boundary conditions to an absolute position *pos* (2D,
        modified in-place).  *rad* is the agent collision radius, used
        by geometries with walls to keep agents inside.
        """
        raise NotImplementedError

    def wall_force(self, pos, rad):
        """
        Return a 2D wall-force vector at position *pos* for an agent
        with collision radius *rad*.  Returns ``[0, 0]`` for geometries
        without walls.
        """
        return np.array([0.0, 0.0])

    def init_agents(self, rad):
        """
        Create initial agent state arrays.

        Args:
            rad: collision radius (used for wall clearance)

        Returns:
            (pos, vel, gvel) — each a numpy array of shape (num, 2)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Torus (flat 2D periodic domain)
# ---------------------------------------------------------------------------

class TorusEnvironment(Environment):
    """
    Flat 2D torus of side length *s*.

    Both axes are periodic: agents exiting one edge re-enter from the
    opposite edge.  No walls.  Matches the default geometry used by
    KGS (ttcSimTK.py).
    """

    def __init__(self, s=4.0, num=18):
        self.s = s
        self._num = num

    @property
    def num(self):
        return self._num

    def wrap_relative(self, d):
        s = self.s
        if d[0] > s / 2.:
            d[0] -= s
        if d[1] > s / 2.:
            d[1] -= s
        if d[0] < -s / 2.:
            d[0] += s
        if d[1] < -s / 2.:
            d[1] += s
        return d

    def apply_boundary(self, pos, rad):
        s = self.s
        if pos[0] < 0:
            pos[0] = s
        if pos[1] < 0:
            pos[1] = s
        if pos[0] > s:
            pos[0] = 0
        if pos[1] > s:
            pos[1] = 0

    def init_agents(self, rad):
        """Random positions and velocities on the torus.

        Agent 0 is the "fast agent" with 2x goal speed (matching KGS).
        """
        num = self._num
        pos = np.zeros((num, 2))
        vel = np.zeros((num, 2))
        gvel = np.zeros((num, 2))

        for i in range(num):
            pos[i, 0] = rnd.uniform(0, self.s)
            pos[i, 1] = rnd.uniform(0, self.s)
            ang = rnd.uniform(0, 2 * 3.141592)
            vel[i, 0] = cos(ang)
            vel[i, 1] = sin(ang)
            gvel[i] = 1.5 * vel[i].copy()
            if i == 0:
                gvel[i] *= 2

        return pos, vel, gvel


# ---------------------------------------------------------------------------
# Corridor (periodic x, walled y)
# ---------------------------------------------------------------------------

class CorridorEnvironment(Environment):
    """
    Rectangular corridor of size *corridor_length* x *corridor_width*.

    The x-axis is periodic (agents wrap around the corridor ends).
    The y-axis has solid walls at y=0 and y=corridor_width, enforced
    by an exponential repulsive force and position clamping.

    Two groups of agents walk in opposite directions:
      - Group 0 (indices 0..num_per_group-1): left-to-right (+x)
      - Group 1 (indices num_per_group..2*num_per_group-1): right-to-left (-x)

    The ``group`` attribute (int array of length *num*) stores group
    membership for use by visualization and analysis code.
    """

    def __init__(self, corridor_length=12.0, corridor_width=3.0,
                 num_per_group=10, wall_A=5.0, wall_B=0.3,
                 walking_speed=1.3):
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.num_per_group = num_per_group
        self.wall_A = wall_A
        self.wall_B = wall_B
        self.walking_speed = walking_speed

        self.group = np.zeros(2 * num_per_group, dtype=int)
        self.group[num_per_group:] = 1

    @property
    def num(self):
        return 2 * self.num_per_group

    def wrap_relative(self, d):
        L = self.corridor_length
        if d[0] > L / 2.:
            d[0] -= L
        elif d[0] < -L / 2.:
            d[0] += L
        # y is not wrapped (solid walls)
        return d

    def apply_boundary(self, pos, rad):
        L = self.corridor_length
        # Periodic x
        if pos[0] < 0:
            pos[0] += L
        elif pos[0] > L:
            pos[0] -= L
        # Clamped y (keep agent inside corridor)
        if pos[1] < rad:
            pos[1] = rad
        elif pos[1] > self.corridor_width - rad:
            pos[1] = self.corridor_width - rad

    def wall_force(self, pos, rad):
        """
        Exponential repulsive wall force in the y-direction.

        Force is  F_y = A * exp(-d / B)  directed away from the wall,
        where d is the distance from the agent edge to the wall surface.
        """
        A = self.wall_A
        B = self.wall_B
        y = pos[1]

        fy = 0.0
        # Bottom wall (y = 0): pushes upward (+y)
        d_bottom = max(y - rad, 0.01)
        fy += A * exp(-d_bottom / B)
        # Top wall (y = corridor_width): pushes downward (-y)
        d_top = max(self.corridor_width - y - rad, 0.01)
        fy -= A * exp(-d_top / B)

        return np.array([0.0, fy])

    def init_agents(self, rad):
        """Place Group 0 near the left end, Group 1 near the right end."""
        num = self.num
        pos = np.zeros((num, 2))
        vel = np.zeros((num, 2))
        gvel = np.zeros((num, 2))

        margin = self.corridor_length * 0.15
        for i in range(num):
            if self.group[i] == 0:
                pos[i, 0] = rnd.uniform(0, margin)
                pos[i, 1] = rnd.uniform(rad * 2, self.corridor_width - rad * 2)
                vel[i] = np.array([self.walking_speed, 0.0])
                gvel[i] = np.array([self.walking_speed, 0.0])
            else:
                pos[i, 0] = rnd.uniform(
                    self.corridor_length - margin, self.corridor_length)
                pos[i, 1] = rnd.uniform(rad * 2, self.corridor_width - rad * 2)
                vel[i] = np.array([-self.walking_speed, 0.0])
                gvel[i] = np.array([-self.walking_speed, 0.0])

        return pos, vel, gvel


# ---------------------------------------------------------------------------
# Narrow Door Corridor (periodic x, walled y, wall with door in middle)
# ---------------------------------------------------------------------------

class NarrowDoorEnvironment(Environment):
    """
    Corridor with a wall in the middle containing a narrow door (gap).

    The x-axis is periodic (agents wrap around the corridor ends).
    The y-axis has solid walls at y=0 and y=corridor_width.
    A vertical wall is placed at x=corridor_length/2 with a small gap
    (door) that only allows one person to pass at a time.

    Two groups of agents walk in opposite directions:
      - Group 0 (indices 0..num_per_group-1): left-to-right (+x)
      - Group 1 (indices num_per_group..2*num_per_group-1): right-to-left (-x)

    Wall structure at x = corridor_length/2:
      - Upper wall segment: from y=door_top to y=corridor_width
      - Lower wall segment: from y=0 to y=door_bottom
      - Door gap: from y=door_bottom to y=door_top (default ~0.6m)
    """

    def __init__(self, corridor_length=12.0, corridor_width=3.0,
                 num_per_group=10, wall_A=5.0, wall_B=0.3,
                 walking_speed=1.3, door_width=0.6, door_center=None):
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.num_per_group = num_per_group
        self.wall_A = wall_A
        self.wall_B = wall_B
        self.walking_speed = walking_speed

        # Door parameters
        self.door_width = door_width
        if door_center is None:
            door_center = corridor_width / 2.0
        self.door_center = door_center
        self.door_bottom = door_center - door_width / 2.0
        self.door_top = door_center + door_width / 2.0

        # Middle wall position
        self.wall_x = corridor_length / 2.0

        self.group = np.zeros(2 * num_per_group, dtype=int)
        self.group[num_per_group:] = 1

    @property
    def num(self):
        return 2 * self.num_per_group

    def wrap_relative(self, d):
        L = self.corridor_length
        if d[0] > L / 2.:
            d[0] -= L
        elif d[0] < -L / 2.:
            d[0] += L
        # y is not wrapped (solid walls)
        return d

    def apply_boundary(self, pos, rad):
        L = self.corridor_length
        # Periodic x
        if pos[0] < 0:
            pos[0] += L
        elif pos[0] > L:
            pos[0] -= L
        # Clamped y (keep agent inside corridor)
        if pos[1] < rad:
            pos[1] = rad
        elif pos[1] > self.corridor_width - rad:
            pos[1] = self.corridor_width - rad

        # --- Hard boundary for middle wall ---
        # Check if agent is trying to cross the wall while not in the door gap
        in_door = (pos[1] - rad >= self.door_bottom) and (pos[1] + rad <= self.door_top)

        if not in_door:
            # Agent cannot pass through wall - clamp x position
            # If agent center is within rad of wall_x, push them back
            if pos[0] > self.wall_x - rad and pos[0] < self.wall_x + rad:
                # Determine which side the agent was on (use velocity or position)
                # Push to the nearest side
                if pos[0] < self.wall_x:
                    pos[0] = self.wall_x - rad
                else:
                    pos[0] = self.wall_x + rad

    def wall_force(self, pos, rad):
        """
        Exponential repulsive wall force from corridor walls and middle wall.

        Force components:
        1. Top/bottom corridor walls (y-direction)
        2. Middle wall with door gap (x-direction, only if agent is not
           aligned with the door opening)
        """
        A = self.wall_A
        B = self.wall_B
        x = pos[0]
        y = pos[1]

        fx = 0.0
        fy = 0.0

        # --- Top/bottom corridor walls (y-direction) ---
        # Bottom wall (y = 0): pushes upward (+y)
        d_bottom = max(y - rad, 0.01)
        fy += A * exp(-d_bottom / B)
        # Top wall (y = corridor_width): pushes downward (-y)
        d_top = max(self.corridor_width - y - rad, 0.01)
        fy -= A * exp(-d_top / B)

        # --- Middle wall with door (x-direction) ---
        # The wall is at x = wall_x. Check if agent is near the wall
        # and NOT aligned with the door gap.
        wall_influence_range = 3.0 * B  # range where wall force is significant

        dx_to_wall = abs(x - self.wall_x)
        if dx_to_wall < wall_influence_range + rad:
            # Check if agent y-position is blocked by wall (not in door gap)
            in_door = (y - rad >= self.door_bottom) and (y + rad <= self.door_top)

            if not in_door:
                # Agent is facing a wall segment
                d_wall = max(dx_to_wall - rad, 0.01)

                # Force pushes away from wall
                wall_force_mag = A * exp(-d_wall / B)

                if x < self.wall_x:
                    fx -= wall_force_mag  # push left (-x)
                else:
                    fx += wall_force_mag  # push right (+x)

                # If agent is near door edge, add y-force to guide toward door
                if y < self.door_bottom:
                    # Below door: guide upward toward door
                    d_to_door_edge = self.door_bottom - y - rad
                    if d_to_door_edge > 0 and d_to_door_edge < wall_influence_range:
                        fy += A * 0.5 * exp(-d_to_door_edge / B)
                elif y > self.door_top:
                    # Above door: guide downward toward door
                    d_to_door_edge = y - rad - self.door_top
                    if d_to_door_edge > 0 and d_to_door_edge < wall_influence_range:
                        fy -= A * 0.5 * exp(-d_to_door_edge / B)

        return np.array([fx, fy])

    def init_agents(self, rad):
        """Place Group 0 on left side, Group 1 on right side of the wall."""
        num = self.num
        pos = np.zeros((num, 2))
        vel = np.zeros((num, 2))
        gvel = np.zeros((num, 2))

        left_margin = self.wall_x * 0.8  # spawn zone on left side
        right_start = self.wall_x + (self.corridor_length - self.wall_x) * 0.2

        for i in range(num):
            if self.group[i] == 0:
                # Group 0: left side, moving right
                pos[i, 0] = rnd.uniform(rad, left_margin)
                pos[i, 1] = rnd.uniform(rad * 2, self.corridor_width - rad * 2)
                vel[i] = np.array([self.walking_speed, 0.0])
                gvel[i] = np.array([self.walking_speed, 0.0])
            else:
                # Group 1: right side, moving left
                pos[i, 0] = rnd.uniform(right_start, self.corridor_length - rad)
                pos[i, 1] = rnd.uniform(rad * 2, self.corridor_width - rad * 2)
                vel[i] = np.array([-self.walking_speed, 0.0])
                gvel[i] = np.array([-self.walking_speed, 0.0])

        return pos, vel, gvel


# ---------------------------------------------------------------------------
# Narrow Door Corridor - One Group (periodic x, walled y, wall with door)
# ---------------------------------------------------------------------------

class NarrowDoorOneGroupEnvironment(Environment):
    """
    Corridor with a wall in the middle containing a narrow door (gap).
    Single group variant: all agents walk in the same direction.

    The x-axis is periodic (agents wrap around the corridor ends).
    The y-axis has solid walls at y=0 and y=corridor_width.
    A vertical wall is placed at x=corridor_length/2 with a small gap
    (door) that only allows one person to pass at a time.

    All agents belong to a single group walking left-to-right (+x).
    They start on the left side, pass through the door, exit on the right,
    and wrap back to the left side (periodic boundary).

    Wall structure at x = corridor_length/2:
      - Upper wall segment: from y=door_top to y=corridor_width
      - Lower wall segment: from y=0 to y=door_bottom
      - Door gap: from y=door_bottom to y=door_top (default ~0.6m)
    """

    def __init__(self, corridor_length=12.0, corridor_width=3.0,
                 num_agents=20, wall_A=5.0, wall_B=0.3,
                 walking_speed=1.3, door_width=0.6, door_center=None):
        self.corridor_length = corridor_length
        self.corridor_width = corridor_width
        self.num_agents = num_agents
        self.wall_A = wall_A
        self.wall_B = wall_B
        self.walking_speed = walking_speed

        # Door parameters
        self.door_width = door_width
        if door_center is None:
            door_center = corridor_width / 2.0
        self.door_center = door_center
        self.door_bottom = door_center - door_width / 2.0
        self.door_top = door_center + door_width / 2.0

        # Middle wall position
        self.wall_x = corridor_length / 2.0

        # Single group: all agents have group=0
        self.group = np.zeros(num_agents, dtype=int)

    @property
    def num(self):
        return self.num_agents

    def wrap_relative(self, d):
        L = self.corridor_length
        if d[0] > L / 2.:
            d[0] -= L
        elif d[0] < -L / 2.:
            d[0] += L
        # y is not wrapped (solid walls)
        return d

    def apply_boundary(self, pos, rad):
        L = self.corridor_length
        # Periodic x
        if pos[0] < 0:
            pos[0] += L
        elif pos[0] > L:
            pos[0] -= L
        # Clamped y (keep agent inside corridor)
        if pos[1] < rad:
            pos[1] = rad
        elif pos[1] > self.corridor_width - rad:
            pos[1] = self.corridor_width - rad

        # --- Hard boundary for middle wall ---
        # Check if agent is trying to cross the wall while not in the door gap
        in_door = (pos[1] - rad >= self.door_bottom) and (pos[1] + rad <= self.door_top)

        if not in_door:
            # Agent cannot pass through wall - clamp x position
            # If agent center is within rad of wall_x, push them back
            if pos[0] > self.wall_x - rad and pos[0] < self.wall_x + rad:
                # Determine which side the agent was on (use velocity or position)
                # Push to the nearest side
                if pos[0] < self.wall_x:
                    pos[0] = self.wall_x - rad
                else:
                    pos[0] = self.wall_x + rad

    def wall_force(self, pos, rad):
        """
        Exponential repulsive wall force from corridor walls and middle wall.

        Force components:
        1. Top/bottom corridor walls (y-direction)
        2. Middle wall with door gap (x-direction, only if agent is not
           aligned with the door opening)
        """
        A = self.wall_A
        B = self.wall_B
        x = pos[0]
        y = pos[1]

        fx = 0.0
        fy = 0.0

        # --- Top/bottom corridor walls (y-direction) ---
        # Bottom wall (y = 0): pushes upward (+y)
        d_bottom = max(y - rad, 0.01)
        fy += A * exp(-d_bottom / B)
        # Top wall (y = corridor_width): pushes downward (-y)
        d_top = max(self.corridor_width - y - rad, 0.01)
        fy -= A * exp(-d_top / B)

        # --- Middle wall with door (x-direction) ---
        # The wall is at x = wall_x. Check if agent is near the wall
        # and NOT aligned with the door gap.
        wall_influence_range = 3.0 * B  # range where wall force is significant

        dx_to_wall = abs(x - self.wall_x)
        if dx_to_wall < wall_influence_range + rad:
            # Check if agent y-position is blocked by wall (not in door gap)
            in_door = (y - rad >= self.door_bottom) and (y + rad <= self.door_top)

            if not in_door:
                # Agent is facing a wall segment
                d_wall = max(dx_to_wall - rad, 0.01)

                # Force pushes away from wall
                wall_force_mag = A * exp(-d_wall / B)

                if x < self.wall_x:
                    fx -= wall_force_mag  # push left (-x)
                else:
                    fx += wall_force_mag  # push right (+x)

                # If agent is near door edge, add y-force to guide toward door
                if y < self.door_bottom:
                    # Below door: guide upward toward door
                    d_to_door_edge = self.door_bottom - y - rad
                    if d_to_door_edge > 0 and d_to_door_edge < wall_influence_range:
                        fy += A * 0.5 * exp(-d_to_door_edge / B)
                elif y > self.door_top:
                    # Above door: guide downward toward door
                    d_to_door_edge = y - rad - self.door_top
                    if d_to_door_edge > 0 and d_to_door_edge < wall_influence_range:
                        fy -= A * 0.5 * exp(-d_to_door_edge / B)

        return np.array([fx, fy])

    def init_agents(self, rad):
        """Place all agents on left side, all moving right toward the door."""
        num = self.num
        pos = np.zeros((num, 2))
        vel = np.zeros((num, 2))
        gvel = np.zeros((num, 2))

        # Spawn zone: left side of the wall
        spawn_margin = self.wall_x * 0.8

        for i in range(num):
            # All agents start on left side, moving right
            pos[i, 0] = rnd.uniform(rad, spawn_margin)
            pos[i, 1] = rnd.uniform(rad * 2, self.corridor_width - rad * 2)
            vel[i] = np.array([self.walking_speed, 0.0])
            gvel[i] = np.array([self.walking_speed, 0.0])

        return pos, vel, gvel


# ---------------------------------------------------------------------------
# YAML configuration loader
# ---------------------------------------------------------------------------

def load_config(yaml_path, env_size=None, sight=None):
    """
    Load an environment from a YAML configuration file.

    The YAML must contain an ``environment`` section with a ``type`` key
    (``torus`` or ``corridor``) plus the corresponding parameters.
    Optionally includes ``agents`` and ``physics`` sections.

    Args:
        yaml_path: path to YAML config file
        env_size: override environment size (torus side length)
        sight: override sight range for neighbor detection

    Returns:
        (env, physics_params) where *env* is an Environment instance
        and *physics_params* is a dict of physics keyword arguments
        suitable for passing to ``TTCSimulation(env=env, **physics_params)``.
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg.get('environment', {})
    agents_cfg = cfg.get('agents', {})
    physics_cfg = cfg.get('physics', {})

    # Apply command-line overrides
    if env_size is not None:
        env_cfg['size'] = env_size
    if sight is not None:
        physics_cfg['sight'] = sight

    env_type = env_cfg.get('type', 'torus')

    if env_type == 'torus':
        env = TorusEnvironment(
            s=env_cfg.get('size', 4.0),
            num=agents_cfg.get('num', 18),
        )
    elif env_type == 'corridor':
        env = CorridorEnvironment(
            corridor_length=env_cfg.get('corridor_length', 12.0),
            corridor_width=env_cfg.get('corridor_width', 3.0),
            num_per_group=agents_cfg.get('num_per_group', 10),
            wall_A=env_cfg.get('wall_A', 5.0),
            wall_B=env_cfg.get('wall_B', 0.3),
            walking_speed=agents_cfg.get('walking_speed', 1.3),
        )
    elif env_type == 'narrowdoor':
        env = NarrowDoorEnvironment(
            corridor_length=env_cfg.get('corridor_length', 12.0),
            corridor_width=env_cfg.get('corridor_width', 3.0),
            num_per_group=agents_cfg.get('num_per_group', 10),
            wall_A=env_cfg.get('wall_A', 5.0),
            wall_B=env_cfg.get('wall_B', 0.3),
            walking_speed=agents_cfg.get('walking_speed', 1.3),
            door_width=env_cfg.get('door_width', 0.6),
            door_center=env_cfg.get('door_center', None),
        )
    elif env_type == 'narrowdoor_onegroup':
        env = NarrowDoorOneGroupEnvironment(
            corridor_length=env_cfg.get('corridor_length', 12.0),
            corridor_width=env_cfg.get('corridor_width', 3.0),
            num_agents=agents_cfg.get('num_agents', 20),
            wall_A=env_cfg.get('wall_A', 5.0),
            wall_B=env_cfg.get('wall_B', 0.3),
            walking_speed=agents_cfg.get('walking_speed', 1.3),
            door_width=env_cfg.get('door_width', 0.6),
            door_center=env_cfg.get('door_center', None),
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type!r}. "
                         f"Supported: 'torus', 'corridor', 'narrowdoor', "
                         f"'narrowdoor_onegroup'.")

    return env, physics_cfg
