"""
Jam duration tracking for TTC pedestrian simulations.

Tracks when agents enter and exit "jam" states (unable to make progress)
and records the duration of each jam event. Used to investigate whether
traffic jams exhibit self-organized criticality via power-law distributions.

A jam state is defined as when an agent's displacement over a time window
falls below a threshold, indicating the agent is stuck or moving very slowly.
"""

import numpy as np
from collections import deque


class JamTracker:
    """
    Track jam states and durations for all agents in a simulation.

    A jam event starts when an agent's displacement over dt_window falls
    below threshold, and ends when displacement exceeds threshold (or the
    agent is removed from simulation).

    Usage:
        tracker = JamTracker(num_agents=20, dt=0.02)
        for step in simulation:
            sim.step()
            tracker.update(sim.pos, sim.active)
        tracker.finalize()
        durations = tracker.get_completed_jam_events()
    """

    def __init__(self, num_agents, dt, dt_window=1.0, threshold=0.4):
        """
        Initialize the jam tracker.

        Args:
            num_agents: Number of agents in simulation
            dt: Simulation timestep (seconds)
            dt_window: Time window for measuring displacement (seconds).
                       Default 1.0s means we check if agent moved threshold
                       distance in the last 1 second.
            dt_window: Time window for displacement check (seconds)
            threshold: Minimum displacement to not be jammed (meters).
                       Default 0.4m = 2 * agent radius (0.2m).
        """
        self.num_agents = num_agents
        self.dt = dt
        self.dt_window = dt_window
        self.threshold = threshold

        # Number of timesteps to look back for displacement calculation
        self.window_steps = max(1, int(dt_window / dt))

        # Position history buffer for each agent (circular buffer)
        # Stores last window_steps positions
        self.position_history = [deque(maxlen=self.window_steps + 1)
                                 for _ in range(num_agents)]

        # Current jam state for each agent (True = jammed)
        self.is_jammed = np.zeros(num_agents, dtype=bool)

        # Timestep when current jam started (for agents currently jammed)
        self.jam_start_step = np.zeros(num_agents, dtype=int)

        # List of completed jam durations (in seconds)
        self.completed_jams = []

        # Current simulation timestep
        self.current_step = 0

        # Track which agents were active last step (to detect removal)
        self.was_active = np.ones(num_agents, dtype=bool)

    def update(self, positions, active):
        """
        Update jam states based on current positions.

        Args:
            positions: np.array of shape (num_agents, 2) - current positions
            active: np.array of bools - which agents are still active

        Called once per simulation timestep.
        """
        self.current_step += 1

        for i in range(self.num_agents):
            # Check if agent was just removed
            if self.was_active[i] and not active[i]:
                # Agent was removed - end any ongoing jam
                if self.is_jammed[i]:
                    duration = (self.current_step - self.jam_start_step[i]) * self.dt
                    self.completed_jams.append(duration)
                    self.is_jammed[i] = False
                self.was_active[i] = False
                continue

            if not active[i]:
                continue

            # Store current position in history
            self.position_history[i].append(positions[i].copy())

            # Need enough history to calculate displacement
            if len(self.position_history[i]) < self.window_steps + 1:
                continue

            # Calculate displacement over the time window
            current_pos = self.position_history[i][-1]
            old_pos = self.position_history[i][0]
            displacement = np.linalg.norm(current_pos - old_pos)

            # Determine if agent is currently jammed
            currently_jammed = displacement < self.threshold

            # State transitions
            if currently_jammed and not self.is_jammed[i]:
                # Entering jam state
                self.is_jammed[i] = True
                self.jam_start_step[i] = self.current_step

            elif not currently_jammed and self.is_jammed[i]:
                # Exiting jam state - record completed jam
                duration = (self.current_step - self.jam_start_step[i]) * self.dt
                self.completed_jams.append(duration)
                self.is_jammed[i] = False

        self.was_active = active.copy()

    def get_completed_jam_events(self):
        """
        Return list of completed jam durations (in seconds).

        Returns:
            List of floats, each representing a jam duration in seconds.
        """
        return self.completed_jams.copy()

    def finalize(self):
        """
        End all ongoing jams (for agents still jammed at simulation end).

        Call this when simulation terminates to capture jams that were
        still in progress.
        """
        for i in range(self.num_agents):
            if self.is_jammed[i] and self.was_active[i]:
                duration = (self.current_step - self.jam_start_step[i]) * self.dt
                self.completed_jams.append(duration)
                self.is_jammed[i] = False

    def get_current_jam_count(self):
        """Return number of agents currently in jam state."""
        return np.sum(self.is_jammed & self.was_active)

    def get_statistics(self):
        """
        Return summary statistics about jam events.

        Returns:
            dict with keys: total_events, mean_duration, max_duration, etc.
        """
        durations = self.completed_jams
        if not durations:
            return {
                'total_events': 0,
                'mean_duration': 0.0,
                'max_duration': 0.0,
                'min_duration': 0.0,
                'total_jam_time': 0.0,
            }

        return {
            'total_events': len(durations),
            'mean_duration': np.mean(durations),
            'max_duration': np.max(durations),
            'min_duration': np.min(durations),
            'total_jam_time': np.sum(durations),
        }
