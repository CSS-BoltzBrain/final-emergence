from statemap import StateMap
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import ArtistAnimation
from maze_solver_v2 import AgentPathfinder
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

from tqdm import tqdm


class SimulationOptimized:
    """Optimized simulation with better pathfinding strategy."""

    def __init__(
        self, filename: str, num_agents=5, use_threading=False
    ) -> None:
        self._state_map = StateMap(filename, scale_factor=1)
        self._agent_list = self._spawn_agents(num_agents)
        self.checkpoints = []

        # Use ThreadPoolExecutor for I/O-bound tasks, ProcessPoolExecutor for CPU-bound
        # For 20 steps, threading is more efficient due to lower overhead
        self.use_threading = use_threading
        if use_threading:
            self._path_executor = ThreadPoolExecutor(max_workers=4)
        else:
            # Still more efficient than Pool for small workloads
            self._path_executor = ProcessPoolExecutor(max_workers=2)

        # Cache for computed paths to avoid recalculation
        self._path_cache = {}

    def _spawn_agents(self, num_agents: int) -> list[Agent | None]:
        """Spawn a given number of agents in the simulation."""
        agent_list = [
            self._state_map.spawn_agent_start() for _ in range(num_agents)
        ]
        return agent_list

    def update(self):
        # Update existing agents
        for agent in self._agent_list:
            if not agent:
                continue
            if agent.update():
                self._agent_list.remove(agent)

        self._state_map.update_agent_map()

        # Spawn new agents
        self._agent_list += self._spawn_agents(5)
        self._agent_list = [
            agent for agent in self._agent_list if agent is not None
        ]

        # Find agents needing paths
        agents_needing_paths = [
            agent
            for agent in self._agent_list
            if agent and agent._route is None
        ]

        # Batch compute routes with futures
        if agents_needing_paths:
            futures = {}
            for agent in agents_needing_paths:
                shop, start, shopping_list, end = agent.request_route()
                # Create cache key from hashable elements
                cache_key = (start, tuple(p.name for p in shopping_list), end)

                # Check cache first
                if cache_key in self._path_cache:
                    agent._route = self._path_cache[cache_key]
                else:
                    # Submit to executor
                    future = self._path_executor.submit(
                        compute_route, (shop, start, shopping_list, end)
                    )
                    futures[future] = (agent, cache_key)

            # Collect results
            for future in futures:
                agent, cache_key = futures[future]
                try:
                    route = future.result(timeout=5)
                    agent._route = route
                    self._path_cache[cache_key] = route
                except Exception as e:
                    print(f"Error computing route: {e}")
                    agent._route = []

    def checkpoint(self):
        self.checkpoints.append(
            np.array(self._state_map.get_agent_map(), dtype=np.int8)
        )

    def cleanup(self):
        """Shutdown executor properly."""
        self._path_executor.shutdown(wait=True)


class SimulationSync:
    """Fully synchronous simulation - best for debugging & small workloads."""

    def __init__(self, filename: str, num_agents=5) -> None:
        self._state_map = StateMap(filename, scale_factor=1)
        self._agent_list = self._spawn_agents(num_agents)
        self.checkpoints = []

    def _spawn_agents(self, num_agents: int) -> list[Agent | None]:
        """Spawn a given number of agents in the simulation."""
        agent_list = [
            self._state_map.spawn_agent_start() for _ in range(num_agents)
        ]
        return agent_list

    def update(self):
        for agent in self._agent_list:
            if not agent:
                continue
            if agent.update():
                self._agent_list.remove(agent)

        self._state_map.update_agent_map()
        self._agent_list += self._spawn_agents(5)
        self._agent_list = [
            agent for agent in self._agent_list if agent is not None
        ]

        # Compute routes synchronously (no multiprocessing overhead)
        agents_needing_paths = [
            agent
            for agent in self._agent_list
            if agent and agent._route is None
        ]

        for agent in agents_needing_paths:
            shop, start, shopping_list, end = agent.request_route()
            pathfinder = AgentPathfinder(shop)
            route = pathfinder.solve_path(start, shopping_list, end)
            agent._route = route

    def checkpoint(self):
        self.checkpoints.append(
            np.array(self._state_map.get_agent_map(), dtype=np.int8)
        )


def compute_route(args):
    shop, start, shopping_list, end = args
    pathfinder = AgentPathfinder(shop)
    return pathfinder.solve_path(start, shopping_list, end)
