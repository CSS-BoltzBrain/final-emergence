from statemap import StateMap
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import ArtistAnimation
from maze_solver_v2 import AgentPathfinder
from multiprocessing import Pool

import sys
import os

from tqdm import tqdm


def initworker(shop_map):
    global pathfinder
    pathfinder = AgentPathfinder(shop_map)


class Simulation:
    def __init__(self, filename: str, num_agents=5, adjust_probability=0.1) -> None:
        self._state_map = StateMap(
            filename, scale_factor=1, adjust_probability=adjust_probability
        )
        self._agent_list = self._spawn_agents(num_agents)
        self._num_agents = num_agents
        self.checkpoints = []

        # self._path_pool = Pool(
        #     initializer=initworker, initargs=(self._state_map.get_shop(),)
        # )

        self._state_map.update_agent_map()

    def _spawn_agents(self, num_agents: int) -> list[Agent | None]:
        """Spawn a given number of agents in the simulation."""
        agent_list = [self._state_map.spawn_agent_start() for _ in range(num_agents)]

        return agent_list

    def plot(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.subplots()
        self._state_map.get_shop()._plot_layout(ax)

        cmap = ListedColormap([(0, 0, 0, 0), "black"])
        ims = []
        for i in range(len(self.checkpoints)):
            im = ax.imshow(self.checkpoints[i], animated=True, cmap=cmap)
            if i == 0:
                im = ax.imshow(self.checkpoints[i], cmap=cmap)
            ims.append([im])

        ani = ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

        plt.tight_layout()
        plt.show()

    def update(self):
        assert np.all(self._state_map._passive_agent_map == 0)

        for agent in self._agent_list:
            if not agent:
                continue
            if agent.update():
                self._agent_list.remove(agent)
            else:
                assert agent.exists()

        self._agent_list += self._spawn_agents(self._num_agents)
        self._state_map.update_agent_map()

        self._agent_list = [agent for agent in self._agent_list if not None]

    def checkpoint(self):
        self.checkpoints += [np.array(self._state_map.get_agent_map(), dtype=np.int8)]

    def save_checkpoints(self, filename):
        arr = np.array(self.checkpoints, dtype=np.int8)
        np.save(filename, arr)

    def load_checkpoints(self, filename):
        arr = np.load(filename)
        self.checkpoints = arr


def compute_route(args):
    start, shopping_list, end = args
    return pathfinder.solve_path(start, shopping_list, end)


def simulate(args):
    scratch_disk, timesteps, prob = args
    simulation = Simulation("configs/surround.yaml", 96, adjust_probability=prob)

    tk0 = tqdm(range(timesteps), total=timesteps, disable=None)
    for i in tk0:
        simulation.update()
        if not i % 1:
            simulation.checkpoint()

    simulation.save_checkpoints(f"{scratch_disk}/simulation_{timesteps}_{prob}")

    simulation.plot()


if __name__ == "__main__":
    scratch_disk = sys.argv[1]
    timesteps = int(sys.argv[2])
    # for prob in [0.1]:
    probs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

    args = [(scratch_disk, timesteps, prob) for prob in probs]

    with Pool() as pool:
        results = pool.map_async(simulate, args)
        results.get()

