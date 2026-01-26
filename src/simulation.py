from statemap import StateMap
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import ArtistAnimation
from maze_solver_v2 import AgentPathfinder
from multiprocessing import Pool

from tqdm import tqdm


def initworker(shop_map):
    global pathfinder
    pathfinder = AgentPathfinder(shop_map)


class Simulation:
    def __init__(self, filename: str, num_agents=5) -> None:
        self._state_map = StateMap(filename, scale_factor=1)
        self._agent_list = self._spawn_agents(num_agents)
        self._num_agents = num_agents
        self.checkpoints = []

        self._path_pool = Pool(
            initializer=initworker, initargs=(self._state_map.get_shop(),)
        )

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
        for agent in self._agent_list:
            if not agent:
                continue
            if agent.update():
                self._agent_list.remove(agent)

        self._state_map.update_agent_map()

        self._agent_list += self._spawn_agents(self._num_agents)

        self._agent_list = [agent for agent in self._agent_list if not None]

        agents_needing_paths = [
            agent for agent in self._agent_list if agent and agent._route is None
        ]

        tasks = [agent.request_route() for agent in agents_needing_paths]

        if tasks:
            routes = self._path_pool.map(compute_route, tasks, len(tasks) // 16)

            for agent, route in zip(agents_needing_paths, routes):
                agent._route = route

    def checkpoint(self):
        self.checkpoints += [np.array(self._state_map.get_agent_map(), dtype=np.int8)]


def compute_route(args):
    start, shopping_list, end = args
    return pathfinder.solve_path(start, shopping_list, end)


if __name__ == "__main__":
    simulation = Simulation("configs/empty.yaml", 50)
    shop_map = simulation._state_map.get_shop()
    pathfinder = AgentPathfinder(shop_map)
    agent_map = simulation._state_map.get_agent_map()
    print(shop_map.layout_array)
    print(shop_map.products_list)
    print(agent_map)

    tk0 = tqdm(range(1000), total=1000)
    for i in tk0:
        simulation.update()
        if not i % 10:
            simulation.checkpoint()

    simulation.plot()
