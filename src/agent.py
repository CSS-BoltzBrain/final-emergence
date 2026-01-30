from product import Product
from typing import List, Tuple
import numpy as np


class Agent:
    def __init__(
        self,
        name: str,
        position: tuple[int, int],
        end_position: tuple[int, int],
        shopping_list: list[Product],
        state_map,
        adjust_probability: float = 0.1,
        init_dir: tuple[int, int] = None,
    ) -> None:
        self.name = name
        self.position = position  # position is a tuple (x, y)
        self._shopping_list = shopping_list
        self._state_map = state_map
        self._end_position = end_position

        self._route = None

        self._pause_length = 0
        self._adjust_probability = adjust_probability

        self._dir = init_dir
        self._dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def move(self, new_position: tuple[int, int]) -> None:
        """Move the agent to a new position."""
        self.position = new_position

    def get_position(self) -> tuple[int, int]:
        """Return the current position of the agent."""
        return self.position

    def __str__(self) -> str:
        return f"Agent {self.name} at position {self.position} going to {self._end_position}"

    def __repr__(self) -> str:
        return f"Agent {self.name} at position {self.position} going to {self._end_position}"

    def _route_generator(self) -> List[Tuple[int, int]]:
        """Generate a route based on the shopping list."""
        return None

    def request_route(self):
        return (
            self.position,
            self._shopping_list,
            self._end_position,
        )

    def _pickup_product(self) -> bool:
        """Simulate picking up a product."""
        # Placeholder for actual pickup logic
        shop_map = self._state_map.get_shop()
        for p in self._shopping_list:
            key = shop_map.product_dict[p.name]
            if shop_map[self.position[::-1]] == key:
                return True

        return False

    def update(self) -> bool:
        """Update the agent's state."""
        x, y = self.position

        assert self._state_map._passive_agent_map[y, x] == 0

        self._route = (0, 0)
        if not self._dir:
            dir = np.random.randint(0, 4)
            self._dir = self._dirs[dir]

        if np.random.rand() < self._adjust_probability:
            prev_dir = self._dir
            dir = np.random.randint(0, 4)
            self._dir = self._dirs[dir]

            while prev_dir == self._dir:
                dir = np.random.randint(0, 4)
                self._dir = self._dirs[dir]

        x, y = self.position
        dx, dy = self._dir
        nx, ny = x + dx, y + dy

        if self._state_map.available_spot((nx, ny)):
            self.move((nx, ny))

        self._state_map.write_agent_map(self.position)

        return self.position == self._end_position

    def exists(self):
        x, y = self.position
        return self._state_map._passive_agent_map[y, x] == 1
