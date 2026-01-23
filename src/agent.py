import product
from maze_solver_v2 import AgentPathfinder
from state import StateMap  # TODO: Implement the StateMap class
from typing import List, Tuple


class Agent:
    def __init__(
        self,
        name: str,
        position: tuple[int, int],
        shopping_list: list[product.Product],
        state_map: StateMap,
    ):
        self.name = name
        self.position = position  # position is a tuple (x, y)
        self._shopping_list = shopping_list
        self._state_map = state_map

        # You will only get the state. That encodes everything you need.
        self.pathfinder = AgentPathfinder(state_map)

        self._route = self._route_generator()

        self._pause_length = 0

    def move(self, new_position: tuple[int, int]) -> None:
        """Move the agent to a new position."""
        self.position = new_position

    def get_position(self) -> tuple[int, int]:
        """Return the current position of the agent."""
        return self.position

    def __str__(self) -> str:
        return f"Agent {self.name} at position {self.position}"

    def __repr__(self) -> str:
        return f"Agent {self.name} at position {self.position}"

    def _route_generator(self) -> List[Tuple[int, int]]:
        """Generate a route based on the shopping list."""
        return self.pathfinder.solve_path(self.position, self._shopping_list)

    def update(self):
        """Update the agent's state."""
        if self._pause_length > 0:
            self._pause_length -= 1
            return

        if self._route:
            next_position = self._route.pop(0)
            if not self._state_map.write_agent_map(
                next_position
            ):  # Return True if successful:
                # Recalculate route if movement was blocked
                self._route = self._route_generator()
            self.position = next_position

            # Product pickup detection
            if True:  # TODO: implement product pickup detection
                product = self._shopping_list.pop(0)
                self._pause_length = product.waiting_time
        else:
            self._route = self._route_generator()  # Recalculate route if needed
