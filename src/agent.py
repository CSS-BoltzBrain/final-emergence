from product import Product
from typing import List, Tuple


class Agent:
    def __init__(
        self,
        name: str,
        position: tuple[int, int],
        end_position: tuple[int, int],
        shopping_list: list[Product],
        state_map,
    ) -> None:
        self.name = name
        self.position = position  # position is a tuple (x, y)
        self._shopping_list = shopping_list
        self._state_map = state_map
        self._end_position = end_position

        self._route = None

        self._pause_length = 0

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
        # Pausing mechanism for product pickup
        if self._pause_length > 0:
            self._pause_length -= 1
            return False

        if self._route:
            next_position = self._route.pop(0)
            if not self._state_map.write_agent_map(
                next_position
            ):  # Return True if successful:
                # Recalculate route if movement was blocked
                self._route = self._route_generator()
                return False  # Skip movement this turn
            self.move(next_position)

            # Product pickup detection
            # if self._pickup_product():  # TODO: implement product pickup detection
            #     product = self._shopping_list.pop(0)
            #     self._pause_length = product.waiting_time
        else:
            self._route = (
                self._route_generator()
            )  # Recalculate route if needed

        return self.position == self._end_position
