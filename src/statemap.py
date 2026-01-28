import numpy as np
from shopmap import ShopMap
from agent import Agent
from product import Product


class StateMap:
    def __init__(
        self, filename: str, scale_factor: int, adjust_probability: float
    ) -> None:
        self._shop_map = ShopMap(filename)
        self._scale_factor = scale_factor
        self._shop_size = self._shop_map.layout_array.shape
        self._active_agent_map = np.zeros(
            (
                self._shop_size[0] * scale_factor,
                self._shop_size[1] * scale_factor,
            ),
            dtype=np.int8,
        )
        self._passive_agent_map = np.zeros_like(self._active_agent_map)

        self._adjust_probability = adjust_probability

        self.entrances = np.argwhere(self._shop_map.layout_array == "I")
        self.exits = np.argwhere(self._shop_map.layout_array == "E")

    def get_shop(self) -> ShopMap:
        return self._shop_map

    def get_agent_map(self) -> np.ndarray:
        return self._active_agent_map

    def write_agent_map(self, position: tuple[int, int]) -> bool:
        x, y = position
        if self._passive_agent_map[y, x] == 1:
            return False  # Position already occupied
        self._passive_agent_map[y, x] = 1
        return True

    def update_agent_map(self) -> None:
        self._active_agent_map[:] = self._passive_agent_map
        self._passive_agent_map.fill(0)

    def spawn_agent(self, position: tuple[int, int]) -> bool:
        x, y = position
        if self._active_agent_map[y, x] == 1:
            return False  # Position already occupied
        self._active_agent_map[y, x] = 1
        return True

    def create_shopping_list(self) -> list[Product]:
        return self._shop_map.generate_shopping_list()

    def spawn_agent_start(self) -> Agent | None:
        np.random.shuffle(self.entrances)

        for entrance in self.entrances:
            y, x = entrance
            y_mask = self.exits[:, 0] != y
            x_mask = self.exits[:, 1] != x

            mask = y_mask & x_mask

            es = self.exits[mask]
            ey, ex = es[np.random.randint(0, len(es))]

            if self._active_agent_map[y, x] == 0:
                self._active_agent_map[y, x] = 1
                return Agent(
                    name="Test agent",
                    position=(x, y),
                    end_position=(ex, ey),
                    shopping_list=self.create_shopping_list(),
                    state_map=self,
                    adjust_probability=self._adjust_probability,
                )

        return None

    def available_spot(self, position: tuple[int, int]) -> bool:
        x, y = position
        shop_x, shop_y = np.array((x, y), dtype=np.int64) // self._scale_factor
        if not self.get_shop().walkable(shop_x, shop_y):
            return False
        if self.get_agent_map()[y, x] == 1:
            return False
        return True
