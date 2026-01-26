import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Dict, List
from product import Product
import random


class ShopMap:
    """
    A class representing the supermarket layout.
    """

    def __init__(self, filename: str) -> None:
        self.layout_array = self.load_layout_yaml(filename)
        self.products_list, self._products_by_code, self.products_by_category = (
            self._load_products(filename)
        )

    @property
    def product_dict(self) -> dict[str, Product]:
        return self._products_by_code

    def load_layout_yaml(self, filename: str) -> np.ndarray[str]:
        """
        Load a supermarket layout from a yaml file, return an np array of strings.
        """
        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        width: int = data["width"]
        height: int = data["height"]

        grid = np.full((height, width), "0", dtype="<U4")

        def fill_rectangle(x: int, y: int, w: int, h: int, value: str):
            for i in range(y, y + h):
                for j in range(x, x + w):
                    if 0 <= i < height and 0 <= j < width:
                        grid[i, j] = value

        # --- Walls ---
        for wall in data.get("walls", []):
            fill_rectangle(wall["x"], wall["y"], wall["width"], wall["height"], "#")

        # --- Entrances ---
        for entrance in data.get("entrance", []):
            x0, y0 = entrance["start"]
            x1, y1 = entrance["end"]
            if x0 == x1:
                for i in range(min(y0, y1), max(y0, y1) + 1):
                    grid[i, x0] = "I"
            elif y0 == y1:
                for j in range(min(x0, x1), max(x0, x1) + 1):
                    grid[y0, j] = "I"

        # --- Exits ---
        for exit_ in data.get("exit", []):
            x0, y0 = exit_["start"]
            x1, y1 = exit_["end"]
            if x0 == x1:
                for i in range(min(y0, y1), max(y0, y1) + 1):
                    grid[i, x0] = "E"
            elif y0 == y1:
                for j in range(min(x0, x1), max(x0, x1) + 1):
                    grid[y0, j] = "E"

        # --- Products ---
        categories: Dict[str, List[dict]] = data.get("categories", {})

        # --- Shelves with categories ---
        for shelf in data.get("shelves", []):
            category = shelf["category"]
            product_list = categories.get(category, [])

            codes = [p["code"] for p in product_list]
            n_products = len(codes)
            cells = shelf["width"] * shelf["height"]

            block_size = max(1, cells // n_products)

            cell_idx = 0
            product_idx = 0

            for i in range(shelf["y"], shelf["y"] + shelf["height"]):
                for j in range(shelf["x"], shelf["x"] + shelf["width"]):
                    grid[i, j] = codes[product_idx]
                    cell_idx += 1

                    if cell_idx >= block_size:
                        cell_idx = 0
                        product_idx = min(product_idx + 1, n_products - 1)

        return grid

    def _plot_layout(self, ax) -> None:
        h, w = self.layout_array.shape
        numeric_grid = np.zeros((h, w), dtype=int)

        for y in range(h):
            for x in range(w):
                cell = self.layout_array[y, x]
                if cell == "0":
                    numeric_grid[y, x] = 0
                elif cell == "#":
                    numeric_grid[y, x] = 1
                elif cell == "I":
                    numeric_grid[y, x] = 2
                elif cell == "E":
                    numeric_grid[y, x] = 3
                elif cell.startswith("P"):
                    numeric_grid[y, x] = 4

        cmap = ListedColormap(
            [
                "white",  # 0 empty
                "saddlebrown",  # 1 wall
                "green",  # 2 entrance
                "red",  # 3 exit
                "gold",  # 4 shelf
            ]
        )

        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = BoundaryNorm(bounds, cmap.N)

        ax.imshow(numeric_grid, origin="lower", cmap=cmap, norm=norm)

        ax.invert_yaxis()

        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.3)
        ax.tick_params(
            which="both", bottom=False, left=False, labelbottom=False, labelleft=False
        )
        for y in range(h):
            for x in range(w):
                cell = self.layout_array[y, x]
                if cell.startswith("P"):
                    ax.text(
                        x, y, cell, ha="center", va="center", fontsize=6, color="black"
                    )

    def _load_products(
        self, filename: str
    ) -> tuple[list[Product], dict[str, Product], dict[str, list[Product]]]:
        """
        Load products & categories from the same YAML file as the layout.
        Returns:
            - products_list: [Product]
            - products_by_code: dict[str, Product]
            - products_by_category: dict[str, list[Product]]
        """
        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        products_by_code: dict[str, Product] = {}
        products_by_category: dict[str, list[Product]] = {}
        products_list = []

        for category, products in data.get("categories", {}).items():
            products_by_category[category] = []

            for p in products:
                product = Product(
                    name=p["name"],
                    category=category,
                )

                products_list.append(product)
                products_by_category[category].append(product)
                products_by_code[p["code"]] = product

        products_by_code["E"] = Product(name="Exit", category="Exit", waiting_time=0)

        return products_list, products_by_code, products_by_category

    def generate_shopping_list(self) -> list[Product]:
        """
        Generate a random shopping list for an agent.
        """
        num_items = random.randint(0, len(self.products_list))
        shopping_list = random.sample(self.products_list, num_items)

        # Add the exit at the end
        shopping_list.append(self._products_by_code["E"])

        return shopping_list
