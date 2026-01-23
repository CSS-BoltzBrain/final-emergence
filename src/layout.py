import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dataclasses import dataclass
from typing import Dict, List


def load_layout_yaml(filename: str) -> np.ndarray[str]:
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
        fill_rectangle(
            wall["x"], wall["y"],
            wall["width"], wall["height"],
            "#"
        )


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

def plot_layout(layout_array: np.ndarray) -> None:
    h, w = layout_array.shape
    numeric_grid = np.zeros((h, w), dtype=int)

    for i in range(h):
        for j in range(w):
            cell = layout_array[i, j]
            if cell == '0':
                numeric_grid[i, j] = 0
            elif cell == '#':
                numeric_grid[i, j] = 1
            elif cell == 'I':
                numeric_grid[i, j] = 2
            elif cell == 'E':
                numeric_grid[i, j] = 3
            elif cell.startswith('P'):
                numeric_grid[i, j] = 4

    cmap = ListedColormap([
        'white',        # 0 empty
        'saddlebrown',  # 1 wall
        'green',        # 2 entrance
        'red',          # 3 exit
        'gold'          # 4 shelf
    ])

    fig, ax = plt.subplots(figsize=(w / 4, h / 4))
    ax.imshow(numeric_grid, origin='lower', cmap=cmap)

    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.3)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    for i in range(h):
        for j in range(w):
            cell = layout_array[i, j]
            if cell.startswith("P"):
                ax.text(
                    j, i, cell,
                    ha='center', va='center',
                    fontsize=6,
                    color='black'
                )
    plt.tight_layout()
    plt.show()

@dataclass
class Product:
    name: str
    code: str
    popularity: float
    category: str


def load_products(filename: str):
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
                code=p["code"],
                popularity=float(p["popularity"]),
                category=category,
            )

            products_list.append(product)
            products_by_code[product.code] = product
            products_by_category[category].append(product)

    return products_list, products_by_code, products_by_category


# # Example usage
# filename = os.path.join("configs", "supermarket1.yaml")
# layout_array = load_layout_yaml(filename)
# products_list, products_by_code, products_by_category= load_products(filename)
# # print(products_by_category)
# # print(products_by_code)
# # print(products_list)
# # np.set_printoptions(threshold=np.inf)
# print(layout_array)
# plot_layout(layout_array)