import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_layout_csv(filename):
    """
    Load a supermarket layout from a csv file, return an np array

    Legend in file:
    0: empty, walkable
    #: shelf
    I: in/entrance
    E: exit
    X: unreachable

    """
    
    layout_list = []
    with open(filename, 'r') as f:
        for line in f:
            # Split by comma, strip whitespace
            row = [item.strip() for item in line.strip().split(',')]
            layout_list.append(row)

    return np.array(layout_list, dtype=str)


def load_layout_yaml(filename):
    """
    Load a supermarket layout from a yaml file, return an np array
    """
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)

    width = data['width']
    height = data['height']

    # Initialize empty grid
    grid = np.full((height, width), '0', dtype=str)

    # Helper: fill rectangle
    def fill_rectangle(x, y, w, h, char):
        # y-axis is vertical (row), x-axis is horizontal (column)
        # Be careful: row index = height - 1 - y if origin is bottom-left
        # Here assume origin at bottom-left
        for i in range(y, y + h):
            for j in range(x, x + w):
                if 0 <= i < height and 0 <= j < width:
                    grid[i, j] = char

    # Fill walls/shelves
    for wall in data.get('walls', []):
        x, y = wall['x'], wall['y']
        w, h = wall['width'], wall['height']
        fill_rectangle(x, y, w, h, '#')

    # Fill entrances
    for entrance in data.get('entrance', []):
        x0, y0 = entrance['start']
        x1, y1 = entrance['end']
        # fill as a line
        if x0 == x1:  # vertical line
            for i in range(min(y0, y1), max(y0, y1) + 1):
                grid[i, x0] = 'I'
        elif y0 == y1:  # horizontal line
            for j in range(min(x0, x1), max(x0, x1) + 1):
                grid[y0, j] = 'I'

    # Fill exits
    for exit_ in data.get('exit', []):
        x0, y0 = exit_['start']
        x1, y1 = exit_['end']
        if x0 == x1:  # vertical line
            for i in range(min(y0, y1), max(y0, y1) + 1):
                grid[i, x0] = 'E'
        elif y0 == y1:  # horizontal line
            for j in range(min(x0, x1), max(x0, x1) + 1):
                grid[y0, j] = 'E'

    return grid


def plot_layout(layout_array):
    # Map characters to numeric codes
    char_to_num = {'0': 0, '#': 1, 'I': 2, 'E': 3, 'X': 4}
    numeric_grid = np.vectorize(char_to_num.get)(layout_array)

    # Create a colormap (order must match numeric codes)
    cmap = ListedColormap(['white', 'saddlebrown', 'green', 'red'])

    # Plot
    height, width = layout_array.shape
    fig, ax = plt.subplots(figsize=(width/5, height/5))
    im = ax.imshow(numeric_grid, origin='lower', cmap=cmap)

    # Optional: grid lines
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.show()

# # Example usage
# filename = os.path.join("configs", "supermarket1.yaml")
# layout_array = load_layout_yaml(filename)
# print(layout_array)
# plot_layout(layout_array)