import numpy as np
import os

def load_layout(filename):
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

filename = os.path.join("configs", "supermarket2.csv")
layout_array = load_layout(filename)
print(layout_array)