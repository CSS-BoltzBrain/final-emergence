"""
MAZE SOLVER MODULE (v4 - Hybrid Scalable with Type Hinting)

This module handles the logic for translating a shopping list into an optimized
walking path through a grid-based store layout.

1. INPUTS:
   - layout_grid (numpy.ndarray): A 2D array of strings representing the map.
     ('0' = walkable aisle, 'I' = Entrance, 'E' = Exit, 'P1'...'Pn' = Shelves).
   - start_pos (Tuple[int, int]): The (row, col) coordinates of the 'I' (Entrance).
   - shopping_list_names (List[str]): A simple list of product names (e.g., ['milk', 'bread']).
   - exit_pos (Tuple[int, int]): The (row, col) coordinates of the 'E' (Exit).

2. OUTPUT:
   - path (List[Tuple[int, int]]): An ordered list of coordinate tuples.
     Example: [(0, 0), (0, 1), (1, 1), ...]
     This represents the step-by-step coordinates the agent must visit to
     collect all items in the optimal order and exit.

ALGORITHM OVERVIEW:
-------------------
1. HYBRID OPTIMIZATION:
   - Small Lists (<= 9 items): Uses Brute Force TSP (N!) for mathematically perfect routes.
   - Large Lists (> 9 items): Uses Nearest Neighbor + 2-Opt Heuristic for speed.
2. PATHFINDING:
   - Uses A* (A-Star) to navigate around shelves/walls between targets.
"""

import heapq
import numpy as np
import itertools
from typing import List, Tuple, Dict, Any

from shopmap import ShopMap

# --- EXPLICIT TYPE DEFINITIONS ---
# Grid: 2D Numpy array representing the store map.
Grid = np.ndarray

# Coord: A specific location on the grid given as (row, column).
Coord = Tuple[int, int]

# Path: An ordered sequence of coordinates representing the agent's route.
# Returns: [(r1, c1), (r2, c2), (r3, c3)...]
Path = List[Coord]

# ProductDict: Maps internal codes (e.g., 'P1') to Product objects/metadata.
ProductDict = Dict[str, Any]

# ShoppingList: Simple list of product names strings.
ShoppingList = List[str]


class AgentPathfinder:
    def __init__(self, shop_map: ShopMap):
        """
        Initialize the pathfinder with the static map data.
        """
        self.grid = shop_map.layout_array
        self.height, self.width = self.grid.shape
        self.products_by_code = shop_map.product_dict
        self.name_to_code = {p.name: code for code, p in self.products_by_code.items()}

        # Cache for distances to avoid re-running BFS for known pairs
        # Key: ((r1, c1), (r2, c2)), Value: int distance
        self.memo_dist = {}

    def solve_path(
        self, start_pos: Coord, shopping_list_names: ShoppingList, exit_pos: Coord
    ) -> Path:
        """
        Main entry point to calculate the full shopping route.

        Returns:
            Path: A list of (row, col) tuples.
        """
        # 1. Resolve Targets
        items_data = []
        for name in shopping_list_names:
            if name not in self.name_to_code:
                # print(f"Warning: Item '{name}' unknown.")
                continue

            code = self.name_to_code[name]
            shelf_locs = self._find_product_locations(code)

            if not shelf_locs:
                # print(f"Warning: Item '{name}' ({code}) not placed on map.")
                continue

            center_loc = shelf_locs[len(shelf_locs) // 2]
            target_aisle = self._get_nearest_aisle(center_loc)

            if target_aisle:
                items_data.append({"name": name, "pos": target_aisle})

        start_node = self._get_nearest_aisle(start_pos)

        exit_node = self._get_nearest_aisle(exit_pos)

        # 2. Optimize Route based on list size
        if len(items_data) <= 9:
            # print(f"Optimization: Using EXACT solver for {len(items_data)} items.")
            sorted_items = self._optimize_route_exact(start_node, items_data, exit_node)
        else:
            # print(f"Optimization: Using HEURISTIC solver for {len(items_data)} items.")
            sorted_items = self._optimize_route_heuristic(
                start_node, items_data, exit_node
            )

        # 3. Generate Path (A*)
        full_path = [start_node]
        curr = start_node

        # for item in sorted_items:
        #     segment = self._a_star(curr, item["pos"])
        #     if segment:
        #         full_path.extend(segment[1:])
        #         curr = item["pos"]

        if exit_node:
            segment = self._a_star(curr[::-1], exit_node[::-1])
            if segment:
                full_path.extend(segment[1:])

        return full_path

    # =========================================================================
    #  OPTIMIZATION STRATEGIES
    # =========================================================================

    def _optimize_route_exact(
        self, start: Coord, items: List[Dict], exit_pos: Coord
    ) -> List[Dict]:
        """
        Brute Force TSP. Checks every possible permutation.
        Guarantees mathematically perfect route.
        """
        n = len(items)
        if n == 0:
            return []

        item_indices = list(range(n))
        best_order = None
        min_total_dist = float("inf")

        for perm in itertools.permutations(item_indices):
            # Distance: Start -> First Item
            current_dist = self._get_memoized_dist(start, items[perm[0]]["pos"])

            # Distance: Item -> Item
            valid = True
            for i in range(n - 1):
                d = self._get_memoized_dist(
                    items[perm[i]]["pos"], items[perm[i + 1]]["pos"]
                )
                if d == float("inf"):
                    valid = False
                    break
                current_dist += d

            if not valid:
                continue

            # Distance: Last Item -> Exit
            d_exit = self._get_memoized_dist(items[perm[-1]]["pos"], exit_pos)
            current_dist += d_exit

            if current_dist < min_total_dist:
                min_total_dist = current_dist
                best_order = perm

        return [items[i] for i in best_order]

    def _optimize_route_heuristic(
        self, start: Coord, items: List[Dict], exit_pos: Coord
    ) -> List[Dict]:
        """
        Scalable Heuristic for Large Lists.
        1. Greedy Nearest Neighbor (using TRUE BFS distance).
        2. 2-Opt Optimization pass to untangle crossing paths.
        """
        if not items:
            return []

        # --- Phase 1: Greedy Nearest Neighbor (with True Distance) ---
        path = []
        remaining = items[:]
        curr = start

        while remaining:
            # Find closest item using cached BFS distance
            best_idx = -1
            min_dist = float("inf")

            for i, item in enumerate(remaining):
                d = self._get_memoized_dist(curr, item["pos"])
                if d < min_dist:
                    min_dist = d
                    best_idx = i

            # Move to best
            next_item = remaining.pop(best_idx)
            path.append(next_item)
            curr = next_item["pos"]

        # --- Phase 2: 2-Opt Local Search ---
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1

            # Calculate current total distance
            current_total = self._calculate_path_cost(start, path, exit_pos)

            for i in range(len(path) - 1):
                for j in range(i + 1, len(path)):
                    # Create a new path with segment reversed
                    new_path = path[:i] + path[i : j + 1][::-1] + path[j + 1 :]

                    new_total = self._calculate_path_cost(start, new_path, exit_pos)

                    if new_total < current_total:
                        path = new_path
                        current_total = new_total
                        improved = True
                        break
                if improved:
                    break

        return path

    def _calculate_path_cost(
        self, start: Coord, item_path: List[Dict], exit_pos: Coord
    ) -> int:
        dist = self._get_memoized_dist(start, item_path[0]["pos"])
        for k in range(len(item_path) - 1):
            dist += self._get_memoized_dist(
                item_path[k]["pos"], item_path[k + 1]["pos"]
            )
        dist += self._get_memoized_dist(item_path[-1]["pos"], exit_pos)
        return dist

    # =========================================================================
    #  CORE UTILS
    # =========================================================================

    def _get_memoized_dist(self, p1: Coord, p2: Coord) -> int:
        """Returns cached BFS distance between two points."""
        if p1 == p2:
            return 0
        key = tuple(sorted((p1, p2)))

        if key in self.memo_dist:
            return self.memo_dist[key]

        d = self._bfs_distance(p1, p2)
        self.memo_dist[key] = d
        return d

    def _bfs_distance(self, start: Coord, goal: Coord) -> int:
        """Actual step count ignoring shelf cost, just walls vs empty."""
        queue = [(start, 0)]
        visited = {start}

        while queue:
            (curr_r, curr_c), dist = queue.pop(0)
            if (curr_r, curr_c) == goal:
                return dist

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if self._is_walkable(nr, nc):
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))
        return float("inf")

    def _get_nearest_aisle(self, target_pos: Coord) -> Coord:
        x, y = target_pos
        if self._is_walkable(x, y):
            return (x, y)

        queue = [(y, x)]
        visited = set([(y, x)])

        while queue:
            curr_y, curr_x = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = curr_y + dy, curr_x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if (ny, nx) not in visited:
                        if self._is_walkable(ny, nx):
                            return (nx, ny)
                        visited.add((ny, nx))
                        queue.append((ny, nx))
        return None

    def _a_star(self, start: Coord, goal: Coord) -> Path:
        if start == goal:
            return [start]
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}
        open_set_hash = {start}

        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)

            if current == goal or g_score[current] > 100:
                path = []
                while current in came_from:
                    path.append(current[::-1])
                    current = came_from[current]
                path.append(start[::-1])
                return path[::-1]

            for dr, dc in dirs:
                nr, nc = current[0] + dr, current[1] + dc
                if self._is_walkable(nr, nc):
                    tent_g = g_score[current] + 1
                    if (nr, nc) not in g_score or tent_g < g_score[(nr, nc)]:
                        came_from[(nr, nc)] = current
                        g_score[(nr, nc)] = tent_g
                        f_score[(nr, nc)] = (
                            tent_g + abs(nr - goal[0]) + abs(nc - goal[1])
                        ) + np.random.uniform(0, 0.001)
                        if (nr, nc) not in open_set_hash:
                            heapq.heappush(open_set, (f_score[(nr, nc)], (nr, nc)))
                            open_set_hash.add((nr, nc))
        return []

    def _find_product_locations(self, code: str) -> List[Coord]:
        matches = np.argwhere(self.grid == code)
        return [(r, c) for r, c in matches]

    def _is_walkable(self, y: int, x: int) -> bool:
        if not (0 <= y < self.height and 0 <= x < self.width):
            return False
        val = self.grid[y, x]
        return val == "0" or val == "I" or val == "E"
