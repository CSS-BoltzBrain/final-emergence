import numpy as np
import heapq

class NavigationGrid:
    """
    Grid distance field via Dijkstra with a gradient lookup.
    """
    def __init__(self, domain_cfg, walls, target_pos, dx=0.1):
        self.xmin, self.xmax = domain_cfg["xmin"], domain_cfg["xmax"]
        self.ymin, self.ymax = domain_cfg["ymin"], domain_cfg["ymax"]
        self.dx = dx
        self.cols = int(np.ceil((self.xmax - self.xmin) / dx))
        self.rows = int(np.ceil((self.ymax - self.ymin) / dx))
        self.grid_shape = (self.rows, self.cols)
        self.dist_map = np.full(self.grid_shape, np.inf)
        self.wall_map = np.zeros(self.grid_shape, dtype=bool)
        self._rasterize_walls(walls)
        self._compute_dijkstra(target_pos)
        self._compute_gradient_field()

    def _compute_gradient_field(self):
        """Finite-difference gradient on valid (non-wall) cells."""
        self.grad_y = np.zeros(self.grid_shape)
        self.grad_x = np.zeros(self.grid_shape)
        
        rows, cols = self.grid_shape
        dist = self.dist_map
        
        for r in range(rows):
            for c in range(cols):
                if np.isinf(dist[r, c]): continue
                val_c = dist[r, c]
                val_minus = dist[r-1, c] if r > 0 else np.inf
                val_plus = dist[r+1, c] if r < rows-1 else np.inf
                
                valid_minus = not np.isinf(val_minus)
                valid_plus = not np.isinf(val_plus)
                
                dy = 0.0
                if valid_minus and valid_plus:
                    dy = (val_minus - val_plus) / (2 * self.dx)
                    
                elif valid_plus:
                    dy = (val_c - val_plus) / self.dx
                    
                elif valid_minus:
                    dy = (val_minus - val_c) / self.dx
                    
                self.grad_y[r, c] = dy
                val_l = dist[r, c-1] if c > 0 else np.inf
                val_r = dist[r, c+1] if c < cols-1 else np.inf
                
                valid_l = not np.isinf(val_l)
                valid_r = not np.isinf(val_r)
                
                dx_val = 0.0
                if valid_l and valid_r:
                    dx_val = (val_l - val_r) / (2 * self.dx)
                    
                elif valid_r:
                    dx_val = (val_c - val_r) / self.dx
                    
                elif valid_l:
                    dx_val = (val_l - val_c) / self.dx
                    
                self.grad_x[r, c] = dx_val

    def _to_grid(self, pos):
        c = int((pos[0] - self.xmin) / self.dx)
        r = int((pos[1] - self.ymin) / self.dx)
        return r, c

    def _rasterize_walls(self, walls):
        for r in range(self.rows):
            y = self.ymin + r * self.dx + self.dx/2
            for c in range(self.cols):
                x = self.xmin + c * self.dx + self.dx/2
                pos = np.array([x, y])
                for w in walls:
                    dist, _ = w.distance_and_normal(pos)
                    if dist <= self.dx * 0.75:
                        self.wall_map[r, c] = True
                        break

    def _compute_dijkstra(self, target_pos):
        pq = []
        targets = [target_pos] if np.ndim(target_pos) == 1 else target_pos
        
        for tp in targets:
            tr, tc = self._to_grid(tp)
            tr = max(0, min(tr, self.rows - 1))
            tc = max(0, min(tc, self.cols - 1))
            if not self.wall_map[tr, tc]:
                self.dist_map[tr, tc] = 0.0
                heapq.heappush(pq, (0.0, tr, tc))
        
        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
        ]
        
        while pq:
            d, r, c = heapq.heappop(pq)
            if d > self.dist_map[r, c]: continue
            
            for dr, dc, cost in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if not self.wall_map[nr, nc]:
                        move_cost = cost * self.dx
                            
                        new_dist = d + move_cost
                        if new_dist < self.dist_map[nr, nc]:
                            self.dist_map[nr, nc] = new_dist
                            heapq.heappush(pq, (new_dist, nr, nc))

    def get_gradient(self, pos):
        """Return normalized gradient at pos using bilinear interpolation."""
        c_f = (pos[0] - self.xmin) / self.dx - 0.5
        r_f = (pos[1] - self.ymin) / self.dx - 0.5
        
        if c_f < 0: c_f = 0
        elif c_f > self.cols - 1.001: c_f = self.cols - 1.001
        if r_f < 0: r_f = 0
        elif r_f > self.rows - 1.001: r_f = self.rows - 1.001
        
        c0, r0 = int(c_f), int(r_f)
        c1, r1 = c0 + 1, r0 + 1
        wr, wc = r_f - r0, c_f - c0
        
        gx00, gx01 = self.grad_x[r0, c0], self.grad_x[r0, c1]
        gx10, gx11 = self.grad_x[r1, c0], self.grad_x[r1, c1]
        
        omr, omc = 1.0 - wr, 1.0 - wc
        gx = omr * (omc * gx00 + wc * gx01) + wr * (omc * gx10 + wc * gx11)
        
        gy00, gy01 = self.grad_y[r0, c0], self.grad_y[r0, c1]
        gy10, gy11 = self.grad_y[r1, c0], self.grad_y[r1, c1]
        gy = omr * (omc * gy00 + wc * gy01) + wr * (omc * gy10 + wc * gy11)
             
        norm_sq = gx*gx + gy*gy
        if norm_sq > 1e-12:
            inv_norm = 1.0 / np.sqrt(norm_sq)
            return np.array([gx * inv_norm, gy * inv_norm])
        return np.zeros(2)
