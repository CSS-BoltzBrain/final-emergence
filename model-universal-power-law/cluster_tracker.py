"""
Cluster size tracking for TTC pedestrian simulations.

Tracks spatial clusters of agents to investigate self-organized criticality.
A cluster is a connected component of agents within a threshold distance
of each other.

If the system exhibits SOC, the cluster size distribution P(s) should
follow a power law: P(s) ~ s^(-tau).
"""

import numpy as np


class UnionFind:
    """Union-Find (Disjoint Set Union) data structure for efficient clustering."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


class ClusterTracker:
    """
    Track spatial clusters of agents in a simulation.

    A cluster is defined as a connected component where two agents are
    connected if their distance is less than cluster_threshold.

    Usage:
        tracker = ClusterTracker(num_agents=20, cluster_threshold=0.6)
        for step in simulation:
            sim.step()
            if step % sample_interval == 0:
                tracker.update(sim.pos, sim.active)
        cluster_sizes = tracker.get_all_cluster_sizes()
    """

    def __init__(self, num_agents, cluster_threshold=0.6, env=None):
        """
        Initialize the cluster tracker.

        Args:
            num_agents: Number of agents in simulation
            cluster_threshold: Max distance for two agents to be connected (meters).
                              Default 0.6m = 3 * agent_radius (0.2m).
            env: Optional environment for distance wrapping (torus/corridor).
        """
        self.num_agents = num_agents
        self.cluster_threshold = cluster_threshold
        self.env = env

        # Storage for all recorded cluster sizes
        self.all_cluster_sizes = []

        # Statistics
        self.num_samples = 0

    def _compute_distance(self, pos_i, pos_j):
        """Compute distance between two positions, with optional wrapping."""
        d = pos_j - pos_i
        if self.env is not None:
            # Use environment's wrapping for shortest distance
            d = d.copy()  # Don't modify original
            self.env.wrap_relative(d)
        return np.sqrt(d[0]**2 + d[1]**2)

    def compute_clusters(self, positions, active):
        """
        Compute clusters at current timestep.

        Args:
            positions: np.array of shape (num_agents, 2)
            active: np.array of bools indicating which agents are active

        Returns:
            List of cluster sizes (integers), one per cluster
        """
        # Get indices of active agents
        active_indices = np.where(active)[0]
        n_active = len(active_indices)

        if n_active == 0:
            return []

        if n_active == 1:
            return [1]

        # Build Union-Find structure
        # Map active agent indices to 0..n_active-1 for Union-Find
        idx_map = {orig: i for i, orig in enumerate(active_indices)}
        uf = UnionFind(n_active)

        # Find connected pairs
        threshold_sq = self.cluster_threshold ** 2
        for i, idx_i in enumerate(active_indices):
            for j, idx_j in enumerate(active_indices):
                if j <= i:
                    continue

                # Compute distance
                d = positions[idx_j] - positions[idx_i]
                if self.env is not None:
                    d = d.copy()
                    self.env.wrap_relative(d)
                dist_sq = d[0]**2 + d[1]**2

                if dist_sq < threshold_sq:
                    uf.union(i, j)

        # Count cluster sizes
        cluster_counts = {}
        for i in range(n_active):
            root = uf.find(i)
            cluster_counts[root] = cluster_counts.get(root, 0) + 1

        return list(cluster_counts.values())

    def update(self, positions, active):
        """
        Record cluster sizes at current timestep.

        Args:
            positions: np.array of shape (num_agents, 2)
            active: np.array of bools indicating which agents are active

        Called at each sample interval during simulation.
        """
        cluster_sizes = self.compute_clusters(positions, active)
        self.all_cluster_sizes.extend(cluster_sizes)
        self.num_samples += 1

    def get_all_cluster_sizes(self):
        """
        Return list of all recorded cluster sizes across all timesteps.
        """
        return self.all_cluster_sizes.copy()

    def get_statistics(self):
        """
        Return summary statistics about cluster sizes.

        Returns:
            dict with keys: total_clusters, mean_size, max_size, etc.
        """
        sizes = self.all_cluster_sizes
        if not sizes:
            return {
                'total_clusters': 0,
                'num_samples': self.num_samples,
                'mean_size': 0.0,
                'max_size': 0,
                'min_size': 0,
            }

        return {
            'total_clusters': len(sizes),
            'num_samples': self.num_samples,
            'mean_size': np.mean(sizes),
            'max_size': int(np.max(sizes)),
            'min_size': int(np.min(sizes)),
            'size_distribution': dict(zip(*np.unique(sizes, return_counts=True))),
        }

    def get_current_clusters(self, positions, active):
        """
        Get cluster information for current state (for visualization).

        Returns:
            List of lists, where each inner list contains agent indices in that cluster.
        """
        active_indices = np.where(active)[0]
        n_active = len(active_indices)

        if n_active == 0:
            return []

        if n_active == 1:
            return [[active_indices[0]]]

        # Build Union-Find
        uf = UnionFind(n_active)
        threshold_sq = self.cluster_threshold ** 2

        for i, idx_i in enumerate(active_indices):
            for j, idx_j in enumerate(active_indices):
                if j <= i:
                    continue
                d = positions[idx_j] - positions[idx_i]
                if self.env is not None:
                    d = d.copy()
                    self.env.wrap_relative(d)
                dist_sq = d[0]**2 + d[1]**2
                if dist_sq < threshold_sq:
                    uf.union(i, j)

        # Group agents by cluster
        clusters = {}
        for i, orig_idx in enumerate(active_indices):
            root = uf.find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(orig_idx)

        return list(clusters.values())
