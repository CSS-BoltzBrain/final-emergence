"""
Headless simulation engine for the TTC (time-to-collision) pedestrian model.

Extracted from ttcSimTK.py by Karamouzas, Skinner & Guy (KGS).
Reference: PRL 113, 238701 (2014) - "Universal Power Law Governing
Pedestrian Interactions"

This module contains the physics of the KGS anticipatory collision avoidance
model without any GUI / Tkinter dependencies.  Geometry-dependent operations
(wrapping, boundary conditions, wall forces, agent placement) are delegated
to an Environment object from environment.py.
"""

import random as rnd
from math import sqrt, exp
import numpy as np

from environment import TorusEnvironment


class TTCSimulation:
    """
    Simulation of pedestrian agents using the KGS anticipatory collision
    avoidance model.

    The simulation geometry is determined by the *env* parameter — an
    Environment instance (from environment.py) that handles wrapping,
    boundary conditions, wall forces, and agent initialization.

    The interaction energy between a pair of pedestrians is (Eq. 2):
        E(tau) = k / tau^m * exp(-tau / t0)

    where tau is the time to collision, k is the energy prefactor,
    m is the power law exponent (= 2 from empirical data), and t0 is
    the interaction time horizon.

    The avoidance force is F = -dE/dr, the negative spatial gradient
    of the interaction energy (Eq. 3).
    """

    def __init__(self, env=None, k=1.5, m=2.0, t0=3, rad=0.2,
                 sight=7, maxF=5, dt=0.02, seed=None,
                 # Backward compatibility (ignored when env is provided):
                 num=18, s=4):
        # Create default torus environment if none provided
        if env is None:
            env = TorusEnvironment(s=s, num=num)

        self.env = env
        self.num = env.num

        # Interaction energy parameters (Eq. 2):
        #   E(tau) = k / tau^m * exp(-tau / t0)
        self.k = k          # energy prefactor
        self.m = m          # power law exponent
        self.t0 = t0        # interaction time horizon tau_0 (seconds)

        self.rad = rad      # collision radius (meters)
        self.sight = sight  # neighbor search range (meters)
        self.maxF = maxF    # maximum force magnitude
        self.dt = dt        # integration timestep (seconds)

        if seed is not None:
            rnd.seed(seed)
            np.random.seed(seed)

        # Agent state arrays — initialized by the environment
        self.pos, self.vel, self.gvel = env.init_agents(self.rad)

        # Track which agents are active (for absorbing boundary conditions)
        self.active = np.ones(self.num, dtype=bool)

    def find_neighbors(self):
        """
        Find neighbors within sight range for each agent.
        Uses the environment's wrapping to compute shortest distance.
        Skips inactive agents (for absorbing boundary conditions).

        Returns:
            nbr: list of lists -- nbr[i] = neighbor indices of agent i
            nd:  list of lists -- nd[i]  = distances to those neighbors
        """
        nbr = []
        nd = []
        sight2 = self.sight ** 2

        for i in range(self.num):
            nbr_i = []
            nd_i = []
            if not self.active[i]:
                nbr.append(nbr_i)
                nd.append(nd_i)
                continue
            for j in range(self.num):
                if i == j or not self.active[j]:
                    continue
                d = self.pos[i] - self.pos[j]
                self.env.wrap_relative(d)
                l2 = d.dot(d)
                if l2 < sight2:
                    nbr_i.append(j)
                    nd_i.append(sqrt(l2))
            nbr.append(nbr_i)
            nd.append(nd_i)

        return nbr, nd

    def compute_ttc(self, pa, pb, va, vb, ra, rb):
        """
        Compute the time to collision (tau) between two agents.

        Time to collision is the smallest positive time t at which two agents,
        continuing at their current velocities, would collide:

            |p_rel + v_rel * t|^2 = (ra + rb)^2

        Expanding gives a quadratic in t:

            a * t^2 + b * t + c = 0

        where:
            p_rel = pb - pa          (relative position)
            v_rel = vb - va          (relative velocity)
            a     = |v_rel|^2
            b     = 2 * dot(p_rel, v_rel)
            c     = |p_rel|^2 - (ra + rb)^2

        We take the smallest positive root as tau.

        Returns:
            tau: time to collision (999 if no collision is predicted)
        """
        maxt = 999

        p = pb - pa  # relative position
        self.env.wrap_relative(p)

        rv = vb - va  # relative velocity

        a = rv.dot(rv)
        b = 2 * rv.dot(p)
        c = p.dot(p) - (ra + rb) ** 2

        det = b * b - 4 * a * c
        t1 = maxt
        t2 = maxt
        if det > 0:
            t1 = (-b + sqrt(det)) / (2 * a)
            t2 = (-b - sqrt(det)) / (2 * a)
        t = min(t1, t2)

        if t < 0 and max(t1, t2) > 0:  # currently overlapping
            t = 100
        if t < 0:
            t = maxt
        if t > maxt:
            t = maxt

        return t

    def compute_dE(self, pa, pb, va, vb, ra, rb):
        """
        Compute the spatial gradient of the interaction energy dE/dr.

        The interaction energy is (Eq. 2):
            E(tau) = k / tau^m * exp(-tau / t0)

        The avoidance force is (Eq. 3):
            F = -grad_r [ k / tau^m * exp(-tau / t0) ]

        This requires differentiating through the tau(r, v) dependence
        using the chain rule.  The sign conventions follow the KGS
        reference code (ttcSimTK.py lines 169-199):

            w = pb - pa          (relative position, note: opposite to ttc)
            v = va - vb          (relative velocity, opposite to ttc)

        The analytical gradient is:
            dE/dr = k * exp(-t/t0) * [v - (v*b - w*a)/sqrt(discr)]
                    / (a * t^m) * (m/t + 1/t0)

        where a, b, discr are the quadratic coefficients in this convention.

        Returns:
            2D gradient vector, or [0, 0] if no collision is predicted.
        """
        k = self.k
        m = self.m
        t0 = self.t0
        maxt = 999

        w = pb - pa
        self.env.wrap_relative(w)

        v = va - vb
        radius = ra + rb
        dist = sqrt(w[0] ** 2 + w[1] ** 2)
        if radius > dist:
            radius = 0.99 * dist

        a = v.dot(v)
        b = w.dot(v)
        c = w.dot(w) - radius * radius
        discr = b * b - a * c

        if discr < 0 or (-0.001 < a < 0.001):
            return np.array([0.0, 0.0])

        discr = sqrt(discr)
        t = (b - discr) / a

        if t < 0:
            return np.array([0.0, 0.0])
        if t > maxt:
            return np.array([0.0, 0.0])

        d = (k * exp(-t / t0) * (v - (v * b - w * a) / discr)
             / (a * t ** m) * (m / t + 1 / t0))

        return d

    def step(self):
        """
        Advance the simulation by one timestep dt.

        1. Neighbor search (geometry-aware, skips inactive agents)
        2. Force computation per agent:
           - Driving force toward goal velocity: (gvel - vel) / 0.5
           - Random perturbation
           - Wall force (from environment; zero for open geometries)
           - Collision avoidance: F_avoid = -dE/dr for each neighbor pair
        3. Euler integration of velocity and position
        4. Boundary conditions (from environment)
        5. Deactivate agents that exit absorbing boundaries
        """
        nbr, _ = self.find_neighbors()
        dt = self.dt

        F = np.zeros((self.num, 2))

        for i in range(self.num):
            if not self.active[i]:
                continue

            # Driving force toward goal velocity (relaxation time = 0.5 s)
            F[i] += (self.gvel[i] - self.vel[i]) / 0.5
            # Random perturbation
            F[i] += np.array([rnd.uniform(-1., 1.), rnd.uniform(-1., 1.)])
            # Wall force (non-zero only for walled environments)
            F[i] += self.env.wall_force(self.pos[i], self.rad)

            for j in nbr[i]:
                d = self.pos[i] - self.pos[j]
                self.env.wrap_relative(d)

                r = self.rad
                dist_val = sqrt(d.dot(d))
                if dist_val < 2 * self.rad:
                    r = dist_val / 2.001  # shrink overlapping agents

                dEdx = self.compute_dE(self.pos[i], self.pos[j],
                                       self.vel[i], self.vel[j], r, r)
                FAvoid = -dEdx

                mag = sqrt(FAvoid.dot(FAvoid))
                if mag > self.maxF:
                    FAvoid = self.maxF * FAvoid / mag

                F[i] += FAvoid

        # Euler integration
        for i in range(self.num):
            if not self.active[i]:
                continue

            self.vel[i] += F[i] * dt
            self.pos[i] += self.vel[i] * dt

            # apply_boundary returns True if agent should be removed
            # (absorbing boundary condition)
            should_remove = self.env.apply_boundary(self.pos[i], self.rad)
            if should_remove:
                self.active[i] = False

    def compute_all_pairwise_ttc(self):
        """
        Compute time to collision for all simultaneously-present agent pairs.

        Each unique pair (i, j) with i < j is counted once.  This produces
        the time-to-collision data used to construct the observed probability
        density in the pair distribution function g(tau).

        Returns:
            list of tau values for pairs with finite TTC (< 999)
        """
        ttc_values = []
        for i in range(self.num):
            for j in range(i + 1, self.num):
                tau = self.compute_ttc(
                    self.pos[i], self.pos[j],
                    self.vel[i], self.vel[j],
                    self.rad, self.rad)
                if tau < 999:
                    ttc_values.append(tau)
        return ttc_values

    def compute_scrambled_ttc(self, snapshots, num_partners=5):
        """
        Compute scrambled time-to-collision by pairing agent states from
        different timesteps.

        The pair distribution function g(tau) requires a non-interacting
        baseline.  Following the paper, this baseline is obtained by
        computing time-to-collision between pairs of pedestrians that are
        not simultaneously present in the scene.

        We approximate non-simultaneously-present pairs by pairing agent
        i's state at frame f1 with agent j's state at a *different* frame
        f2.  Using multiple partner frames per source frame gives a larger
        and more representative scrambled sample.

        Args:
            snapshots: list of (pos, vel) tuples — one per sampled frame
            num_partners: number of partner frames per source frame

        Returns:
            list of tau values for scrambled pairs with finite TTC (< 999)
        """
        n_frames = len(snapshots)
        if n_frames < 2:
            return []

        scrambled_ttc = []
        rng = rnd.Random(42)  # deterministic scrambling

        for f1 in range(n_frames):
            partners = set()
            while len(partners) < min(num_partners, n_frames - 1):
                f2 = rng.randint(0, n_frames - 1)
                if f2 != f1:
                    partners.add(f2)

            pos1, vel1 = snapshots[f1]
            for f2 in partners:
                pos2, vel2 = snapshots[f2]

                for i in range(self.num):
                    for j in range(i + 1, self.num):
                        tau = self.compute_ttc(pos1[i], pos2[j],
                                              vel1[i], vel2[j],
                                              self.rad, self.rad)
                        if tau < 999:
                            scrambled_ttc.append(tau)

        return scrambled_ttc
