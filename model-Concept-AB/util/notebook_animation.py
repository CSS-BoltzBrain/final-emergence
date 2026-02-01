import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class NotebookAnimation:
    def __init__(self, engine):
        self.engine = engine
        self.cfg = engine.cfg

        self.fig, self.ax = plt.subplots(figsize=(15, 3))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)

        self.ax.set_xlim(self.cfg["domain"]["xmin"], self.cfg["domain"]["xmax"])
        self.ax.set_ylim(self.cfg["domain"]["ymin"], self.cfg["domain"]["ymax"])
        self.ax.set_aspect("equal")

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self._draw_walls()

        self.scat = self.ax.scatter([], [], s=80, edgecolors="black", linewidth=0.5, alpha=0.9)

        self.time_text = self.ax.text(
            0.01, 0.9, "", transform=self.ax.transAxes,
            fontsize=10, fontweight="bold", va="top"
        )
        self.status_text = self.ax.text(
            0.99, 0.9, "", transform=self.ax.transAxes,
            fontsize=10, fontweight="bold", ha="right", va="top"
        )

    def _max_agents(self):
        if hasattr(self.engine, "max_agents"):
            return self.engine.max_agents
        return self.cfg.get("spawn", {}).get("max_agents", 0)

    def _total_spawned(self):
        if hasattr(self.engine, "total_spawned"):
            return self.engine.total_spawned
        return 0

    def _draw_walls(self):
        for wall in self.engine.walls:
            if wall.type == "horizontal":
                x0, x1 = wall.x_range
                self.ax.plot([x0, x1], [wall.y, wall.y], "k-", linewidth=3)

            elif wall.type == "vertical":
                y0, y1 = wall.y_range
                self.ax.plot([wall.x, wall.x], [y0, y1], "k-", linewidth=3)

    def update(self, frame):
        steps_per_frame = 5
        for _ in range(steps_per_frame):
            self.engine.step()

        agents = self.engine.agents
        if agents:
            pos = np.array([ag.pos for ag in agents])

            # colour by direction (same heuristic you were using)
            colors = []
            for ag in agents:
                colors.append("#d32f2f" if ag.target_pos[0] > 10 else "#1976d2")

            self.scat.set_offsets(pos)
            self.scat.set_facecolors(colors)
        else:
            self.scat.set_offsets(np.empty((0, 2)))

        total_spawned = self._total_spawned()
        max_agents = self._max_agents()
        self.time_text.set_text(
            f"Time: {self.engine.time:.2f} s | Active: {len(self.engine.agents)} | "
            f"Total: {total_spawned}/{max_agents}"
        )
        return self.scat, self.time_text

    def simulation_generator(self, max_frames):
        for i in range(max_frames):
            total_spawned = self._total_spawned()
            max_agents = self._max_agents()
            if total_spawned >= max_agents and len(self.engine.agents) == 0:
                print(f"Simulation Finished at frame {i}")
                return
            yield i

    def run(self, max_frames=2000, interval=30):
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=self.simulation_generator(max_frames),
            interval=interval,
            blit=True,
            save_count=max_frames,
        )
        plt.close(self.fig)
        return anim
