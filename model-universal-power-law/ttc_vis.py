"""
Unified Tkinter visualization for TTC pedestrian simulations.

Supports all environment types defined in environment.py:
  - torus: flat 2D torus (periodic in both x and y)
  - corridor: rectangular corridor (periodic x, walled y)
  - narrowdoor: corridor with a wall in the middle containing a narrow door
  - narrowdoor_onegroup: same as narrowdoor but with a single group

The visualization automatically adapts to the environment type loaded from
the YAML config file:
  - Canvas shape: square (torus) or rectangular (corridor-based)
  - Walls: drawn based on environment attributes
  - Agent colors: based on group membership or special roles (fast agent)

Usage:
    python ttc_vis.py --config config/torus.yaml
    python ttc_vis.py --config config/corridor.yaml
    python ttc_vis.py --config config/narrowdoor.yaml
    python ttc_vis.py --config config/narrowdoor_onegroup.yaml

Controls:
    SPACE   pause / resume
    S       step one frame (then pause)
    V       toggle velocity arrows on/off
    Escape  quit
"""

import argparse
from tkinter import Tk, Canvas, mainloop

from ttc_engine import TTCSimulation
from environment import load_config

# --- Display parameters (visualization-specific, not in YAML) -----------------
CANVAS_WIDTH = 900     # base canvas width for corridor environments
PIXEL_SIZE = 600       # canvas size for torus (square)
FRAME_DELAY = 30       # ms between frames
MAX_ITERATIONS = 10000
DRAW_VELS = True
WALL_THICKNESS = 6     # pixels


class TTCVisualization:
    """
    Unified Tkinter visualization for all TTC pedestrian simulations.

    Automatically adapts to the environment type:
    - Torus: square canvas, no walls, agent 0 (pink) is fast agent
    - Corridor: rectangular canvas, top/bottom walls, agents colored by group
    - NarrowDoor: corridor + middle wall with door gap
    - NarrowDoorOneGroup: narrowdoor with single group (all same color)
    """

    def __init__(self, sim, title="TTC Pedestrian Simulation"):
        self.sim = sim
        self.env = sim.env
        self.env_type = self._detect_env_type()

        # Compute canvas dimensions based on environment type
        if self.env_type == 'torus':
            self.canvas_w = PIXEL_SIZE
            self.canvas_h = PIXEL_SIZE
        else:
            # Corridor-based: preserve aspect ratio
            aspect = self.env.corridor_width / self.env.corridor_length
            self.canvas_w = CANVAS_WIDTH
            self.canvas_h = max(int(CANVAS_WIDTH * aspect), 100)

        # Compute scaling factors
        if self.env_type == 'torus':
            self.sx = PIXEL_SIZE / self.env.s
            self.sy = PIXEL_SIZE / self.env.s
        else:
            self.sx = self.canvas_w / self.env.corridor_length
            self.sy = self.canvas_h / self.env.corridor_width

        # GUI state
        self.paused = False
        self.step_once = False
        self.quit = False
        self.draw_vels = DRAW_VELS
        self.iteration = 0

        # --- Tkinter setup ----------------------------------------------------
        self.win = Tk()
        self.win.title(title)
        self.canvas = Canvas(self.win,
                             width=self.canvas_w, height=self.canvas_h,
                             background="#444")
        self.canvas.pack()

        # Draw walls based on environment type
        self._draw_walls()

        # Create canvas items for each agent
        self.circles = []
        self.vel_lines = []
        self.gvel_lines = []

        for i in range(self.sim.num):
            color = self._get_agent_color(i)
            circ = self.canvas.create_oval(0, 0, 0, 0, fill=color)
            vl = self.canvas.create_line(0, 0, 0, 0, fill="red")
            gl = self.canvas.create_line(0, 0, 0, 0, fill="green")
            self.circles.append(circ)
            self.vel_lines.append(vl)
            self.gvel_lines.append(gl)

        # Key bindings
        self.win.bind("<space>", self._on_key)
        self.win.bind("s", self._on_key)
        self.win.bind("<Escape>", self._on_key)
        self.win.bind("v", self._on_key)

        self._print_help()

    def _detect_env_type(self):
        """Detect the environment type from its attributes."""
        env = self.env
        class_name = type(env).__name__

        if class_name == 'TorusEnvironment':
            return 'torus'
        elif class_name == 'NarrowDoorOneGroupEnvironment':
            return 'narrowdoor_onegroup'
        elif class_name == 'NarrowDoorEnvironment':
            return 'narrowdoor'
        elif class_name == 'CorridorEnvironment':
            return 'corridor'
        else:
            # Fallback: detect by attributes
            if hasattr(env, 's') and not hasattr(env, 'corridor_length'):
                return 'torus'
            elif hasattr(env, 'wall_x'):
                if hasattr(env, 'num_agents'):
                    return 'narrowdoor_onegroup'
                return 'narrowdoor'
            elif hasattr(env, 'corridor_length'):
                return 'corridor'
            return 'unknown'

    def _get_agent_color(self, agent_idx):
        """Get the color for an agent based on environment type and group."""
        if self.env_type == 'torus':
            # Torus: agent 0 is fast agent (pink), others white
            return "#FAA" if agent_idx == 0 else "white"
        elif self.env_type == 'narrowdoor_onegroup':
            # Single group: all blue
            return "#4488FF"
        else:
            # Corridor / narrowdoor: color by group
            if hasattr(self.env, 'group'):
                return "#4488FF" if self.env.group[agent_idx] == 0 else "#FF8844"
            return "white"

    def _draw_walls(self):
        """Draw walls based on environment type."""
        if self.env_type == 'torus':
            # No walls for torus
            return

        # All corridor-based environments have top/bottom walls
        # Top wall (gray bar at top of canvas)
        self.canvas.create_rectangle(
            0, 0, self.canvas_w, WALL_THICKNESS,
            fill="#888", outline="")

        # Bottom wall (gray bar at bottom of canvas)
        self.canvas.create_rectangle(
            0, self.canvas_h - WALL_THICKNESS, self.canvas_w, self.canvas_h,
            fill="#888", outline="")

        # NarrowDoor environments have a middle wall with door gap
        if self.env_type in ('narrowdoor', 'narrowdoor_onegroup'):
            self._draw_middle_wall_with_door()

    def _draw_middle_wall_with_door(self):
        """Draw the middle wall with door gap for narrowdoor environments."""
        env = self.env

        # Wall is at x = wall_x, door from y=door_bottom to y=door_top
        # Note: Tkinter y=0 is at TOP of canvas, so:
        #   - door_bottom_px is closer to TOP of canvas (smaller y in Tkinter)
        #   - door_top_px is closer to BOTTOM of canvas (larger y in Tkinter)
        wall_x_px = self.sx * env.wall_x
        wall_half_width = WALL_THICKNESS / 2

        door_bottom_px = self.sy * env.door_bottom  # Tkinter y near top
        door_top_px = self.sy * env.door_top        # Tkinter y near bottom

        # Upper wall segment (Tkinter: from top corridor wall to door opening)
        self.canvas.create_rectangle(
            wall_x_px - wall_half_width, WALL_THICKNESS,
            wall_x_px + wall_half_width, door_bottom_px,
            fill="#888", outline="")

        # Lower wall segment (Tkinter: from door opening to bottom corridor wall)
        self.canvas.create_rectangle(
            wall_x_px - wall_half_width, door_top_px,
            wall_x_px + wall_half_width, self.canvas_h - WALL_THICKNESS,
            fill="#888", outline="")

        # The door gap between door_bottom_px and door_top_px is left
        # transparent (shows background), allowing agents to pass through

    def _draw_world(self):
        """Redraw all agents, velocity arrows, and goal-velocity arrows."""
        sx = self.sx
        sy = self.sy
        rad = self.sim.rad

        for i in range(self.sim.num):
            cx = self.sim.pos[i, 0]
            cy = self.sim.pos[i, 1]
            vx = self.sim.vel[i, 0]
            vy = self.sim.vel[i, 1]
            gx = self.sim.gvel[i, 0]
            gy = self.sim.gvel[i, 1]

            # Agent circle
            self.canvas.coords(
                self.circles[i],
                sx * (cx - rad), sy * (cy - rad),
                sx * (cx + rad), sy * (cy + rad))

            # Current velocity arrow (red)
            self.canvas.coords(
                self.vel_lines[i],
                sx * cx, sy * cy,
                sx * (cx + rad * vx), sy * (cy + rad * vy))

            # Goal velocity arrow (green)
            self.canvas.coords(
                self.gvel_lines[i],
                sx * cx, sy * cy,
                sx * (cx + rad * gx), sy * (cy + rad * gy))

            # Show / hide velocity arrows
            state = "normal" if self.draw_vels else "hidden"
            self.canvas.itemconfigure(self.vel_lines[i], state=state)
            self.canvas.itemconfigure(self.gvel_lines[i], state=state)

    def _on_key(self, event):
        if event.keysym == "space":
            self.paused = not self.paused
        elif event.keysym == "s":
            self.step_once = True
            self.paused = False
        elif event.keysym == "v":
            self.draw_vels = not self.draw_vels
        elif event.keysym == "Escape":
            self.quit = True

    def _frame(self):
        if self.iteration > MAX_ITERATIONS or self.quit:
            print(f"{self.iteration} iterations ran ... quitting")
            self.win.destroy()
            return

        if not self.paused:
            self.sim.step()
            self.iteration += 1

        self._draw_world()

        if self.step_once:
            self.step_once = False
            self.paused = True

        self.win.after(FRAME_DELAY, self._frame)

    def run(self):
        self.win.after(FRAME_DELAY, self._frame)
        mainloop()

    def _print_help(self):
        print()
        print(f"TTC Pedestrian Simulation ({self.env_type})")
        print()

        if self.env_type == 'torus':
            print("Agents move on a flat 2D torus (periodic in both x and y).")
            print("Pink agent (agent 0) moves faster than others.")
        elif self.env_type == 'corridor':
            print("Two groups walk in opposite directions through a corridor.")
            print("Blue agents walk left-to-right, orange agents walk right-to-left.")
        elif self.env_type == 'narrowdoor':
            print("Two groups walk in opposite directions through a corridor")
            print("with a wall in the middle containing a narrow door.")
            print("Blue agents walk left-to-right, orange agents walk right-to-left.")
        elif self.env_type == 'narrowdoor_onegroup':
            print("All agents walk left-to-right through a corridor with a wall")
            print("in the middle containing a narrow door (single-file bottleneck).")

        print()
        print("Agents avoid collisions using the TTC anticipatory force model.")
        print("Green arrow = goal velocity, Red arrow = current velocity.")
        print()
        print("Controls:")
        print("  SPACE   pause / resume")
        print("  S       step one frame (then pause)")
        print("  V       toggle velocity arrows on/off")
        print("  Escape  quit")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TTC pedestrian simulation (supports all environment types).')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML config file (e.g. config/torus.yaml, config/corridor.yaml)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    env, physics = load_config(args.config)
    sim = TTCSimulation(env=env, seed=args.seed, **physics)

    # Generate title based on environment type
    env_type = type(env).__name__.replace('Environment', '')
    title = f"TTC Simulation - {env_type}"

    print(f"Loaded config: {args.config}")
    print(f"Environment: {type(env).__name__}")
    print(f"Agents: {sim.num}")

    if hasattr(env, 's'):
        print(f"Torus size: {env.s} x {env.s} m")
    if hasattr(env, 'corridor_length'):
        print(f"Corridor: {env.corridor_length} x {env.corridor_width} m")
    if hasattr(env, 'door_width'):
        print(f"Door width: {env.door_width} m at x = {env.wall_x} m")

    TTCVisualization(sim, title=title).run()


if __name__ == "__main__":
    main()
