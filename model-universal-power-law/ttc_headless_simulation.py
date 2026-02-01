"""
Unified headless simulation runner for TTC pedestrian models.

Supports all environment types defined in environment.py:
  - torus: flat 2D torus (periodic in both x and y)
  - corridor: rectangular corridor (periodic x, walled y)
  - narrowdoor: corridor with a wall in the middle containing a narrow door
  - narrowdoor_onegroup: same as narrowdoor but with a single group

Exports time-to-collision data:
  - tau_<env_type>_data.csv:      TTC for simultaneously-present agent pairs
  - tau_<env_type>_scrambled.csv: scrambled TTC (non-interacting baseline)

The scrambled data is produced by pairing agent states from different timesteps,
so that the paired agents were not simultaneously present and therefore not
interacting. This follows the methodology of Karamouzas, Skinner & Guy,
PRL 113, 238701 (2014).

Usage:
    python run_simulation.py --config config/torus.yaml
    python run_simulation.py --config config/corridor.yaml
    python run_simulation.py --config config/narrowdoor.yaml
    python run_simulation.py --config config/narrowdoor_onegroup.yaml

    # Custom output directory:
    python run_simulation.py --config config/torus.yaml --output-dir my_data

Plotting results:
    python plot_powerlaw.py --data <output_dir>/tau_<type>_data.csv \\
                            --scrambled <output_dir>/tau_<type>_scrambled.csv \\
                            --fit --output <output_dir>/<type>_powerlaw.png
"""

import argparse
import json
import os
import csv

from ttc_engine import TTCSimulation
from environment import load_config
from jam_tracker import JamTracker
from cluster_tracker import ClusterTracker


def detect_env_type(env):
    """Detect the environment type from its class name."""
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
        return 'unknown'


def get_default_output_dir(env_type):
    """Get the default output directory for an environment type."""
    return {
        'torus': 'data-torus',
        'corridor': 'data-corridor',
        'narrowdoor': 'data-corridor-narrowdoor',
        'narrowdoor_onegroup': 'data-corridor-narrowdoor_onegroup',
    }.get(env_type, 'data')


def get_file_prefix(env_type):
    """Get the file name prefix for an environment type."""
    return {
        'torus': 'tau_torus',
        'corridor': 'tau_corridor',
        'narrowdoor': 'tau_narrowdoor',
        'narrowdoor_onegroup': 'tau_narrowdoor_onegroup',
    }.get(env_type, 'tau')


def run_simulation(sim, num_iterations, sample_interval,
                   jam_tracker=None, cluster_tracker=None):
    """Run the simulation and collect time-to-collision data.

    Args:
        sim: TTCSimulation instance
        num_iterations: Number of timesteps to run
        sample_interval: Record TTC every N iterations
        jam_tracker: Optional JamTracker instance for tracking jam durations
        cluster_tracker: Optional ClusterTracker instance for tracking cluster sizes

    Returns:
        (all_ttc, scrambled_ttc) tuple of TTC value lists
    """
    # Storage for snapshots of (positions, velocities) at sampled frames
    snapshots = []
    all_ttc = []

    for iteration in range(num_iterations):
        sim.step()

        # Update jam tracker if enabled (every timestep)
        if jam_tracker is not None:
            jam_tracker.update(sim.pos, sim.active)

        if iteration % sample_interval == 0:
            snapshots.append((sim.pos.copy(), sim.vel.copy()))
            ttc_values = sim.compute_all_pairwise_ttc()
            all_ttc.extend(ttc_values)

            # Update cluster tracker at sample interval
            if cluster_tracker is not None:
                cluster_tracker.update(sim.pos, sim.active)

        if (iteration + 1) % 1000 == 0:
            active_count = sim.active.sum()
            status = f"  iteration {iteration + 1}/{num_iterations}"
            if jam_tracker is not None:
                jam_count = jam_tracker.get_current_jam_count()
                status += f" (active: {active_count}, jammed: {jam_count})"
            else:
                status += f" (active: {active_count})"
            print(status)

        # Stop early if no agents remain
        if sim.active.sum() == 0:
            print(f"  All agents removed at iteration {iteration + 1}")
            break

    # Finalize jam tracker to capture ongoing jams
    if jam_tracker is not None:
        jam_tracker.finalize()

    # Compute scrambled time-to-collision data.
    #
    # The pair distribution function g(tau) requires a non-interacting
    # baseline.  Following the paper, this baseline is obtained by computing
    # time-to-collision between pairs of pedestrians that are not
    # simultaneously present in the scene.
    #
    # We approximate non-simultaneously-present pairs by pairing agent i's
    # state at frame f1 with agent j's state at a *different* frame f2.
    scrambled_ttc = sim.compute_scrambled_ttc(snapshots)

    return all_ttc, scrambled_ttc


def write_csv(values, filepath):
    """Write tau values to a single-column CSV with header 'tau'."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tau'])
        for v in values:
            writer.writerow([v])


def print_env_info(env, env_type):
    """Print environment-specific information."""
    if env_type == 'torus':
        print(f"Torus size: {env.s} x {env.s} m")
    elif env_type in ('corridor', 'narrowdoor', 'narrowdoor_onegroup'):
        print(f"Corridor: {env.corridor_length} x {env.corridor_width} m")
        if env_type in ('narrowdoor', 'narrowdoor_onegroup'):
            print(f"Door width: {env.door_width} m at x = {env.wall_x} m")


def print_agent_info(sim, env_type):
    """Print agent count information based on environment type."""
    if env_type == 'torus':
        print(f"Agents: {sim.num}")
    elif env_type == 'narrowdoor_onegroup':
        print(f"Agents: {sim.num} (single group)")
    else:
        print(f"Agents: {sim.num} ({sim.num // 2} per group)")


def main():
    parser = argparse.ArgumentParser(
        description='Run headless TTC pedestrian simulation and export '
                    'time-to-collision data. Supports all environment types.')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML config file (e.g. config/torus.yaml, '
                             'config/corridor.yaml, config/narrowdoor.yaml)')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Number of simulation iterations (default: 10000)')
    parser.add_argument('--sample-interval', type=int, default=5,
                        help='Record TTC every N iterations (default: 5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-detected from env type)')
    parser.add_argument('--env-size', type=float, default=None,
                        help='Override environment size (torus side length)')
    parser.add_argument('--sight', type=float, default=None,
                        help='Override sight range for neighbor detection')
    parser.add_argument('--track-jams', action='store_true',
                        help='Enable jam duration tracking')
    parser.add_argument('--jam-dt-window', type=float, default=1.0,
                        help='Time window for jam detection in seconds (default: 1.0)')
    parser.add_argument('--jam-threshold', type=float, default=None,
                        help='Displacement threshold for jam detection in meters '
                             '(default: 2 * agent radius = 0.4m)')
    parser.add_argument('--track-clusters', action='store_true',
                        help='Enable cluster size tracking')
    parser.add_argument('--cluster-threshold', type=float, default=None,
                        help='Distance threshold for cluster connectivity in meters '
                             '(default: 3 * agent radius = 0.6m)')
    args = parser.parse_args()

    # Load configuration and create simulation
    env, physics = load_config(args.config,
                               env_size=args.env_size,
                               sight=args.sight)
    sim = TTCSimulation(env=env, seed=args.seed, **physics)

    # Detect environment type
    env_type = detect_env_type(env)

    # Determine output directory
    output_dir = args.output_dir or get_default_output_dir(env_type)

    # Print configuration info
    print(f"Loaded config: {args.config}")
    print(f"Environment: {type(env).__name__} ({env_type})")
    print_env_info(env, env_type)
    print_agent_info(sim, env_type)
    if args.track_jams:
        jam_threshold = args.jam_threshold or (2 * sim.rad)
        print(f"Jam tracking: enabled (window={args.jam_dt_window}s, "
              f"threshold={jam_threshold}m)")
    if args.track_clusters:
        cluster_threshold = args.cluster_threshold or (3 * sim.rad)
        print(f"Cluster tracking: enabled (threshold={cluster_threshold}m)")
    print(f"Running {args.iterations} iterations...")

    # Create jam tracker if enabled
    jam_tracker = None
    if args.track_jams:
        jam_threshold = args.jam_threshold or (2 * sim.rad)
        jam_tracker = JamTracker(
            num_agents=sim.num,
            dt=sim.dt,
            dt_window=args.jam_dt_window,
            threshold=jam_threshold,
        )

    # Create cluster tracker if enabled
    cluster_tracker = None
    if args.track_clusters:
        cluster_threshold = args.cluster_threshold or (3 * sim.rad)
        cluster_tracker = ClusterTracker(
            num_agents=sim.num,
            cluster_threshold=cluster_threshold,
            env=env,
        )

    # Run simulation
    all_ttc, scrambled_ttc = run_simulation(
        sim=sim,
        num_iterations=args.iterations,
        sample_interval=args.sample_interval,
        jam_tracker=jam_tracker,
        cluster_tracker=cluster_tracker,
    )

    # Write output files
    file_prefix = get_file_prefix(env_type)
    data_path = os.path.join(output_dir, f'{file_prefix}_data.csv')
    scrambled_path = os.path.join(output_dir, f'{file_prefix}_scrambled.csv')

    write_csv(all_ttc, data_path)
    write_csv(scrambled_ttc, scrambled_path)

    print(f"Time-to-collision data: {len(all_ttc)} values -> {data_path}")
    print(f"Scrambled data:         {len(scrambled_ttc)} values -> {scrambled_path}")

    # Write jam duration data if tracking was enabled
    if jam_tracker is not None:
        jam_durations = jam_tracker.get_completed_jam_events()
        jam_path = os.path.join(output_dir, 'jam_durations.csv')
        os.makedirs(output_dir, exist_ok=True)
        with open(jam_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['duration'])
            for d in jam_durations:
                writer.writerow([d])

        # Write metadata
        meta_path = os.path.join(output_dir, 'jam_durations_meta.json')
        stats = jam_tracker.get_statistics()
        meta = {
            'dt': sim.dt,
            'dt_window': args.jam_dt_window,
            'threshold': args.jam_threshold or (2 * sim.rad),
            'total_jam_events': stats['total_events'],
            'mean_duration': stats['mean_duration'],
            'max_duration': stats['max_duration'],
            'total_simulation_time': args.iterations * sim.dt,
            'config_file': args.config,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Jam durations:          {len(jam_durations)} events -> {jam_path}")
        print(f"  Mean duration: {stats['mean_duration']:.2f}s, "
              f"Max: {stats['max_duration']:.2f}s")

    # Write cluster size data if tracking was enabled
    if cluster_tracker is not None:
        cluster_sizes = cluster_tracker.get_all_cluster_sizes()
        cluster_path = os.path.join(output_dir, 'cluster_sizes.csv')
        os.makedirs(output_dir, exist_ok=True)
        with open(cluster_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['size'])
            for s in cluster_sizes:
                writer.writerow([s])

        # Write metadata
        meta_path = os.path.join(output_dir, 'cluster_sizes_meta.json')
        stats = cluster_tracker.get_statistics()
        meta = {
            'cluster_threshold': args.cluster_threshold or (3 * sim.rad),
            'total_clusters': stats['total_clusters'],
            'num_samples': stats['num_samples'],
            'mean_size': stats['mean_size'],
            'max_size': stats['max_size'],
            'total_simulation_time': args.iterations * sim.dt,
            'config_file': args.config,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Cluster sizes:          {len(cluster_sizes)} clusters -> {cluster_path}")
        print(f"  Mean size: {stats['mean_size']:.2f}, Max: {stats['max_size']}")

    print()
    print("To plot results:")
    print(f"  python plot_powerlaw.py --data {data_path} \\")
    print(f"      --scrambled {scrambled_path} \\")
    print(f"      --fit --output {output_dir}/{env_type}_powerlaw.png")

    if jam_tracker is not None:
        print()
        print("To plot jam duration distribution:")
        print(f"  python plot_powerlaw_jam_duration.py --data {jam_path} \\")
        print(f"      --fit --output {output_dir}/jam_powerlaw.png")

    if cluster_tracker is not None:
        print()
        print("To plot cluster size distribution:")
        print(f"  python plot_powerlaw_cluster_size.py --data {cluster_path} \\")
        print(f"      --fit --output {output_dir}/cluster_powerlaw.png")


if __name__ == '__main__':
    main()
