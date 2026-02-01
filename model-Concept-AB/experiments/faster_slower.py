#Work in Progress
import sys
import os
# ERROR FIX: Ensure we can import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import numpy as np
import matplotlib.pyplot as plt
from model.engine import Engine

def run_experiment():
    print("Running 'Faster is Slower' Experiment...")
    config_path = os.path.join(os.path.dirname(__file__), '../configs/corridor_empty.yaml')
    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
        
    speeds = np.arange(0.5, 5.0, 0.5)
    # Using the high rate from your updated config
    injection_rate = base_cfg["spawn"]["rate"] 
    
    results = []
    
    for v in speeds:
        cfg = base_cfg.copy()
        cfg['agent']['desired_speed'] = float(v)
        cfg['agent']['max_speed'] = float(v * 1.3)
        cfg['spawn']['rate'] = injection_rate
        # Ensure buffer is large enough
        cfg['spawn']['max_agents'] = 2000 
        
        engine = Engine(cfg)
        entry_times = {}
        durations = []
        target_exits = 200
        
        # Run loop
        for step in range(20000):
            t = engine.time
            # Fast access to active IDs
            active_ids = engine.ids[engine.active]
            
            # Track Entries
            for aid in active_ids:
                if aid not in entry_times:
                    entry_times[aid] = t
            
            ids_pre = set(active_ids)
            engine.step()
            ids_post = set(engine.ids[engine.active])
            
            # Track Exits
            exited = ids_pre - ids_post
            for aid in exited:
                if aid in entry_times:
                    durations.append(t - entry_times[aid])
            
            if len(durations) >= target_exits:
                break
                
        if len(durations) > 20:
            avg = np.mean(durations[20:])
            print(f"Speed {v:.1f} m/s: {avg:.3f}s")
            results.append(avg)
        else:
            print(f"Speed {v:.1f} m/s: Jammed")
            results.append(np.nan)
            
    plt.plot(speeds, results, 'o-', color='red', lw=2)
    plt.title("Faster-is-Slower Effect")
    plt.xlabel("Desired Speed (m/s)")
    plt.ylabel("Avg Evacuation Time (s)")
    plt.grid(True)
    plt.savefig("faster_slower_result.png")
    plt.show()
import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt


def _import_engine():
    try:
        from model.engine import Engine  # type: ignore
        return Engine
    except Exception:
        pass

    try:
        from engine import Engine  # type: ignore
        return Engine
    except Exception:
        pass

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        from model.engine import Engine  # type: ignore
        return Engine
    except Exception:
        from engine import Engine  # type: ignore
        return Engine


Engine = _import_engine()


def _find_config_path():
    here = os.path.dirname(__file__)
    candidates = [
        os.path.join(here, "corridor_empty.yaml"),
        os.path.join(here, "../configs/corridor_empty.yaml"),
        os.path.join(here, "../config/corridor_empty.yaml"),
        os.path.join(here, "../corridor_empty.yaml"),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find corridor_empty.yaml. Tried: "
        + ", ".join(os.path.abspath(c) for c in candidates)
    )


def run_experiment():
    print("Running 'Faster is Slower' Experiment...")

    config_path = _find_config_path()
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    speeds = np.arange(0.5, 5.0, 0.5)
    injection_rate = float(base_cfg["spawn"]["rate"])

    results = []

    for v in speeds:
        cfg = {
            **base_cfg,
            "agent": dict(base_cfg["agent"]),
            "spawn": dict(base_cfg["spawn"]),
        }

        cfg["agent"]["desired_speed"] = float(v)
        cfg["agent"]["max_speed"] = float(v * 1.3)
        cfg["spawn"]["rate"] = injection_rate
        cfg["spawn"]["max_agents"] = 2000

        engine = Engine(cfg)

        entry_times = {}
        durations = []
        target_exits = 200

        for _ in range(20000):
            t = engine.time

            active_ids = engine.ids[engine.active]
            for aid in active_ids:
                if aid not in entry_times:
                    entry_times[aid] = t

            ids_pre = set(active_ids)
            engine.step()
            ids_post = set(engine.ids[engine.active])

            exited = ids_pre - ids_post
            for aid in exited:
                if aid in entry_times:
                    durations.append(t - entry_times[aid])

            if len(durations) >= target_exits:
                break

        if len(durations) > 20:
            avg = float(np.mean(durations[20:]))
            print(f"Speed {v:.1f} m/s: {avg:.3f}s")
            results.append(avg)
        else:
            print(f"Speed {v:.1f} m/s: Jammed")
            results.append(np.nan)

    plt.plot(speeds, results, "o-", lw=2)
    plt.title("Faster-is-Slower Effect")
    plt.xlabel("Desired Speed (m/s)")
    plt.ylabel("Avg Evacuation Time (s)")
    plt.grid(True)
    plt.savefig("faster_slower_result.png")
    plt.show()


if __name__ == "__main__":
    run_experiment()

if __name__ == "__main__":
    run_experiment()