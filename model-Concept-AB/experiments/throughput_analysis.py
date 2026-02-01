import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt


def _import_engine():
    # 1) package style
    try:
        from model.engine import Engine  # type: ignore
        return Engine
    except Exception:
        pass

    # 2) same folder style
    try:
        from engine import Engine  # type: ignore
        return Engine
    except Exception:
        pass

    # 3) parent folder on path (scripts/ style)
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


def measure_throughput(injection_rate):
    config_path = _find_config_path()
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["spawn"]["rate"] = float(injection_rate)

    engine = Engine(cfg)

    t_start, t_end = 20.0, 50.0
    duration = t_end - t_start
    exits = 0

    while engine.time < t_end:
        n_before = len(engine.agents)
        engine.step()
        n_after = len(engine.agents)

        if engine.time >= t_start:
            diff = n_before - n_after
            if diff > 0:
                exits += diff

    return exits / duration


def find_critical_rate():
    print("Running Throughput Saturation Analysis...")
    print(f"{'Input Rate':<12} | {'Output Flux':<12} | {'Efficiency':<10}")
    print("-" * 40)

    rates = np.arange(1.0, 16.0, 1.0)
    fluxes = []

    for r in rates:
        j = measure_throughput(r)
        fluxes.append(j)
        efficiency = j / r if r > 0 else 0.0
        print(f"{r:<12.1f} | {j:<12.3f} | {efficiency:.2f}")

    max_flux = float(np.nanmax(fluxes))
    threshold = 0.95 * max_flux
    critical_rate = float(rates[-1])

    for r, j in zip(rates, fluxes):
        if j >= threshold:
            critical_rate = float(r)
            break

    print("-" * 40)
    print(f"MAX CAPACITY: {max_flux:.2f} agents/s")
    print(f"CRITICAL INJECTION RATE (Start of Saturation): {critical_rate} agents/s")

    with open("critical_rate.txt", "w") as f:
        f.write(str(critical_rate))

    plt.figure(figsize=(8, 6))
    plt.plot(rates, fluxes, "o-", linewidth=2, label="Measured Flux")
    plt.plot(rates, rates, "k--", alpha=0.3, label="Ideal (Free Flow)")
    plt.axvline(critical_rate, linestyle="--", label=f"Critical Rate ({critical_rate})")
    plt.axhline(max_flux, linestyle=":", label=f"Capacity ({max_flux:.2f})")
    plt.xlabel("Injection Rate [agents/s]")
    plt.ylabel("Exit Throughput [agents/s]")
    plt.title("Crowd Flow Saturation Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("throughput_saturation.png")
    plt.show()


if __name__ == "__main__":
    find_critical_rate()
