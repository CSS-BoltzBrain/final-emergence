# Final Emergence - Complex System Simulation Project (Team 10, 2025)
This repository is the code base used for grading and presentation.

The presentation slides can be found here: [file placeholder]

# File Hierarchy
- model-random-probability
- model-AB
- model-unversal-power-law
- project-longshot-hpc-supermarket

# Model: Universal Power Law Between Internal Energy and Time-to-collision
## Quickstart

Reproduced figures used in the presentation:

```
model-unversal-power-law$ ./reproduce-exp.sh
```

Run simulation:

```
python ttc_headless_simulation.py --config ./config/torus.yaml
```

Check in the animation of that simulation:

```
python ttc_vis.py --config ./config/torus.yaml
```

You can switch to different configuration to define the dynamics parameters of agents and the enrionment they are moving around.
For example, let us try "corridor" instead of "torus" open space (to mimic the "outdoor" data provided by K-S-G):

```
python ttc_vis.py --config ./config/corridor.yaml
```

For more details, including how GenAI is used for this model implementation, please see [this README](model-unversal-power-law/README.md).
