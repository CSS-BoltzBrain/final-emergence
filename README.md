# Final Emergence - Complex System Simulation Project (Team 10, 2025)
This repository is the code base used for grading and presentation.

The presentation slides can be found here: [file placeholder]

# File Hierarchy
- model-random-probability
- model-AB
- model-unversal-power-law
- project-longshot-hpc-supermarket

# Model: Random probability

To reproduce the figures in the presentation:

- Animation in slide 3:
    ```
    python model_random_probability/bottleneck.py
    ```
    If you flag save as True, the final animation is saved as bottleneck.gif in results

- Animation in slide 5:
    ```
    python model_random_probability/clusters.py
    ```
    If you flag save as True, the final animation is saved as clusters.gif in results

- Heatmap in slide 6:
    Beware this will take some time to run
    ```
    python model_random_probability/clusters_analysis.py --heatmap=True
    ```
    If you flag save as True, the final animation is saved as heatmap.png in results

- Powerlaw in slide 6:
    ```
    python model_random_probability/clusters_analysis.py --powerlaw=True
    ```
    If you flag save as True, the final animation is saved as powerlaw.png in results

Variables you can change:
| Parameter | Command | Default | Description |
| :--- | :--- | :--- | :--- |
| **Population** | `--people` | `300` or `6000` | Total number of pedestrians in the system. |
| **Grid Width** | `--width` | `30` or `100` | The horizontal size of the simulation area. |
| **Grid Height** | `--height` | `30` or `100` | The vertical size of the simulation area. |
| **Persistence** | `--p` | `0.05` | Probability of a pedestrian changing direction randomly. |
| **Timesteps** | `--steps` | `400` | Max. number of timesteps. |
| **Runs** | `--runs` | `20` | Number of runs. |
| **Save** | `--save`| `False` | Save the animation or png. |

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
