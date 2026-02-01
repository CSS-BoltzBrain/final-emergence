# Final Emergence - Complex System Simulation Project (Team 10, 2025)
This repository is the code base used for grading and presentation.

The presentation slides can be found [here (Google Presentation link)](https://docs.google.com/presentation/d/14hL5iyFlzlP5nZoBibPYshgV_cjp7G_TS_9RRHDuvAc/edit?pli=1&slide=id.g3c21385d9cb_0_0#slide=id.g3c21385d9cb_0_0)
If the above link on the internet does not work, you can find the static presentation slides [here](presentation/pedestrian_dynamics.pdf) as well.


# File Hierarchy
Our project is composed of a few different models and sub-projects.
The folder names are self-descriptive.
They are (in the order presented in the presentation slides):

- model_random_probability
- model-Concept-AB
- model-universal-power-law
- large-scale-simulation



The below are more logistics folders and files:
- **presentation**: where you can find the static file of the presentation slides.
- **data**: where you can figure out how to access data we use.
- README.md
- requirements.txt


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

Reproduced figures used in the presentation with the below commands. You need to go to `model-universal-power-law` first:

```
./reproduce-exp.sh
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

For more details, including how GenAI is used for this model implementation, please see [this README](model-universal-power-law/README.md) of this model sub-project.

# Model: Large-Scale Supermarket Crowd Simulation

This model scales to hundreds of agents with optimized updates and checkpointed visualization. You can run either a single baseline simulation or a probability sweep.

Quickstart (from the repository root):

```
python large-scale-simulation/src/main.py large-scale-simulation/configs/empty.yaml large-scale-simulation/scratch 1000 --mode baseline --probability 0.1 --num-agents 96
```

Run a sweep across multiple probabilities:

```
python large-scale-simulation/src/main.py large-scale-simulation/configs/surround.yaml large-scale-simulation/scratch 1000 --mode sweep --num-agents 200
```

Visualize a saved run (creates an .mp4 next to the .npy checkpoint file):

```
python large-scale-simulation/src/visualize.py large-scale-simulation/scratch/simulation_1000_0.1.npy
```

Note: saving videos requires ffmpeg installed on your system.

For more details, including information about the use of GenAI for this model, please see [this README](large-scale-simulation/README.md) of this model sub-project.
