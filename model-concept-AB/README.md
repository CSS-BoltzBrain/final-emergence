# Pedestrian Counter Flow Simulation (Agent-Based Model)

This project simulates bidirectional pedestrian flow in a corridor using an agent-based model. Agents are injected at a controlled rate, move toward opposing targets, interact through social forces, and exit once they reach their destination on configurable layouts

## Reference
Based on the methodology described in:
> Jicai Dai, Xia Li, Lin Liu, "Simulation of pedestrian counter flow through bottlenecks by using an agent-based model", *Physica A: Statistical Mechanics and its Applications*, Vol 392, Issue 9, 2013, pp. 2202-2211. [DOI: 10.1016/j.physa.2013.01.012](https://doi.org/10.1016/j.physa.2013.01.012)

## Model Summary
- **State**: position `x_i`, velocity `v_i`, radius `r`.
- **Targets**: agents move toward a fixed target point on the opposite side.
- **Spawn**: deterministic injection rate, capped by `max_agents`.
- **Update**: explicit Euler integration with speed capping and overlap resolution.

## Forces (per agent)
Let `x_i` be position, `t_i` target, `v_i` velocity.

Gradient (direction to target):
```
g_i = (t_i - x_i) / ||t_i - x_i||
```

Gradient force:
```
f_grad,i = k_g * g_i
```

Repulsive force (within perception radius R):
```
f_rep,i = Σ_j 2 k_r exp(-k_d ||x_i - x_j||^2) (x_i - x_j)
```

Resistive force (crowding drag):
```
f_res,i = -exp(-gamma) f_rep,i
```

Random force (applied with probability p_rand per agent per step):
```
θ_i ~ U(0, 2π)
f_rand,i = k_rand [cos(θ_i), sin(θ_i)]
```

Total force:
```
f_i = f_grad,i + f_rep,i + f_res,i + f_rand,i
```

Density-based speed reduction (local density within R):
```
ρ_i = N_i / (π R^2)
v_cap,i = v_max exp(-α ρ_i)
```

## Speed and Update Process
Per time step `dt`:
```
 v_i <- v_i + f_i * dt
 if ||v_i|| > v_cap,i: v_i <- v_cap,i * v_i / ||v_i||
 x_i <- x_i + v_i * dt
```
Then:
- wall overlaps are projected out,
- inter-agent overlaps are resolved by symmetric separation,
- agents that reach their target are removed.
 
## Artefacts and Limitations
- Gradient handling can introduce discontinuities at cell boundaries or target-switching points.
- The explicit velocity update and hard speed cap can create visible jumps in trajectories.
- The sheer number of parameters affecting pedestrian dynamics in this model means that the choice of parameters greatly affected the "realism" of the simulation. One may question if such hyper tuning of parameters brings forth the emergent behaviour exhibited by this model.

## File Structure

```
model-AB/
├── configs/               # Scenario configs
│   ├── corridor_empty.yaml
│   └── corridor_bottleneck.yaml
├── experiments/
│   ├── simulation.ipynb   #simulation animation lives here
│   ├── faster_slower.py   (WIP)
│   └── throughput_analysis.py
├── model/
│   ├── engine.py
│   ├── agents.py
│   ├── forces.py
│   ├── navigation.py.     #optional
│   └── walls.py
├── util/
│   └── notebook_animation.py
└── requirements.txt
```

## Setup & Running

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Visual Simulation**:
   Open `experiments/simulation.ipynb` in Jupyter Notebook or VSCode.

3. **Quantitative Experiments**:
   ```bash
   python experiments/throughput_analysis.py
   ```

## GENAI Note
GenAI was used for rapid prototyping and for verifying model parameters by developing sample test cases.
