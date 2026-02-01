# Model: Universal Power Law Between Internal Energy and Time-to-collision
The implementation is based on K-S-G's sample code.
You can check in the sample code and the corresponding data used to validate our own implementation.

## Reference
- “Universal Power Law Governing Pedestrian Interactions”, PRL 113, 238701 (2014), Ioannis Karamouzas, Brian Skinner, and Stephen J. Guy
- “Quantitative Sociodynamics”, 2nd Edition, Dirk Helbing

# Data
The real world data provided by Skinner et.al. to reproduce their paper can be downloaded from their website. Here is [the link](https://motion.cs.umn.edu/PowerLaw/dl/data.zip), and you can also browser [their website](https://motion.cs.umn.edu) to have a more comprehensive view.

We reproduce the paper by checking the "outdoor" data with our "torus" simulation.
It is optional to download their data if you do not want to reproduce the simulation based on their data, which we use it simply for validation. See "Reproduce the Simulation" section below for more details.

# Reproduce the Simulation
Invoke this bash script to reproduce the simulation. Note: you can feel free to ignore the warning message regarding `KSG Recomputed (Outdoor) files not found`. The plotting script will skip to put their data on their power law fitting plot. You can still see the result of our simulation.

```
$ ./reproduce-exp.sh

<skipped pytest output>

Loaded config: ./config/torus.yaml
Environment: TorusEnvironment (torus)
Torus size: 15.0 x 15.0 m
Agents: 18
Running 10000 iterations...
  iteration 1000/10000 (active: 18)
  iteration 2000/10000 (active: 18)
  iteration 3000/10000 (active: 18)
  iteration 4000/10000 (active: 18)
  iteration 5000/10000 (active: 18)
  iteration 6000/10000 (active: 18)
  iteration 7000/10000 (active: 18)
  iteration 8000/10000 (active: 18)
  iteration 9000/10000 (active: 18)
  iteration 10000/10000 (active: 18)
Time-to-collision data: 5898 values -> data-presentation/tau_torus_data.csv
Scrambled data:         47789 values -> data-presentation/tau_torus_scrambled.csv

To plot results:
  python plot_powerlaw.py --data data-presentation/tau_torus_data.csv \
      --scrambled data-presentation/tau_torus_scrambled.csv \
      --fit --output data-presentation/torus_powerlaw.png
Warning: data-KSG/Outdoor_E_ttc.csv not found, skipping
Warning: KSG Recomputed (Outdoor) files not found ([Errno 2] No such file or directory: 'data-KSG/Outdoor_ttc.csv'), skipping
Loaded My Simulation: 63 points

=== Power Law Fit: My Simulation ===
Calculating best minimal value for power law fit
Fitting xmin: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 57/57 [00:00<00:00, 845.68it/s]
  Fitted alpha (exponent): 2.2987
  xmin: 0.7925
  Power law vs exponential: R=-6.8107, p=0.0004
  Power law vs lognormal:   R=-7.3409, p=0.0377
===================================

Plot saved to data-presentation/comparison_E_tau_torus.png

<skipped more similar output messages>

```

# GenAI Usage Description
We usually draft a software specification, and then ask for GenAI to proofread
and extend the specification based on their best knowledge as a computational
scientist and software developer with a lof of best practice knowledge.

Then we validate GenAI's work firstly by animation due to the nature of our simulation: agents moving,
which animation could be a powerful tool to have a quick check.

If the animation shows the overall behavior "seems right",
we started investigating time to iterate the development process
by test cases designed with specific scenarios owned by the model or corner cases.

The first shot of Gen AI's work is very critical to ensure the following iteration can lead the implementation goes well.
By "goes will" we mean less redundant if conditions since GenAI prefers to deliver works looking right rather than always following the best practice of software architecture.

## The Specification
The specification usually contains:
- how the simulation tool looks like. usually we choose a command line tool.
- software architecture
- what model and what algorithm we want to implement

### Command line specification
- it is a command line tool
- what are the parameters and options the command line tool should have
- the command line would consume a configuration file in yaml format
- the yaml configuration file describes simulation parameters
- the values of the options of the command line can override the configuration values, if specified.

### Software Architecture
Usually we started with MVP or something similar to ensure we can deliver visualization including animation, and export simulation data easily for post-processing.

MVP has lower computation performance, but it pays off for such 2-week time intensive project.


## Additional Refection
For this project, we made a lot of different prototypes based on different models.
By such unified approach to make the first shot with GenAI's help.
We can make ephemeral model implementation and test it in a short time.
