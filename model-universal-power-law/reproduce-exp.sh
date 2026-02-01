#!/usr/bin/env bash

# power law of internal energy and time-to-collision
python ttc_headless_simulation.py --seed 42 --env-size 15 --sight 4 --config ./config/torus.yaml --output data-presentation

python plot_powerlaw.py --compare --fit --fit-all \
    --data data-presentation/tau_torus_data.csv \
    --scrambled data-presentation/tau_torus_scrambled.csv \
    --output data-presentation/comparison_E_tau_torus.png

xdg-open data-presentation/comparison_E_tau_torus.png

# power law of cluster distribution
python ttc_headless_simulation.py --seed 42 --config config/corridor-cluster.size.yaml --track-clusters

# power law of jam duration
python ttc_headless_simulation.py --seed 42 --config config/corridor-jam.yaml --track-jams
