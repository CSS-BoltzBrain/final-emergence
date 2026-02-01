#!/usr/bin/env bash

# ensure everything is still valid
pytest -v

# power law of internal energy and time-to-collision
python ttc_headless_simulation.py --seed 42 --env-size 15 --sight 4 --config ./config/torus.yaml --output data-presentation

python plot_powerlaw.py --compare --fit --fit-all \
    --data data-presentation/tau_torus_data.csv \
    --scrambled data-presentation/tau_torus_scrambled.csv \
    --output data-presentation/comparison_E_tau_torus.png

# if you are using freedesktop you usually have xdg-open
# if not, please open this png with your image viewer
xdg-open data-presentation/comparison_E_tau_torus.png

# power law of cluster distribution
python ttc_headless_simulation.py --seed 42 --config config/corridor-cluster.size.yaml --track-clusters
python plot_powerlaw_cluster_size.py --data data-corridor/cluster_sizes.csv --fit --output data-corridor/cluster_powerlaw.png

xdg-open data-corridor/cluster_powerlaw.png

# power law of jam duration
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! Job takes long time (around 2 hours) !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Disabled by default
#
#python ttc_headless_simulation.py --seed 42 --config config/corridor-jam.yaml --track-jams
#python plot_powerlaw_jam_duration.py --data data-corridor/jam_durations.csv --fit --output data-corridor/jam_powerlaw.png
#
#xdg-open data-corridor/jam_powerlaw.png
