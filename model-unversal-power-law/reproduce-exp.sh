#!/usr/bin/env bash
python ttc_headless_simulation.py --seed 42 --env-size 15 --sight 4 --config ./config/torus.yaml --output data-presentation

python plot_powerlaw.py --compare --fit --fit-all \
    --data data-presentation/tau_torus_data.csv \
    --scrambled data-presentation/tau_torus_scrambled.csv \
    --output data-presentation/comparison_E_tau_torus.png

xdg-open data-presentation/comparison_E_tau_torus.png

