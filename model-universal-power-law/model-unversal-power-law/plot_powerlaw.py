"""
Plot the interaction energy E(tau) on a log-log scale to verify the
power law E(tau) ~ 1/tau^2 from Karamouzas, Skinner & Guy (KGS),
PRL 113, 238701 (2014), Fig. 2b.

HOW THE INTERACTION ENERGY IS CALCULATED
=========================================

1. Pair distribution function g(tau)
   ----------------------------------
   g(tau) is defined as the ratio of the observed probability density for
   a pair of simultaneously-present pedestrians to have time-to-collision
   tau, divided by the expected density for non-interacting pairs:

       g(tau) = P_observed(tau) / P_non-interacting(tau)

   - P_observed(tau) comes from the histogram of time-to-collision values
     for simultaneously-present pedestrian pairs (the --data input).
   - P_non-interacting(tau) comes from the histogram of *scrambled*
     time-to-collision values (the --scrambled input), computed between
     pairs of pedestrians that are not simultaneously present in the scene
     and therefore not interacting.

   When g(tau) < 1, the configuration is less likely than expected under
   non-interaction, indicating avoidance / repulsion.

2. Interaction energy E(tau)
   --------------------------
   From the Boltzmann-like relation (paper Eq. 1):

       g(tau) ~ exp[-E(tau) / E0]

   we obtain the interaction energy:

       E(tau) = -ln(g(tau)) = ln(1 / g(tau))

   E(tau) > 0 in the avoidance regime (g < 1).

3. Expected result
   ----------------
   On a log-log plot, E(tau) vs tau should be a straight line with
   slope -2, i.e. E(tau) ~ tau^{-2} (paper Fig. 2b, Eq. 2).
"""

import argparse
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_ttc_data(filepath):
    """
    Load time-to-collision values from a CSV file.

    Handles both formats:
      - Single column, no header (KGS Outdoor_ttc.csv)
      - Single column with header 'tau' (our simulation output)
    """
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line))
            except ValueError:
                continue  # skip header or non-numeric lines
    return np.array(values)


def load_precomputed_energy(filepath):
    """
    Load precomputed E(tau) from KGS format (e.g. Outdoor_E_ttc.csv).

    Expected: two columns with header 'tau,E(tau)'.
    """
    tau_vals = []
    e_vals = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            try:
                tau_vals.append(float(parts[0]))
                e_vals.append(float(parts[1]))
            except ValueError:
                continue  # skip header
    return np.array(tau_vals), np.array(e_vals)


def compute_energy(ttc_data, ttc_scrambled, num_bins=200,
                   tau_min=None, tau_max=None, verbose=False):
    """
    Compute the interaction energy E(tau) from time-to-collision data
    and scrambled time-to-collision data.

    Pair distribution function (see module docstring for derivation):
        g(tau) = pdf_observed(tau) / pdf_scrambled(tau)

    Interaction energy:
        E(tau) = -ln(g(tau))

    Only bins in the avoidance regime (g > 0 and g < 1, giving E > 0) are
    returned, since those are the values plotted on a log-log scale.

    Args:
        ttc_data:      time-to-collision values (simultaneously-present pairs)
        ttc_scrambled: scrambled time-to-collision values
        num_bins:      number of histogram bins (default: 200)
        tau_min:       minimum tau for binning (default: 0.4 s, matching
                       KSG's precomputed range; below this, histogram bins
                       have too few data counts for reliable g(tau))
        tau_max:       maximum tau for binning (default: auto-detected from
                       the data by finding where g(tau) transitions to ~1)
        verbose:       print diagnostic information (default: False)

    Returns:
        tau_centers: bin centers where E is well-defined and positive
        E_values:    corresponding interaction energy values
    """
    # --- Filter sentinel values ---------------------------------------------
    # compute_ttc() returns 100 for overlapping agents and 999 for no
    # predicted collision.  These are not physical tau values and must be
    # removed before histogram analysis.
    SENTINEL_THRESHOLD = 10.0  # well above any real interaction range
    ttc_data = ttc_data[ttc_data < SENTINEL_THRESHOLD]
    ttc_scrambled = ttc_scrambled[ttc_scrambled < SENTINEL_THRESHOLD]

    if verbose:
        print(f"After filtering sentinels (tau >= {SENTINEL_THRESHOLD}):")
        print(f"  ttc_data:      {len(ttc_data)} values")
        print(f"  ttc_scrambled: {len(ttc_scrambled)} values")

    # --- Determine bin range ------------------------------------------------
    # The paper shows E(tau) is well-defined from ~0.2 s (reaction-time
    # floor) up to t0 (the screening time: ~2.4 s Outdoor, ~1.4 s
    # Bottleneck).  Beyond t0 the interaction vanishes and g(tau) ~ 1,
    # producing noise.
    if tau_min is None:
        tau_min = 0.4
    if tau_max is None:
        # Auto-detect: use a coarse histogram to find where g(tau) first
        # exceeds 0.8 (interaction vanishing).  Fall back to 5 s.
        coarse_max = float(np.percentile(
            np.concatenate([ttc_data, ttc_scrambled]), 80))
        coarse_edges = np.linspace(tau_min, max(coarse_max, 1.0), 100)
        h_d, _ = np.histogram(ttc_data, bins=coarse_edges)
        h_s, _ = np.histogram(ttc_scrambled, bins=coarse_edges)
        cw = np.diff(coarse_edges)
        sum_d = h_d.sum()
        sum_s = h_s.sum()
        p_d = np.where(sum_d > 0, h_d / (sum_d * cw), 0)
        p_s = np.where(sum_s > 0, h_s / (sum_s * cw), 0)
        ok = (h_d >= 5) & (h_s >= 5) & (p_s > 0)
        g_coarse = np.full(len(p_d), np.nan)
        g_coarse[ok] = p_d[ok] / p_s[ok]
        # Find the first bin (from left) where g crosses above 0.8
        centers = (coarse_edges[:-1] + coarse_edges[1:]) / 2
        crossed = np.where(ok & (g_coarse > 0.8))[0]
        if len(crossed) > 0:
            tau_max = float(centers[crossed[-1]])
        else:
            tau_max = 5.0
        tau_max = max(tau_max, 1.0)

    # --- Build fine-grained histogram --------------------------------------
    # Linear bins: give uniform resolution across the interaction range.
    # The paper's Fig. 2b uses a roughly linear binning from ~0.2 to ~3 s.
    bin_edges = np.linspace(tau_min, tau_max, num_bins + 1)
    bin_widths = np.diff(bin_edges)

    hist_data, _ = np.histogram(ttc_data, bins=bin_edges)
    hist_scrambled, _ = np.histogram(ttc_scrambled, bins=bin_edges)

    # Normalize to probability densities
    pdf_data = hist_data / (hist_data.sum() * bin_widths)
    pdf_scrambled = hist_scrambled / (hist_scrambled.sum() * bin_widths)

    # g(tau) = pdf_observed / pdf_scrambled
    # Only require non-empty scrambled bins (division by zero guard).
    valid = (hist_data > 0) & (hist_scrambled > 0)
    g = np.full(len(pdf_data), np.nan)
    g[valid] = pdf_data[valid] / pdf_scrambled[valid]

    # E(tau) = -ln(g) is positive when g < 1 (avoidance regime).
    energy_mask = valid & (g > 0) & (g < 1)

    tau_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if verbose:
        print(f"\n=== Compute Energy Diagnostics ===")
        print(f"Total bins: {num_bins}")
        print(f"Tau range: [{tau_min:.3f}, {tau_max:.3f}]")
        print(f"Valid bins (both histograms > 0): {np.sum(valid)}")
        print(f"\ng(tau) distribution:")
        print(f"  Min: {np.nanmin(g):.4f}")
        print(f"  Max: {np.nanmax(g):.4f}")
        print(f"  Mean: {np.nanmean(g):.4f}")
        print(f"  Bins with g < 1 (avoidance): {np.sum(valid & (g < 1))}")
        print(f"  Bins with g >= 1 (no avoidance): {np.sum(valid & (g >= 1))}")
        print(f"\nFinal E(tau) points (g < 1): {np.sum(energy_mask)}")
        if np.sum(energy_mask) > 0:
            print(f"  Tau range: [{tau_centers[energy_mask].min():.4f}, {tau_centers[energy_mask].max():.4f}]")
        print(f"===================================\n")

    return tau_centers[energy_mask], -np.log(g[energy_mask])


def _filter_positive(tau, E):
    """Filter to keep only positive tau and E values for log-log plotting."""
    pos = (tau > 0) & (E > 0)
    return tau[pos], E[pos]


def _fit_powerlaw(tau, E, label='Data'):
    """
    Fit E(tau) data using the powerlaw package with verbose output.

    Args:
        tau: array of tau values
        E: array of E values
        label: dataset label for verbose output

    Returns:
        dict with keys: slope, alpha, tau_fit, E_fit, R_exp, p_exp, R_ln, p_ln
    """
    import powerlaw

    print(f"\n=== Power Law Fit: {label} ===")
    # Fit using powerlaw package with verbose=True
    pl = powerlaw.Fit(E, verbose=True)
    alpha = pl.alpha
    xmin = pl.xmin

    print(f"  Fitted alpha (exponent): {alpha:.4f}")
    print(f"  xmin: {xmin:.4f}")

    # Compare power law to alternatives
    R_exp, p_exp = pl.distribution_compare('power_law', 'exponential')
    R_ln, p_ln = pl.distribution_compare('power_law', 'lognormal')
    print(f"  Power law vs exponential: R={R_exp:.4f}, p={p_exp:.4f}")
    print(f"  Power law vs lognormal:   R={R_ln:.4f}, p={p_ln:.4f}")
    print(f"{'=' * 35}\n")

    # Generate fit line for plotting (using log-log linear regression for visual)
    log_tau = np.log10(tau)
    log_E = np.log10(E)
    slope, intercept = np.polyfit(log_tau, log_E, 1)
    tau_fit = np.logspace(np.log10(tau.min()), np.log10(tau.max()), 100)
    E_fit = 10 ** intercept * tau_fit ** slope

    return {
        'slope': slope,
        'alpha': alpha,
        'tau_fit': tau_fit,
        'E_fit': E_fit,
        'R_exp': R_exp,
        'p_exp': p_exp,
        'R_ln': R_ln,
        'p_ln': p_ln,
    }


def _reference_line(tau, E):
    """
    Compute a tau^{-2} reference line anchored at the geometric mean.

    Returns:
        tau_ref, E_ref arrays
    """
    tau_ref = np.logspace(np.log10(tau.min()), np.log10(tau.max()), 100)
    geo_mean_tau = np.exp(np.mean(np.log(tau)))
    geo_mean_E = np.exp(np.mean(np.log(E)))
    E_ref = geo_mean_E * (tau_ref / geo_mean_tau) ** (-2)
    return tau_ref, E_ref


def plot_energy(datasets, output_path, fit=False, fit_all=False,
                title='Interaction Energy E(tau)'):
    """
    Create a log-log plot of E(tau) vs tau for one or more datasets.

    Args:
        datasets: list of dicts with keys:
            - 'tau': array of tau values
            - 'E': array of E values
            - 'label': legend label
            - 'color': plot color (optional)
            - 'marker': marker style (optional)
        output_path: path to save the plot
        fit: whether to show fit lines (in compare mode, only fits "My Simulation"
             unless fit_all is True)
        fit_all: in compare mode, fit all datasets (not just "My Simulation")
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['steelblue', 'forestgreen', 'darkorange', 'crimson', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    all_tau = []
    all_E = []
    fit_info = []
    is_compare_mode = len(datasets) > 1

    for i, ds in enumerate(datasets):
        tau_pos, E_pos = _filter_positive(ds['tau'], ds['E'])
        if len(tau_pos) == 0:
            continue

        label = ds.get('label', f'Dataset {i+1}')
        color = ds.get('color', colors[i % len(colors)])
        marker = ds.get('marker', markers[i % len(markers)])

        all_tau.extend(tau_pos)
        all_E.extend(E_pos)

        ax.scatter(tau_pos, E_pos, s=20, alpha=0.6, color=color,
                   marker=marker, label=label)

        # In compare mode, only fit "My Simulation" unless fit_all is True
        # In single mode, always fit
        should_fit = fit and len(tau_pos) > 2
        if is_compare_mode and not fit_all:
            should_fit = should_fit and (label == 'My Simulation')

        if should_fit:
            fit_result = _fit_powerlaw(tau_pos, E_pos, label)
            # Build legend label with R/p statistics
            fit_label = (f'{label} fit: slope={fit_result["slope"]:.2f}\n'
                         f'  vs exp: R={fit_result["R_exp"]:.2f}, p={fit_result["p_exp"]:.3f}\n'
                         f'  vs ln:  R={fit_result["R_ln"]:.2f}, p={fit_result["p_ln"]:.3f}')
            ax.plot(fit_result['tau_fit'], fit_result['E_fit'], color=color,
                    linestyle='-', linewidth=1.5, alpha=0.7, label=fit_label)
            fit_info.append(fit_result)

    # Reference line spanning all data
    if len(all_tau) > 0:
        all_tau = np.array(all_tau)
        all_E = np.array(all_E)
        tau_ref, E_ref = _reference_line(all_tau, all_E)
        ax.plot(tau_ref, E_ref, 'k--', linewidth=2, alpha=0.7,
                label=r'Theory: $E \propto \tau^{-2}$')

    # Build title with fit info for single dataset
    if len(datasets) == 1 and fit_info:
        f = fit_info[0]
        subtitle = (f'slope={f["slope"]:.2f} (expected: -2.0)  |  '
                    f'alpha={f["alpha"]:.2f}')
        title = f'{title}\n{subtitle}'

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau$ [s]', fontsize=12)
    ax.set_ylabel(r'$E(\tau)$ [arb. units]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def _load_dataset(data_path, scrambled_path, label, color, marker,
                  num_bins, tau_min, tau_max, verbose):
    """
    Load and compute E(tau) for a dataset from raw TTC files.

    Returns a dataset dict or None if files not found.
    """
    try:
        ttc_data = load_ttc_data(data_path)
        ttc_scrambled = load_ttc_data(scrambled_path)
        if verbose:
            print(f"\n--- {label} ---")
        tau, E = compute_energy(ttc_data, ttc_scrambled, num_bins=num_bins,
                                tau_min=tau_min, tau_max=tau_max, verbose=verbose)
        if len(tau) > 0:
            print(f"Loaded {label}: {len(tau)} points")
            return {'tau': tau, 'E': E, 'label': label,
                    'color': color, 'marker': marker}
    except FileNotFoundError as e:
        print(f"Warning: {label} files not found ({e}), skipping")
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Plot interaction energy E(tau) on a log-log scale.  '
                    'Reproduces Fig. 2b from KGS PRL 113, 238701 (2014).')

    parser.add_argument('--data', type=str,
                        help='Path to time-to-collision data CSV '
                             '(used as simulation data in --compare mode)')
    parser.add_argument('--scrambled', type=str,
                        help='Path to scrambled time-to-collision data CSV '
                             '(used as simulation scrambled in --compare mode)')
    parser.add_argument('--precomputed', type=str,
                        help='Path to precomputed E(tau) CSV '
                             '(e.g. data-KSG/Outdoor_E_ttc.csv)')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison plot with KSG precomputed, '
                             'KSG recomputed, and simulation data')
    parser.add_argument('--fit', action='store_true',
                        help='Enable power-law fitting')
    parser.add_argument('--fit-all', action='store_true',
                        help='In compare mode, fit all datasets (not just My Simulation)')
    parser.add_argument('--output', type=str, default='data/energy_plot.png',
                        help='Output plot path (default: data/energy_plot.png)')
    parser.add_argument('--bins', type=int, default=200,
                        help='Number of histogram bins (default: 200)')
    parser.add_argument('--tau-min', type=float, default=None,
                        help='Minimum tau for binning in seconds '
                             '(default: 0.4, matching KSG range)')
    parser.add_argument('--tau-max', type=float, default=None,
                        help='Maximum tau for binning in seconds '
                             '(default: auto-detected from data)')
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title')
    parser.add_argument('--verbose', action='store_true',
                        help='Print diagnostic information')

    args = parser.parse_args()

    # Common parameters for compute_energy
    compute_params = {
        'num_bins': args.bins,
        'tau_min': args.tau_min,
        'tau_max': args.tau_max,
        'verbose': args.verbose
    }

    # --- Comparison mode: plot multiple datasets together ---
    if args.compare:
        datasets = []

        # 1. KSG precomputed E(tau) from Outdoor_E_ttc.csv
        precomputed_path = args.precomputed or 'data-KSG/Outdoor_E_ttc.csv'
        try:
            tau_pre, E_pre = load_precomputed_energy(precomputed_path)
            if len(tau_pre) > 0:
                datasets.append({
                    'tau': tau_pre, 'E': E_pre,
                    'label': 'KSG Precomputed (Outdoor)',
                    'color': 'green', 'marker': '^'
                })
                print(f"Loaded KSG precomputed: {len(tau_pre)} points")
        except FileNotFoundError:
            print(f"Warning: {precomputed_path} not found, skipping")

        # 2. KSG recomputed from Outdoor_ttc.csv + Outdoor_ttc_scrambled.csv
        ds = _load_dataset('data-KSG/Outdoor_ttc.csv',
                           'data-KSG/Outdoor_ttc_scrambled.csv',
                           'KSG Recomputed (Outdoor)', 'red', 'x',
                           **compute_params)
        if ds:
            datasets.append(ds)

        # 3. Simulation data (use --data/--scrambled if provided, else defaults)
        sim_data = args.data or 'data/tau_data.csv'
        sim_scrambled = args.scrambled or 'data/tau_scrambled.csv'
        ds = _load_dataset(sim_data, sim_scrambled,
                           'My Simulation', 'blue', 'o',
                           **compute_params)
        if ds:
            datasets.append(ds)

        if len(datasets) == 0:
            print("ERROR: No datasets could be loaded for comparison.")
            sys.exit(1)

        output = args.output if args.output != 'data/energy_plot.png' \
            else 'data/comparison_E_tau.png'
        title = args.title or r'Comparison: $E(\tau)$ Power Law'
        plot_energy(datasets, output, fit=args.fit, fit_all=args.fit_all,
                    title=title)
        return

    # --- Single dataset mode ---
    if args.precomputed:
        tau, E = load_precomputed_energy(args.precomputed)
        title = args.title or f'Precomputed E(tau) from {args.precomputed}'
    elif args.data and args.scrambled:
        ttc_data = load_ttc_data(args.data)
        ttc_scrambled = load_ttc_data(args.scrambled)
        print(f"Loaded {len(ttc_data)} time-to-collision values")
        print(f"Loaded {len(ttc_scrambled)} scrambled time-to-collision values")
        tau, E = compute_energy(ttc_data, ttc_scrambled, **compute_params)
        title = args.title or f'E(tau) from {args.data}'
    else:
        parser.error(
            'Provide either --precomputed OR both --data and --scrambled')
        return

    if len(tau) == 0:
        print("ERROR: No valid E(tau) data points.  Check input data.")
        sys.exit(1)

    print(f"E(tau): {len(tau)} valid data points")
    datasets = [{'tau': tau, 'E': E, 'label': 'E(tau) = ln(1/g)'}]
    plot_energy(datasets, args.output, fit=args.fit, title=title)


if __name__ == '__main__':
    main()
