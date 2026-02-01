"""
Plot jam duration distribution on a log-log scale and fit power law.

Analyzes jam duration data from pedestrian simulations to investigate
self-organized criticality. If jams exhibit SOC, the duration distribution
should follow a power law: P(T) ~ T^(-alpha).

Usage:
    python plot_powerlaw_jam_duration.py --data jam_durations.csv --fit --output jam_powerlaw.png
"""

import argparse
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_jam_durations(filepath):
    """
    Load jam duration values from a CSV file.

    Expected format: single column with header 'duration'.
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


def fit_powerlaw(durations, label='Jam Duration'):
    """
    Fit power law distribution using the powerlaw package.

    Args:
        durations: array of jam duration values
        label: label for verbose output

    Returns:
        dict with keys: alpha, xmin, R_exp, p_exp, R_ln, p_ln
    """
    import powerlaw

    print(f"\n=== Power Law Fit: {label} ===")
    pl = powerlaw.Fit(durations, verbose=True)

    print(f"  Fitted alpha (exponent): {pl.alpha:.4f}")
    print(f"  xmin: {pl.xmin:.4f}")

    # Compare power law to alternatives
    R_exp, p_exp = pl.distribution_compare('power_law', 'exponential')
    R_ln, p_ln = pl.distribution_compare('power_law', 'lognormal')
    print(f"  Power law vs exponential: R={R_exp:.4f}, p={p_exp:.4f}")
    print(f"  Power law vs lognormal:   R={R_ln:.4f}, p={p_ln:.4f}")
    print(f"{'=' * 35}\n")

    return {
        'alpha': pl.alpha,
        'xmin': pl.xmin,
        'R_exp': R_exp,
        'p_exp': p_exp,
        'R_ln': R_ln,
        'p_ln': p_ln,
        'fit_object': pl,
    }


def plot_jam_durations(durations, output_path, fit=False, title='Jam Duration Distribution'):
    """
    Create a log-log plot of jam duration distribution.

    Args:
        durations: array of jam duration values
        output_path: path to save the plot
        fit: whether to fit and show power law
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Filter to positive values
    durations = durations[durations > 0]

    if len(durations) == 0:
        print("ERROR: No valid jam duration data points.")
        sys.exit(1)

    # Create histogram (PDF)
    # Use logarithmic bins for better visualization
    min_val = durations.min()
    max_val = durations.max()

    if min_val == max_val:
        print("ERROR: All jam durations are identical.")
        sys.exit(1)

    bins = np.logspace(np.log10(min_val), np.log10(max_val), 30)
    hist, bin_edges = np.histogram(durations, bins=bins, density=True)

    # Bin centers for plotting
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean

    # Filter out zero bins
    valid = hist > 0
    bin_centers = bin_centers[valid]
    hist = hist[valid]

    # Plot histogram
    ax.scatter(bin_centers, hist, s=50, alpha=0.7, color='steelblue',
               label=f'Data (n={len(durations)} events)')

    fit_result = None
    if fit and len(durations) > 10:
        fit_result = fit_powerlaw(durations)

        # Plot power law fit line
        # P(x) ~ x^(-alpha) for x >= xmin
        x_fit = np.logspace(np.log10(fit_result['xmin']),
                           np.log10(max_val), 100)

        # Normalize to match histogram scale
        # Find the histogram value closest to xmin for anchoring
        anchor_idx = np.argmin(np.abs(bin_centers - fit_result['xmin']))
        if anchor_idx < len(hist):
            anchor_y = hist[anchor_idx]
            anchor_x = bin_centers[anchor_idx]
            y_fit = anchor_y * (x_fit / anchor_x) ** (-fit_result['alpha'])

            # Build legend label with R/p statistics
            fit_label = (f"Power law fit: α={fit_result['alpha']:.2f}\n"
                        f"  vs exp: R={fit_result['R_exp']:.2f}, p={fit_result['p_exp']:.3f}\n"
                        f"  vs ln:  R={fit_result['R_ln']:.2f}, p={fit_result['p_ln']:.3f}")

            ax.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.8, label=fit_label)

    # Reference line: slope -2 (typical SOC exponent)
    if len(bin_centers) > 0:
        x_ref = np.logspace(np.log10(bin_centers.min()),
                           np.log10(bin_centers.max()), 100)
        # Anchor at geometric mean
        geo_mean_x = np.exp(np.mean(np.log(bin_centers)))
        geo_mean_y = np.exp(np.mean(np.log(hist)))
        y_ref = geo_mean_y * (x_ref / geo_mean_x) ** (-2)
        ax.plot(x_ref, y_ref, 'k--', linewidth=1.5, alpha=0.5,
                label=r'Reference: $P \propto T^{-2}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Jam Duration T [s]', fontsize=12)
    ax.set_ylabel('Probability Density P(T)', fontsize=12)

    # Build title with statistics
    subtitle = f'n={len(durations)} events, mean={durations.mean():.2f}s, max={durations.max():.2f}s'
    if fit_result:
        subtitle += f'\nα={fit_result["alpha"]:.2f} (xmin={fit_result["xmin"]:.2f}s)'
    ax.set_title(f'{title}\n{subtitle}', fontsize=14)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot jam duration distribution and fit power law. '
                    'Investigates self-organized criticality in pedestrian jams.')

    parser.add_argument('--data', type=str, required=True,
                        help='Path to jam durations CSV file')
    parser.add_argument('--fit', action='store_true',
                        help='Enable power-law fitting using powerlaw package')
    parser.add_argument('--output', type=str, default='jam_powerlaw.png',
                        help='Output plot path (default: jam_powerlaw.png)')
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title')
    parser.add_argument('--verbose', action='store_true',
                        help='Print additional diagnostic information')

    args = parser.parse_args()

    # Load data
    durations = load_jam_durations(args.data)
    print(f"Loaded {len(durations)} jam duration values from {args.data}")

    if len(durations) == 0:
        print("ERROR: No jam duration data found.")
        sys.exit(1)

    if args.verbose:
        print(f"  Min: {durations.min():.4f}s")
        print(f"  Max: {durations.max():.4f}s")
        print(f"  Mean: {durations.mean():.4f}s")
        print(f"  Median: {np.median(durations):.4f}s")
        print(f"  Std: {durations.std():.4f}s")

    # Plot
    title = args.title or f'Jam Duration Distribution'
    plot_jam_durations(durations, args.output, fit=args.fit, title=title)


if __name__ == '__main__':
    main()
