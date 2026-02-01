"""
Plot cluster size distribution on a log-log scale and fit power law.

Analyzes cluster size data from pedestrian simulations to investigate
self-organized criticality. If the system exhibits SOC, the cluster size
distribution should follow a power law: P(s) ~ s^(-tau).

Usage:
    python plot_powerlaw_cluster_size.py --data cluster_sizes.csv --fit --output cluster_powerlaw.png
"""

import argparse
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_cluster_sizes(filepath):
    """
    Load cluster size values from a CSV file.

    Expected format: single column with header 'size'.
    """
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(int(float(line)))
            except ValueError:
                continue  # skip header or non-numeric lines
    return np.array(values)


def fit_powerlaw(sizes, label='Cluster Size'):
    """
    Fit power law distribution using the powerlaw package.

    Args:
        sizes: array of cluster size values
        label: label for verbose output

    Returns:
        dict with keys: tau (alpha), xmin, R_exp, p_exp, R_ln, p_ln
    """
    import powerlaw

    print(f"\n=== Power Law Fit: {label} ===")
    # Use discrete=True since cluster sizes are integers
    pl = powerlaw.Fit(sizes, discrete=True, verbose=True)

    print(f"  Fitted tau (exponent): {pl.alpha:.4f}")
    print(f"  xmin: {pl.xmin:.4f}")

    # Compare power law to alternatives
    R_exp, p_exp = pl.distribution_compare('power_law', 'exponential')
    R_ln, p_ln = pl.distribution_compare('power_law', 'lognormal')
    print(f"  Power law vs exponential: R={R_exp:.4f}, p={p_exp:.4f}")
    print(f"  Power law vs lognormal:   R={R_ln:.4f}, p={p_ln:.4f}")
    print(f"{'=' * 35}\n")

    return {
        'tau': pl.alpha,
        'xmin': pl.xmin,
        'R_exp': R_exp,
        'p_exp': p_exp,
        'R_ln': R_ln,
        'p_ln': p_ln,
        'fit_object': pl,
    }


def plot_cluster_sizes(sizes, output_path, fit=False, title='Cluster Size Distribution'):
    """
    Create a log-log plot of cluster size distribution.

    Args:
        sizes: array of cluster size values
        output_path: path to save the plot
        fit: whether to fit and show power law
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Filter to positive values
    sizes = sizes[sizes > 0]

    if len(sizes) == 0:
        print("ERROR: No valid cluster size data points.")
        sys.exit(1)

    # Compute empirical probability distribution
    unique_sizes, counts = np.unique(sizes, return_counts=True)
    probs = counts / len(sizes)

    # Plot empirical distribution
    ax.scatter(unique_sizes, probs, s=60, alpha=0.7, color='steelblue',
               label=f'Data (n={len(sizes)} clusters)')

    fit_result = None
    if fit and len(sizes) > 10:
        fit_result = fit_powerlaw(sizes)

        # Plot power law fit line
        # P(s) ~ s^(-tau) for s >= xmin
        max_size = unique_sizes.max()
        s_fit = np.arange(int(fit_result['xmin']), max_size + 1)

        if len(s_fit) > 1:
            # Find probability at xmin for anchoring
            xmin_idx = np.where(unique_sizes >= fit_result['xmin'])[0]
            if len(xmin_idx) > 0:
                anchor_idx = xmin_idx[0]
                anchor_s = unique_sizes[anchor_idx]
                anchor_p = probs[anchor_idx]

                p_fit = anchor_p * (s_fit / anchor_s) ** (-fit_result['tau'])

                # Build legend label with R/p statistics
                fit_label = (f"Power law fit: τ={fit_result['tau']:.2f}\n"
                            f"  vs exp: R={fit_result['R_exp']:.2f}, p={fit_result['p_exp']:.3f}\n"
                            f"  vs ln:  R={fit_result['R_ln']:.2f}, p={fit_result['p_ln']:.3f}")

                ax.plot(s_fit, p_fit, 'r-', linewidth=2, alpha=0.8, label=fit_label)

    # Reference line: slope -2.05 (2D percolation universality class)
    if len(unique_sizes) > 1:
        s_ref = np.arange(1, unique_sizes.max() + 1)
        # Anchor at geometric mean
        geo_mean_s = np.exp(np.mean(np.log(unique_sizes)))
        geo_mean_p = np.exp(np.mean(np.log(probs)))
        p_ref = geo_mean_p * (s_ref / geo_mean_s) ** (-2.05)
        ax.plot(s_ref, p_ref, 'k--', linewidth=1.5, alpha=0.5,
                label=r'Reference: $P \propto s^{-2.05}$ (2D percolation)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cluster Size s', fontsize=12)
    ax.set_ylabel('Probability P(s)', fontsize=12)

    # Build title with statistics
    subtitle = f'n={len(sizes)} clusters, mean={sizes.mean():.2f}, max={sizes.max()}'
    if fit_result:
        subtitle += f'\nτ={fit_result["tau"]:.2f} (xmin={int(fit_result["xmin"])})'
    ax.set_title(f'{title}\n{subtitle}', fontsize=14)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot cluster size distribution and fit power law. '
                    'Investigates self-organized criticality in pedestrian clustering.')

    parser.add_argument('--data', type=str, required=True,
                        help='Path to cluster sizes CSV file')
    parser.add_argument('--fit', action='store_true',
                        help='Enable power-law fitting using powerlaw package')
    parser.add_argument('--output', type=str, default='cluster_powerlaw.png',
                        help='Output plot path (default: cluster_powerlaw.png)')
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title')
    parser.add_argument('--verbose', action='store_true',
                        help='Print additional diagnostic information')

    args = parser.parse_args()

    # Load data
    sizes = load_cluster_sizes(args.data)
    print(f"Loaded {len(sizes)} cluster size values from {args.data}")

    if len(sizes) == 0:
        print("ERROR: No cluster size data found.")
        sys.exit(1)

    if args.verbose:
        unique_sizes, counts = np.unique(sizes, return_counts=True)
        print(f"  Unique sizes: {len(unique_sizes)}")
        print(f"  Min: {sizes.min()}")
        print(f"  Max: {sizes.max()}")
        print(f"  Mean: {sizes.mean():.4f}")
        print(f"  Median: {np.median(sizes):.4f}")
        print("  Size distribution (top 10):")
        sorted_idx = np.argsort(-counts)[:10]
        for idx in sorted_idx:
            print(f"    size={unique_sizes[idx]}: {counts[idx]} ({100*counts[idx]/len(sizes):.1f}%)")

    # Plot
    title = args.title or 'Cluster Size Distribution'
    plot_cluster_sizes(sizes, args.output, fit=args.fit, title=title)


if __name__ == '__main__':
    main()
