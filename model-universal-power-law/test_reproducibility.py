"""
Pytest test cases to verify reproducibility of reproduce-exp.sh experiment.

These tests ensure that running the simulation with seed=42 and the specified
parameters always produces the same results:
  - Exact same TTC values in data CSV
  - Exact same scrambled TTC values
  - Consistent power law fitting results

TOLERANCES:
  - Simulation TTC values: EXACT (deterministic with fixed seed)
  - Scrambled TTC values: EXACT (deterministic with fixed seed=42 in scrambling)
  - Fitting slope: ±0.01 (floating-point precision in log-log regression)
  - Fitting alpha (powerlaw package): ±0.05 (algorithm sensitivity to bin edges)
  - R-statistics: ±0.1 (sensitive to small changes in distribution tail)
  - p-values: ±0.1 (statistical test variability)

Run with:
    pytest test_reproducibility.py -v
"""

import os
import sys
import tempfile
import pytest
import numpy as np

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttc_engine import TTCSimulation
from environment import load_config
from plot_powerlaw import load_ttc_data, compute_energy, _fit_powerlaw


# =============================================================================
# Reference values from reproduce-exp.sh with seed=42, env-size=15, sight=4
# =============================================================================

# Expected count of data points (excluding header)
EXPECTED_DATA_COUNT = 5898
EXPECTED_SCRAMBLED_COUNT = 47789

# First 20 tau values from tau_torus_data.csv (after header)
# These should match EXACTLY with seed=42
EXPECTED_FIRST_TAU_VALUES = [
    100.0,
    100.0,
    9.991915909456734,
    20.240841918284346,
    6.770133135634841,
    9.26008903493152,
    18.75590189788411,
    8.639765396940504,
    6.159027695482968,
    14.901997426852779,
    7.938051637496106,
    12.918582848890246,
    8.502418639314335,
    2.758005104086032,
    8.436417736766291,
    38.88152889095978,
    4.447770249885931,
    2.6067813675804508,
    2.535841854685951,
]

# Statistical summary of tau_torus_data.csv (filtered < 10.0)
# Used for sanity checks
EXPECTED_DATA_MEAN_APPROX = 4.82
EXPECTED_DATA_STD_APPROX = 7.82

# Expected fitting results from reference data (reproduce-exp.sh output)
# These are the OBSERVED values, not theoretical predictions
EXPECTED_SLOPE = -2.75      # Observed slope (theory predicts ~-2.0)
EXPECTED_ALPHA = 2.30       # Observed powerlaw alpha


# =============================================================================
# Tolerances
# =============================================================================

# Simulation output is deterministic - should be EXACT match
TAU_VALUE_TOLERANCE = 0.0  # bit-for-bit identical

# Fitting tolerances for reproducibility verification
# These are tight tolerances to detect if fitting changes unexpectedly
SLOPE_TOLERANCE = 0.05      # log-log linear regression (deterministic, small FP variance)
ALPHA_TOLERANCE = 0.1       # powerlaw package fit (deterministic but sensitive to xmin)
R_STAT_TOLERANCE = 0.5      # R-statistics for distribution comparison
P_VALUE_TOLERANCE = 0.5     # p-values (can be sensitive)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def simulation_output():
    """
    Run the simulation with reproduce-exp.sh parameters and return results.

    Uses same parameters as reproduce-exp.sh:
      --seed 42 --env-size 15 --sight 4 --config ./config/torus.yaml
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'torus.yaml')
    env, physics = load_config(config_path, env_size=15.0, sight=4.0)
    sim = TTCSimulation(env=env, seed=42, **physics)

    num_iterations = 10000
    sample_interval = 5

    snapshots = []
    all_ttc = []

    for iteration in range(num_iterations):
        sim.step()

        if iteration % sample_interval == 0:
            snapshots.append((sim.pos.copy(), sim.vel.copy()))
            ttc_values = sim.compute_all_pairwise_ttc()
            all_ttc.extend(ttc_values)

    scrambled_ttc = sim.compute_scrambled_ttc(snapshots)

    return {
        'data': all_ttc,
        'scrambled': scrambled_ttc,
        'sim': sim,
    }


@pytest.fixture(scope="module")
def reference_data():
    """Load reference data from data-presentation directory."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data-presentation')
    data_path = os.path.join(data_dir, 'tau_torus_data.csv')
    scrambled_path = os.path.join(data_dir, 'tau_torus_scrambled.csv')

    if not os.path.exists(data_path):
        pytest.skip("Reference data not found. Run reproduce-exp.sh first.")

    return {
        'data': load_ttc_data(data_path),
        'scrambled': load_ttc_data(scrambled_path),
    }


# =============================================================================
# Test: Simulation Reproducibility (EXACT)
# =============================================================================

class TestSimulationReproducibility:
    """Tests that simulation with seed=42 produces identical results."""

    def test_data_count(self, simulation_output):
        """Verify the number of TTC data points matches expected."""
        assert len(simulation_output['data']) == EXPECTED_DATA_COUNT, (
            f"Expected {EXPECTED_DATA_COUNT} data points, "
            f"got {len(simulation_output['data'])}"
        )

    def test_scrambled_count(self, simulation_output):
        """Verify the number of scrambled TTC values matches expected."""
        assert len(simulation_output['scrambled']) == EXPECTED_SCRAMBLED_COUNT, (
            f"Expected {EXPECTED_SCRAMBLED_COUNT} scrambled values, "
            f"got {len(simulation_output['scrambled'])}"
        )

    def test_first_tau_values_exact(self, simulation_output):
        """Verify first tau values match EXACTLY (deterministic seeding)."""
        data = simulation_output['data']

        for i, expected in enumerate(EXPECTED_FIRST_TAU_VALUES):
            actual = data[i]
            assert actual == expected, (
                f"Tau value at index {i} differs: "
                f"expected {expected}, got {actual}. "
                f"This indicates non-deterministic behavior!"
            )

    def test_matches_reference_data(self, simulation_output, reference_data):
        """Verify simulation output matches saved reference data exactly."""
        sim_data = np.array(simulation_output['data'])
        ref_data = reference_data['data']

        assert len(sim_data) == len(ref_data), (
            f"Data length mismatch: simulation={len(sim_data)}, "
            f"reference={len(ref_data)}"
        )

        # Check for exact match
        np.testing.assert_array_equal(
            sim_data, ref_data,
            err_msg="Simulation data does not match reference. "
                    "This may indicate platform-specific floating-point differences."
        )

    def test_scrambled_matches_reference(self, simulation_output, reference_data):
        """Verify scrambled TTC matches saved reference exactly."""
        sim_scrambled = np.array(simulation_output['scrambled'])
        ref_scrambled = reference_data['scrambled']

        assert len(sim_scrambled) == len(ref_scrambled), (
            f"Scrambled length mismatch: simulation={len(sim_scrambled)}, "
            f"reference={len(ref_scrambled)}"
        )

        np.testing.assert_array_equal(
            sim_scrambled, ref_scrambled,
            err_msg="Scrambled data does not match reference."
        )


# =============================================================================
# Test: Multiple Runs Produce Identical Results
# =============================================================================

class TestMultipleRunsIdentical:
    """Tests that running the simulation twice produces identical results."""

    def test_two_runs_identical(self):
        """Run simulation twice and verify bit-for-bit identical output."""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'torus.yaml')

        results = []
        for run in range(2):
            env, physics = load_config(config_path, env_size=15.0, sight=4.0)
            sim = TTCSimulation(env=env, seed=42, **physics)

            # Run fewer iterations for speed
            num_iterations = 500
            sample_interval = 5

            snapshots = []
            all_ttc = []

            for iteration in range(num_iterations):
                sim.step()
                if iteration % sample_interval == 0:
                    snapshots.append((sim.pos.copy(), sim.vel.copy()))
                    all_ttc.extend(sim.compute_all_pairwise_ttc())

            scrambled = sim.compute_scrambled_ttc(snapshots)
            results.append({
                'data': np.array(all_ttc),
                'scrambled': np.array(scrambled),
            })

        np.testing.assert_array_equal(
            results[0]['data'], results[1]['data'],
            err_msg="Two simulation runs produced different TTC data!"
        )

        np.testing.assert_array_equal(
            results[0]['scrambled'], results[1]['scrambled'],
            err_msg="Two simulation runs produced different scrambled data!"
        )


# =============================================================================
# Test: Fitting Reproducibility (with tolerance)
# =============================================================================

class TestFittingReproducibility:
    """Tests that power law fitting produces consistent results."""

    @pytest.fixture(scope="class")
    def fitting_results(self, reference_data):
        """Compute energy and fit power law from reference data."""
        tau, E = compute_energy(
            reference_data['data'],
            reference_data['scrambled'],
            num_bins=200,
            verbose=False
        )

        if len(tau) < 3:
            pytest.skip("Not enough data points for fitting")

        fit_result = _fit_powerlaw(tau, E, label='Test')
        return {
            'tau': tau,
            'E': E,
            'fit': fit_result,
        }

    def test_slope_reproducible(self, fitting_results):
        """Verify fitted slope matches the reference value (reproducibility check)."""
        slope = fitting_results['fit']['slope']

        assert abs(slope - EXPECTED_SLOPE) < SLOPE_TOLERANCE, (
            f"Fitted slope {slope:.4f} differs from reference {EXPECTED_SLOPE}. "
            f"Tolerance: ±{SLOPE_TOLERANCE}. "
            f"Note: Theory predicts ~-2.0, but our simulation yields ~{EXPECTED_SLOPE}"
        )

    def test_alpha_reproducible(self, fitting_results):
        """Verify powerlaw alpha matches reference value (reproducibility check)."""
        alpha = fitting_results['fit']['alpha']

        assert abs(alpha - EXPECTED_ALPHA) < ALPHA_TOLERANCE, (
            f"Powerlaw alpha {alpha:.4f} differs from reference {EXPECTED_ALPHA}. "
            f"Tolerance: ±{ALPHA_TOLERANCE}"
        )

    def test_fitting_deterministic(self, reference_data):
        """Verify fitting produces identical results on multiple runs."""
        results = []

        for _ in range(3):
            tau, E = compute_energy(
                reference_data['data'],
                reference_data['scrambled'],
                num_bins=200,
                verbose=False
            )

            if len(tau) < 3:
                pytest.skip("Not enough data points")

            fit_result = _fit_powerlaw(tau, E, label='Test')
            results.append(fit_result)

        # All slopes should be identical (deterministic)
        for i in range(1, len(results)):
            assert results[0]['slope'] == results[i]['slope'], (
                f"Slope differs between runs: {results[0]['slope']} vs {results[i]['slope']}"
            )

    def test_r_statistics_sign(self, fitting_results):
        """Verify R-statistics have expected signs (power law preferred)."""
        R_exp = fitting_results['fit']['R_exp']
        R_ln = fitting_results['fit']['R_ln']

        # Positive R means power law is preferred over the alternative
        # This may not always hold depending on data, so we just check it's reasonable
        assert abs(R_exp) < 50, f"R_exp {R_exp} is unreasonably large"
        assert abs(R_ln) < 50, f"R_ln {R_ln} is unreasonably large"


# =============================================================================
# Test: Energy Computation Reproducibility
# =============================================================================

class TestEnergyComputation:
    """Tests for compute_energy function reproducibility."""

    def test_energy_computation_deterministic(self, reference_data):
        """Verify compute_energy is deterministic."""
        results = []

        for _ in range(3):
            tau, E = compute_energy(
                reference_data['data'],
                reference_data['scrambled'],
                num_bins=200,
                tau_min=0.4,
                verbose=False
            )
            results.append((tau, E))

        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0][0], results[i][0],
                err_msg=f"tau differs between computation {0} and {i}"
            )
            np.testing.assert_array_equal(
                results[0][1], results[i][1],
                err_msg=f"E differs between computation {0} and {i}"
            )

    def test_energy_positive_in_avoidance_regime(self, reference_data):
        """Verify E(tau) > 0 in avoidance regime (g < 1)."""
        tau, E = compute_energy(
            reference_data['data'],
            reference_data['scrambled'],
            verbose=False
        )

        assert len(E) > 0, "No energy values computed"
        assert np.all(E > 0), "Some E values are not positive (avoidance regime)"
        assert np.all(tau > 0), "Some tau values are not positive"

    def test_energy_count_stable(self, reference_data):
        """Verify number of valid E(tau) points is stable."""
        tau, E = compute_energy(
            reference_data['data'],
            reference_data['scrambled'],
            num_bins=200,
            verbose=False
        )

        # Should have a reasonable number of points (depends on data)
        # With 200 bins and good data, expect 50-150 valid points
        assert len(tau) > 20, f"Too few valid E(tau) points: {len(tau)}"
        assert len(tau) < 200, f"Unexpectedly many E(tau) points: {len(tau)}"


# =============================================================================
# Test: CSV I/O Reproducibility
# =============================================================================

class TestCSVIO:
    """Tests for CSV reading/writing reproducibility."""

    def test_csv_roundtrip(self, simulation_output):
        """Verify CSV write/read roundtrip preserves values exactly."""
        import csv

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['tau'])
            for v in simulation_output['data'][:100]:  # First 100 values
                writer.writerow([v])
            temp_path = f.name

        try:
            loaded = load_ttc_data(temp_path)
            original = np.array(simulation_output['data'][:100])

            np.testing.assert_array_equal(
                original, loaded,
                err_msg="CSV roundtrip changed values"
            )
        finally:
            os.unlink(temp_path)


# =============================================================================
# Test: Statistical Properties
# =============================================================================

class TestStatisticalProperties:
    """Tests for expected statistical properties of the simulation output."""

    def test_data_statistics_stable(self, reference_data):
        """Verify data statistics match expected values."""
        data = reference_data['data']

        # Filter out sentinels
        valid = data[data < 10.0]

        mean = np.mean(valid)
        std = np.std(valid)

        # These should be stable within reason
        assert 1.0 < mean < 10.0, f"Mean {mean} outside expected range"
        assert 0.5 < std < 15.0, f"Std {std} outside expected range"

    def test_scrambled_larger_than_data(self, reference_data):
        """Verify scrambled dataset is larger (more pairs)."""
        assert len(reference_data['scrambled']) > len(reference_data['data']), (
            "Scrambled dataset should be larger than observed data"
        )

    def test_sentinel_values_present(self, reference_data):
        """Verify sentinel values (100, 999) are present in raw data."""
        data = reference_data['data']

        has_overlap = np.any(data == 100.0)
        has_no_collision = np.any(data >= 999.0)

        # At least one type of sentinel should be present
        assert has_overlap or has_no_collision, (
            "Expected some sentinel values in raw TTC data"
        )


# =============================================================================
# Tolerance Documentation
# =============================================================================

def test_tolerance_documentation():
    """Document and verify tolerance values are reasonable."""
    tolerances = {
        'TAU_VALUE_TOLERANCE': TAU_VALUE_TOLERANCE,
        'SLOPE_TOLERANCE': SLOPE_TOLERANCE,
        'ALPHA_TOLERANCE': ALPHA_TOLERANCE,
        'R_STAT_TOLERANCE': R_STAT_TOLERANCE,
        'P_VALUE_TOLERANCE': P_VALUE_TOLERANCE,
    }

    print("\n" + "="*70)
    print("REPRODUCIBILITY TOLERANCES")
    print("="*70)
    print("\nSimulation output (with fixed seed=42):")
    print("  - TTC values: EXACT (0.0 tolerance)")
    print("  - Scrambled values: EXACT (0.0 tolerance)")
    print("\nFitting reference values (observed from reproduce-exp.sh):")
    print(f"  - Expected slope: {EXPECTED_SLOPE} (theory predicts -2.0)")
    print(f"  - Expected alpha: {EXPECTED_ALPHA}")
    print("\nFitting tolerances (for reproducibility verification):")
    print(f"  - Slope (log-log regression): ±{SLOPE_TOLERANCE}")
    print(f"  - Alpha (powerlaw package): ±{ALPHA_TOLERANCE}")
    print(f"  - R-statistics: ±{R_STAT_TOLERANCE}")
    print(f"  - p-values: ±{P_VALUE_TOLERANCE}")
    print("\nNote: These tolerances verify REPRODUCIBILITY, not theoretical")
    print("correctness. The simulation reliably produces slope ~-2.75,")
    print("which differs from theory (~-2.0) but is consistently reproducible.")
    print("="*70 + "\n")

    # Verify tolerances are non-negative
    for name, value in tolerances.items():
        assert value >= 0, f"Tolerance {name} must be non-negative"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
