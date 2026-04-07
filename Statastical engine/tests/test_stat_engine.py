"""
test_stat_engine.py
===================
Comprehensive unittest suite for StatEngine and simulate_crashes.

Run from the project root:
    python -m unittest discover -s tests -v

Or directly:
    python tests/test_stat_engine.py
"""

import math
import sys
import os
import unittest

# ---------------------------------------------------------------------------
# Path bootstrap so the tests work from any working directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stat_engine import StatEngine, EmptyDataError, InvalidDataTypeError
from src.monte_carlo import simulate_crashes, THEORETICAL_CRASH_PROB


# ===========================================================================
# 1.  Instantiation & data cleaning
# ===========================================================================

class TestInstantiation(unittest.TestCase):
    """Tests for __init__ and the internal _clean() method."""

    def test_valid_int_list(self):
        """Integer lists are accepted and stored as floats."""
        engine = StatEngine([1, 2, 3, 4, 5])
        self.assertEqual(engine.n, 5)
        self.assertEqual(engine.data, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_valid_float_list(self):
        engine = StatEngine([1.5, 2.5, 3.5])
        self.assertAlmostEqual(engine.data[0], 1.5)

    def test_valid_tuple_input(self):
        """Tuples are accepted as well as lists."""
        engine = StatEngine((10, 20, 30))
        self.assertEqual(engine.n, 3)

    def test_numeric_strings_are_invalid(self):
        """Strings that look like numbers must raise InvalidDataTypeError."""
        with self.assertRaises(InvalidDataTypeError):
            StatEngine([1, 2, "3", 4])

    def test_none_raises_invalid_type(self):
        with self.assertRaises(InvalidDataTypeError):
            StatEngine([1, 2, None, 4])

    def test_boolean_raises_invalid_type(self):
        """Booleans are a common Python gotcha; they must be rejected."""
        with self.assertRaises(InvalidDataTypeError):
            StatEngine([1, 2, True, 4])

    def test_mixed_types_raises(self):
        with self.assertRaises(InvalidDataTypeError):
            StatEngine([1, 2, "three", None, 5])

    def test_empty_list_raises(self):
        with self.assertRaises(EmptyDataError):
            StatEngine([])

    def test_empty_tuple_raises(self):
        with self.assertRaises(EmptyDataError):
            StatEngine(())

    def test_wrong_outer_type_raises(self):
        """Passing a dict or scalar must raise TypeError."""
        with self.assertRaises(TypeError):
            StatEngine({"a": 1})

    def test_single_element_accepted(self):
        engine = StatEngine([42])
        self.assertEqual(engine.n, 1)


# ===========================================================================
# 2.  get_mean()
# ===========================================================================

class TestMean(unittest.TestCase):

    def test_mean_simple(self):
        engine = StatEngine([1, 2, 3, 4, 5])
        self.assertAlmostEqual(engine.get_mean(), 3.0)

    def test_mean_single_value(self):
        engine = StatEngine([7])
        self.assertAlmostEqual(engine.get_mean(), 7.0)

    def test_mean_negative_values(self):
        engine = StatEngine([-5, -3, -1, 1, 3, 5])
        self.assertAlmostEqual(engine.get_mean(), 0.0)

    def test_mean_floats(self):
        engine = StatEngine([1.1, 2.2, 3.3])
        self.assertAlmostEqual(engine.get_mean(), 2.2, places=10)

    def test_mean_large_dataset(self):
        data = list(range(1, 101))   # 1..100  →  mean = 50.5
        engine = StatEngine(data)
        self.assertAlmostEqual(engine.get_mean(), 50.5)


# ===========================================================================
# 3.  get_median()
# ===========================================================================

class TestMedian(unittest.TestCase):

    def test_median_odd_count(self):
        """Odd n: median is the single middle element."""
        engine = StatEngine([3, 1, 4, 1, 5])   # sorted: 1 1 3 4 5
        self.assertAlmostEqual(engine.get_median(), 3.0)

    def test_median_even_count(self):
        """Even n: median is the mean of the two central elements."""
        engine = StatEngine([1, 2, 3, 4])      # (2 + 3) / 2 = 2.5
        self.assertAlmostEqual(engine.get_median(), 2.5)

    def test_median_already_sorted(self):
        engine = StatEngine([10, 20, 30, 40, 50])
        self.assertAlmostEqual(engine.get_median(), 30.0)

    def test_median_reverse_sorted(self):
        """Input order must not affect the result."""
        engine = StatEngine([50, 40, 30, 20, 10])
        self.assertAlmostEqual(engine.get_median(), 30.0)

    def test_median_two_elements(self):
        engine = StatEngine([4, 8])
        self.assertAlmostEqual(engine.get_median(), 6.0)

    def test_median_single_element(self):
        engine = StatEngine([99])
        self.assertAlmostEqual(engine.get_median(), 99.0)

    def test_median_with_duplicates(self):
        engine = StatEngine([2, 2, 2, 2, 2])
        self.assertAlmostEqual(engine.get_median(), 2.0)


# ===========================================================================
# 4.  get_mode()
# ===========================================================================

class TestMode(unittest.TestCase):

    def test_mode_unimodal(self):
        engine = StatEngine([1, 2, 2, 3])
        self.assertEqual(engine.get_mode(), [2.0])

    def test_mode_multimodal(self):
        engine = StatEngine([1, 1, 2, 2, 3])
        self.assertEqual(engine.get_mode(), [1.0, 2.0])

    def test_mode_all_unique_returns_string(self):
        engine = StatEngine([1, 2, 3, 4, 5])
        result = engine.get_mode()
        self.assertIsInstance(result, str)
        self.assertIn("unique", result.lower())

    def test_mode_triple_modal(self):
        engine = StatEngine([5, 5, 3, 3, 7, 7, 1])
        self.assertEqual(engine.get_mode(), [3.0, 5.0, 7.0])

    def test_mode_single_element(self):
        """A single element technically appears once — all-unique rule applies."""
        engine = StatEngine([42])
        result = engine.get_mode()
        self.assertIsInstance(result, str)


# ===========================================================================
# 5.  get_variance()
# ===========================================================================

class TestVariance(unittest.TestCase):
    """
    Known-outcome tests.
    Dataset: [2, 4, 4, 4, 5, 5, 7, 9]  (n=8, μ=5)
      Population variance: 4.0
      Sample variance:     4.571428…
    """

    DATASET = [2, 4, 4, 4, 5, 5, 7, 9]

    def test_population_variance(self):
        engine = StatEngine(self.DATASET)
        self.assertAlmostEqual(engine.get_variance(is_sample=False), 4.0, places=10)

    def test_sample_variance(self):
        engine = StatEngine(self.DATASET)
        expected = 32.0 / 7        # Σ(xi-μ)² = 32, divided by n-1=7
        self.assertAlmostEqual(engine.get_variance(is_sample=True), expected, places=10)

    def test_sample_variance_larger_than_population(self):
        """Bessel's correction always makes s² > σ² for n > 1."""
        engine = StatEngine(self.DATASET)
        self.assertGreater(
            engine.get_variance(is_sample=True),
            engine.get_variance(is_sample=False),
        )

    def test_variance_single_element_sample_raises(self):
        """Sample variance is undefined for n=1 (denominator = 0)."""
        engine = StatEngine([5])
        with self.assertRaises(EmptyDataError):
            engine.get_variance(is_sample=True)

    def test_variance_single_element_population_is_zero(self):
        """Population variance of a single value = 0 (no spread)."""
        engine = StatEngine([5])
        self.assertAlmostEqual(engine.get_variance(is_sample=False), 0.0)

    def test_variance_identical_values(self):
        """All identical values → zero variance."""
        engine = StatEngine([7, 7, 7, 7])
        self.assertAlmostEqual(engine.get_variance(is_sample=False), 0.0)
        self.assertAlmostEqual(engine.get_variance(is_sample=True), 0.0)


# ===========================================================================
# 6.  get_standard_deviation()
# ===========================================================================

class TestStandardDeviation(unittest.TestCase):

    def test_population_std_known_value(self):
        """
        Dataset [2, 4, 4, 4, 5, 5, 7, 9]:
          σ² = 4.0  →  σ = 2.0 (exact)
        """
        engine = StatEngine([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(engine.get_standard_deviation(is_sample=False), 2.0, places=10)

    def test_sample_std_is_sqrt_of_sample_variance(self):
        """s = √(s²) must hold to full precision."""
        data = [10, 20, 30, 40, 50]
        engine = StatEngine(data)
        expected = math.sqrt(engine.get_variance(is_sample=True))
        self.assertAlmostEqual(
            engine.get_standard_deviation(is_sample=True), expected, places=12
        )

    def test_population_std_is_sqrt_of_population_variance(self):
        data = [10, 20, 30, 40, 50]
        engine = StatEngine(data)
        expected = math.sqrt(engine.get_variance(is_sample=False))
        self.assertAlmostEqual(
            engine.get_standard_deviation(is_sample=False), expected, places=12
        )

    def test_std_identical_values(self):
        engine = StatEngine([3, 3, 3])
        self.assertAlmostEqual(engine.get_standard_deviation(is_sample=True), 0.0)

    def test_std_is_non_negative(self):
        engine = StatEngine([-100, 0, 100])
        self.assertGreaterEqual(engine.get_standard_deviation(), 0.0)


# ===========================================================================
# 7.  get_outliers()
# ===========================================================================

class TestOutliers(unittest.TestCase):

    def test_outliers_detected(self):
        """Values far from the mean are flagged as outliers."""
        # Tightly clustered around 100; 200 and 0 are unambiguous high-z outliers.
        # Mean ≈ 100, std ≈ 1.4 for the core cluster, so 200 and 0 → |z| >> 2
        core = [100] * 20          # 20 values at exactly 100
        data = core + [200, 0]     # two extreme outliers on both sides
        engine = StatEngine(data)
        outliers = engine.get_outliers(threshold=2.0)
        self.assertIn(200.0, outliers)
        self.assertIn(0.0, outliers)

    def test_no_outliers_normal_data(self):
        """Tightly clustered data should have zero outliers at threshold=2."""
        engine = StatEngine([10, 11, 10, 9, 10, 11, 9, 10])
        self.assertEqual(engine.get_outliers(threshold=2.0), [])

    def test_threshold_zero_raises(self):
        engine = StatEngine([1, 2, 3])
        with self.assertRaises(ValueError):
            engine.get_outliers(threshold=0)

    def test_threshold_negative_raises(self):
        engine = StatEngine([1, 2, 3])
        with self.assertRaises(ValueError):
            engine.get_outliers(threshold=-1)

    def test_identical_values_no_outliers(self):
        """When std dev == 0 there can be no z-score outliers."""
        engine = StatEngine([5, 5, 5, 5])
        self.assertEqual(engine.get_outliers(), [])

    def test_outliers_sorted(self):
        """Return value must be sorted ascending."""
        data = [10] * 8 + [100, -80]
        engine = StatEngine(data)
        outliers = engine.get_outliers(threshold=1.5)
        self.assertEqual(outliers, sorted(outliers))

    def test_strict_threshold(self):
        """A very tight threshold (0.5 std) should flag more points."""
        engine = StatEngine([2, 4, 4, 4, 5, 5, 7, 9])
        loose = engine.get_outliers(threshold=2.0)
        strict = engine.get_outliers(threshold=0.5)
        self.assertGreaterEqual(len(strict), len(loose))


# ===========================================================================
# 8.  Monte Carlo simulation
# ===========================================================================

class TestMonteCarlo(unittest.TestCase):

    def test_returns_tuple(self):
        result = simulate_crashes(100, seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_crashes_non_negative(self):
        crashes, _ = simulate_crashes(1000, seed=1)
        self.assertGreaterEqual(crashes, 0)

    def test_crashes_le_days(self):
        days = 500
        crashes, prob = simulate_crashes(days, seed=2)
        self.assertLessEqual(crashes, days)

    def test_probability_range(self):
        _, prob = simulate_crashes(1000, seed=3)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_large_n_converges_to_theory(self):
        """
        LLN: at 100,000 days the simulated probability should be within
        0.5 percentage points of the theoretical 4.5%.
        """
        _, prob = simulate_crashes(100_000, seed=99)
        self.assertAlmostEqual(prob, THEORETICAL_CRASH_PROB, delta=0.005)

    def test_invalid_days_raises(self):
        with self.assertRaises(ValueError):
            simulate_crashes(0)

    def test_reproducibility_with_seed(self):
        """Same seed must always produce the same result."""
        r1 = simulate_crashes(1000, seed=7)
        r2 = simulate_crashes(1000, seed=7)
        self.assertEqual(r1, r2)

    def test_different_seeds_differ(self):
        r1 = simulate_crashes(1000, seed=1)
        r2 = simulate_crashes(1000, seed=2)
        # Extremely unlikely to be equal for different seeds
        self.assertNotEqual(r1, r2)


# ===========================================================================
# 9.  Error message quality
# ===========================================================================

class TestErrorMessages(unittest.TestCase):

    def test_empty_data_error_message(self):
        try:
            StatEngine([])
        except EmptyDataError as e:
            self.assertIn("empty", str(e).lower())

    def test_invalid_type_error_lists_bad_values(self):
        try:
            StatEngine([1, "bad", None])
        except InvalidDataTypeError as e:
            msg = str(e)
            self.assertIn("bad", msg)
            self.assertIn("None", msg)


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)

