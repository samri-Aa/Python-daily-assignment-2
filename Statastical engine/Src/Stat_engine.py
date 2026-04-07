"""
stat_engine.py
==============
Pure-Python statistical engine built exclusively on the standard library.
No third-party packages (numpy, pandas, statistics, etc.) are used.

Mathematical foundations implemented from scratch:
  - Central tendency  : mean, median, mode
  - Dispersion        : variance (population & sample), standard deviation
  - Outlier detection : z-score threshold method
"""

import math
from typing import List, Union, Tuple


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class EmptyDataError(ValueError):
    """Raised when a StatEngine method is called on an empty dataset."""

    def __str__(self) -> str:
        return (
            "EmptyDataError: The dataset is empty. "
            "Please provide at least one numeric value."
        )


class InvalidDataTypeError(TypeError):
    """Raised when the raw data contains non-numeric, un-cleanable values."""

    def __init__(self, bad_values: list):
        self.bad_values = bad_values

    def __str__(self) -> str:
        return (
            f"InvalidDataTypeError: The following non-numeric values could not "
            f"be coerced to float and were rejected: {self.bad_values}. "
            "Please ensure all data points are numeric (int or float)."
        )


# ---------------------------------------------------------------------------
# StatEngine
# ---------------------------------------------------------------------------

class StatEngine:
    """
    A self-contained statistical analysis engine.

    Parameters
    ----------
    data : list or tuple
        Raw 1-D numerical data.  Mixed-type inputs are cleaned automatically:
        - Values that can be cast to float are kept.
        - Values that cannot (e.g. strings that aren't numeric, None, booleans)
          are collected and surfaced via InvalidDataTypeError.

    Attributes
    ----------
    data : list[float]
        The cleaned, validated dataset.
    n    : int
        Number of observations after cleaning.

    Raises
    ------
    EmptyDataError
        If the cleaned dataset is empty.
    InvalidDataTypeError
        If any value cannot be coerced to a numeric type.
    """

    # ------------------------------------------------------------------
    # Construction & data validation
    # ------------------------------------------------------------------

    def __init__(self, data: Union[List, Tuple]) -> None:
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"Expected a list or tuple, got {type(data).__name__}."
            )

        cleaned, bad = self._clean(data)

        if bad:
            raise InvalidDataTypeError(bad)

        if len(cleaned) == 0:
            raise EmptyDataError()

        self.data: List[float] = cleaned
        self.n: int = len(self.data)

    @staticmethod
    def _clean(raw: Union[List, Tuple]) -> Tuple[List[float], list]:
        """
        Attempt to coerce every element in *raw* to float.

        Returns
        -------
        (good, bad) where *good* is a list[float] of all convertible values
        and *bad* is a list of the original values that failed conversion.

        Booleans are intentionally rejected: ``True``/``False`` are subclasses
        of int in Python, but accepting them silently would be misleading for
        a numeric dataset.
        """
        good: List[float] = []
        bad: list = []

        for item in raw:
            # Reject booleans explicitly before the int/float check
            if isinstance(item, bool):
                bad.append(item)
                continue
            # Accept only genuine numeric types (int, float)
            # Strings — even those that look like numbers — are rejected.
            # This prevents silent data corruption from mixed-type datasets.
            if not isinstance(item, (int, float)):
                bad.append(item)
                continue
            good.append(float(item))

        return good, bad

    # ------------------------------------------------------------------
    # Central Tendency
    # ------------------------------------------------------------------

    def get_mean(self) -> float:
        """
        Arithmetic mean: μ = (Σ xᵢ) / n

        Returns
        -------
        float
        """
        return sum(self.data) / self.n

    def get_median(self) -> float:
        """
        The middle value of the sorted dataset.

        Algorithm
        ---------
        Sort the data, then:
          - Odd  n  → middle element  → data[n // 2]
          - Even n  → average of the two central elements
                       → (data[n//2 - 1] + data[n//2]) / 2

        Returns
        -------
        float
        """
        sorted_data = sorted(self.data)

        if self.n % 2 == 1:
            # Odd: single middle element
            return float(sorted_data[self.n // 2])
        else:
            # Even: mean of the two central elements
            lower = sorted_data[self.n // 2 - 1]
            upper = sorted_data[self.n // 2]
            return (lower + upper) / 2.0

    def get_mode(self) -> Union[List[float], str]:
        """
        The most frequently occurring value(s).

        Handles three cases:
          1. Unimodal   – returns a single-element list, e.g. [4.0]
          2. Multimodal – returns all tied modes sorted ascending, e.g. [2.0, 5.0]
          3. All unique – returns the explanatory string below

        Returns
        -------
        list[float] | str
            List of mode(s), or
            "No mode: all values are unique." if every value appears exactly once.
        """
        # Build a frequency map without collections.Counter
        freq: dict = {}
        for val in self.data:
            freq[val] = freq.get(val, 0) + 1

        max_freq = max(freq.values())

        # If every value appears exactly once there is no meaningful mode
        if max_freq == 1:
            return "No mode: all values are unique."

        modes = sorted(k for k, v in freq.items() if v == max_freq)
        return modes

    # ------------------------------------------------------------------
    # Dispersion
    # ------------------------------------------------------------------

    def get_variance(self, is_sample: bool = True) -> float:
        """
        Variance measures the average squared deviation from the mean.

        Formulae
        --------
        Population variance (σ²):
            σ² = Σ(xᵢ − μ)² / N

        Sample variance (s²) — Bessel's correction:
            s² = Σ(xᵢ − x̄)² / (N − 1)

        Bessel's correction (dividing by N-1 instead of N) compensates for
        the fact that a sample mean is computed *from* the same data, which
        causes naive (÷N) variance to systematically under-estimate the true
        population variance.  Dividing by N-1 produces an *unbiased* estimator.

        Parameters
        ----------
        is_sample : bool, default True
            True  → sample variance   (÷ N-1, Bessel's correction)
            False → population variance (÷ N)

        Returns
        -------
        float

        Raises
        ------
        EmptyDataError
            If is_sample=True but n == 1 (denominator would be zero).
        """
        if is_sample and self.n == 1:
            raise EmptyDataError()   # reuse; single point → undefined s²

        mean = self.get_mean()
        squared_deviations = [(x - mean) ** 2 for x in self.data]
        sum_sq_dev = sum(squared_deviations)

        denominator = (self.n - 1) if is_sample else self.n
        return sum_sq_dev / denominator

    def get_standard_deviation(self, is_sample: bool = True) -> float:
        """
        Standard deviation: the square root of variance.

            s  = √(sample variance)
            σ  = √(population variance)

        Parameters
        ----------
        is_sample : bool, default True
            Passed directly to get_variance().

        Returns
        -------
        float
        """
        return math.sqrt(self.get_variance(is_sample=is_sample))

    # ------------------------------------------------------------------
    # Outlier Detection
    # ------------------------------------------------------------------

    def get_outliers(self, threshold: float = 2.0) -> List[float]:
        """
        Identify data points whose z-score magnitude exceeds *threshold*.

        z-score: z = (xᵢ − x̄) / s

        A point is flagged as an outlier if |z| > threshold.

        The default threshold of 2 captures ~95 % of normally distributed
        data inside the boundary, flagging the extreme ~5 %.

        Parameters
        ----------
        threshold : float, default 2.0
            Number of standard deviations beyond which a point is an outlier.

        Returns
        -------
        list[float]
            Sorted list of outlier values (may be empty).

        Raises
        ------
        ValueError
            If threshold ≤ 0.
        """
        if threshold <= 0:
            raise ValueError("threshold must be a positive number.")

        # If all values are identical std dev == 0 → no outliers possible
        std = self.get_standard_deviation(is_sample=True)
        if std == 0:
            return []

        mean = self.get_mean()
        outliers = [x for x in self.data if abs(x - mean) / std > threshold]
        return sorted(outliers)

    # ------------------------------------------------------------------
    # Convenience / display
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dictionary with all key statistics for quick inspection."""
        return {
            "n"                   : self.n,
            "mean"                : self.get_mean(),
            "median"              : self.get_median(),
            "mode"                : self.get_mode(),
            "sample_variance"     : self.get_variance(is_sample=True),
            "population_variance" : self.get_variance(is_sample=False),
            "sample_std_dev"      : self.get_standard_deviation(is_sample=True),
            "population_std_dev"  : self.get_standard_deviation(is_sample=False),
            "outliers_2std"       : self.get_outliers(threshold=2.0),
        }

    def __repr__(self) -> str:
        preview = self.data[:5]
        more = f"... +{self.n - 5} more" if self.n > 5 else ""
        return f"StatEngine(n={self.n}, data={preview}{more})"

