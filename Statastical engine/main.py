"""
main.py
=======
Entry point for the Statistical Engineering & Simulation project.

Run with:
    python main.py

This script:
  1. Loads the mock salary dataset.
  2. Runs a full StatEngine analysis, including outlier detection.
  3. Explains why the mean alone is misleading for startup salaries.
  4. Runs the Monte Carlo server-crash simulation across multiple trial sizes
     to demonstrate the Law of Large Numbers.
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — makes the project runnable from the repo root or from
# inside the statistical_engine/ folder.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.stat_engine import StatEngine
from src.monte_carlo import run_lln_demonstration


# ===========================================================================
# Helpers
# ===========================================================================

def _fmt(value) -> str:
    """Format a float as a readable currency string."""
    if isinstance(value, float):
        return f"${value:,.2f}"
    return str(value)


def _section(title: str) -> None:
    width = 65
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


# ===========================================================================
# Part 1 — Salary analysis
# ===========================================================================

def run_salary_analysis() -> None:
    _section("PART 1 — Startup Salary Dataset Analysis")

    # Load data
    data_path = os.path.join(BASE_DIR, "data", "sample_salaries.json")
    with open(data_path, "r") as fh:
        payload = json.load(fh)

    salaries = payload["salaries"]
    print(f"\nLoaded {len(salaries)} salary records from '{data_path}'")
    print(f"Raw data preview (first 10): {salaries[:10]} …")

    engine = StatEngine(salaries)
    stats = engine.summary()

    print(f"\n{'─'*65}")
    print(f"  n (observations)          : {stats['n']}")
    print(f"  Mean salary               : {_fmt(stats['mean'])}")
    print(f"  Median salary             : {_fmt(stats['median'])}")
    print(f"  Mode                      : {stats['mode']}")
    print(f"  Sample Variance           : {_fmt(stats['sample_variance'])}")
    print(f"  Population Variance       : {_fmt(stats['population_variance'])}")
    print(f"  Sample Std Dev            : {_fmt(stats['sample_std_dev'])}")
    print(f"  Population Std Dev        : {_fmt(stats['population_std_dev'])}")
    print(f"{'─'*65}")

    # Outlier analysis
    outliers_2 = engine.get_outliers(threshold=2.0)
    outliers_3 = engine.get_outliers(threshold=3.0)

    print(f"\n  Outliers (> 2 std from mean) [{len(outliers_2)} found]:")
    for o in outliers_2:
        print(f"    {_fmt(o)}")

    print(f"\n  Outliers (> 3 std from mean) [{len(outliers_3)} found]:")
    for o in outliers_3:
        print(f"    {_fmt(o)}")

    # -----------------------------------------------------------------------
    # Qualitative interpretation
    # -----------------------------------------------------------------------
    mean   = stats["mean"]
    median = stats["median"]
    std    = stats["sample_std_dev"]

    print(
        f"\n{'─'*65}\n"
        f"  📈  WHY MEAN IS DANGEROUS FOR STARTUP SALARIES\n"
        f"{'─'*65}\n"
        f"  Mean   : {_fmt(mean)}\n"
        f"  Median : {_fmt(median)}\n"
        f"  Std Dev: {_fmt(std)}\n\n"
        f"  The mean salary of {_fmt(mean)} is inflated dramatically by a\n"
        f"  handful of executive earners (CEO: $12M, CTO: $3.5M, etc.).\n"
        f"  In contrast, the *median* salary of {_fmt(median)} represents\n"
        f"  the 'typical' employee far more honestly — more than half the\n"
        f"  workforce earns below this figure.\n\n"
        f"  The standard deviation of {_fmt(std)} tells us the data is\n"
        f"  wildly dispersed: a one-standard-deviation band stretches from\n"
        f"  roughly {_fmt(max(0, mean - std))} to {_fmt(mean + std)}, a range\n"
        f"  that is larger than the median itself.  This is the hallmark of\n"
        f"  a heavily right-skewed distribution driven by extreme outliers.\n\n"
        f"  Key takeaway: when a distribution is skewed, the median + std dev\n"
        f"  (or IQR) paint the real picture; the mean alone is misleading.\n"
    )


# ===========================================================================
# Part 2 — Monte Carlo / LLN
# ===========================================================================

def run_server_simulation() -> None:
    _section("PART 2 — Monte Carlo Server Crash Simulation (LLN)")
    run_lln_demonstration(trial_sizes=[30, 365, 1_000, 10_000], seed=42)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "★" * 65)
    print("  Statistical Engineering & Simulation — Full Report")
    print("★" * 65)

    run_salary_analysis()
    run_server_simulation()

    print("\n✅  Analysis complete.")

