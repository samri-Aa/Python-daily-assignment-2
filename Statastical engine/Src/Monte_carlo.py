"""
monte_carlo.py
==============
Monte Carlo simulation of a startup server's daily crash probability.

Theoretical crash probability: 4.5% (p = 0.045) per day.

The simulation demonstrates the Law of Large Numbers (LLN):
  As the number of trials n → ∞, the observed (empirical) probability
  of an event converges to the event's true theoretical probability.

Only standard-library modules are used: ``random``.
"""

import random
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

THEORETICAL_CRASH_PROB: float = 0.045   # 4.5 % — the ground truth


def simulate_crashes(days: int, seed: int = None) -> Tuple[int, float]:
    """
    Simulate *days* of server operation, each day having a 4.5 % chance of
    crashing.

    Parameters
    ----------
    days : int
        Number of days to simulate.  Must be ≥ 1.
    seed : int, optional
        Random seed for reproducibility.  Default None (non-deterministic).

    Returns
    -------
    (total_crashes, simulated_probability) : Tuple[int, float]
        total_crashes         – raw count of crash events
        simulated_probability – total_crashes / days

    Raises
    ------
    ValueError
        If days < 1.

    Algorithm
    ---------
    Each day, draw a uniform random float in [0, 1).
    If the value < THEORETICAL_CRASH_PROB (0.045), the server crashes that day.
    This is a Bernoulli trial with p = 0.045.
    """
    if days < 1:
        raise ValueError(f"days must be ≥ 1, got {days}.")

    rng = random.Random(seed)          # isolated RNG so callers don't pollute global state
    total_crashes: int = 0

    for _ in range(days):
        if rng.random() < THEORETICAL_CRASH_PROB:
            total_crashes += 1

    simulated_probability: float = total_crashes / days
    return total_crashes, simulated_probability


# ---------------------------------------------------------------------------
# LLN demonstration
# ---------------------------------------------------------------------------

def run_lln_demonstration(
    trial_sizes: List[int] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Run simulate_crashes() at multiple trial sizes and print a formatted
    report that illustrates the Law of Large Numbers.

    Parameters
    ----------
    trial_sizes : list[int], optional
        Day-counts to test.  Defaults to [30, 365, 1_000, 10_000].
    seed : int
        Base seed; each trial gets seed + index so results are reproducible
        yet independent of each other.

    Returns
    -------
    list[dict]
        One record per trial size with keys:
          days, crashes, simulated_prob, theoretical_prob, absolute_error
    """
    if trial_sizes is None:
        trial_sizes = [30, 365, 1_000, 10_000]

    results: List[dict] = []

    header = (
        f"\n{'='*65}\n"
        f"  LAW OF LARGE NUMBERS — Server Crash Simulation\n"
        f"  Theoretical crash probability: {THEORETICAL_CRASH_PROB:.1%}\n"
        f"{'='*65}\n"
        f"{'Days':>10}  {'Crashes':>8}  {'Sim. Prob':>10}  "
        f"{'Theory':>8}  {'|Error|':>10}\n"
        f"{'-'*65}"
    )
    print(header)

    for idx, days in enumerate(trial_sizes):
        crashes, sim_prob = simulate_crashes(days, seed=seed + idx)
        error = abs(sim_prob - THEORETICAL_CRASH_PROB)

        record = {
            "days"             : days,
            "crashes"          : crashes,
            "simulated_prob"   : sim_prob,
            "theoretical_prob" : THEORETICAL_CRASH_PROB,
            "absolute_error"   : error,
        }
        results.append(record)

        print(
            f"{days:>10,}  {crashes:>8,}  {sim_prob:>10.4%}  "
            f"{THEORETICAL_CRASH_PROB:>8.4%}  {error:>10.4%}"
        )

    print(f"{'='*65}")
    _print_interpretation(results)
    return results


def _print_interpretation(results: List[dict]) -> None:
    """Print a qualitative LLN interpretation based on the simulation results."""

    smallest = results[0]
    largest  = results[-1]

    print(
        f"\n📊  INTERPRETATION — Law of Large Numbers\n"
        f"{'-'*65}\n"
        f"Theoretical probability (ground truth): {THEORETICAL_CRASH_PROB:.1%}\n"
    )

    print(
        f"At n = {smallest['days']:,} days:\n"
        f"  Simulated probability : {smallest['simulated_prob']:.4%}\n"
        f"  Absolute error        : {smallest['absolute_error']:.4%}\n"
        f"  → With only {smallest['days']} observations the estimate is highly\n"
        f"    volatile.  Random chance can produce 0 crashes (0.00%) or\n"
        f"    several crashes (>>4.5%), both far from the true rate.\n"
    )

    print(
        f"At n = {largest['days']:,} days:\n"
        f"  Simulated probability : {largest['simulated_prob']:.4%}\n"
        f"  Absolute error        : {largest['absolute_error']:.4%}\n"
        f"  → With {largest['days']:,} observations the simulated probability\n"
        f"    converges tightly to 4.50%, demonstrating LLN.\n"
    )

    print(
        f"⚠️   WHY 30-DAY DATA IS DANGEROUS FOR BUDGET PLANNING\n"
        f"{'-'*65}\n"
        "A startup that records crashes for only one month (30 days) may\n"
        "observe 0, 1, or 2 crashes — a range of 0%–6.7% estimated\n"
        "probability.  If they observed 0 crashes they might budget $0\n"
        "for maintenance; if 2 crashes, they might massively over-budget.\n"
        "Neither estimate is reliable.\n\n"
        "The LLN guarantees that the *long-run* frequency stabilises at\n"
        "4.5%, but a 30-day sample sits in the high-variance zone of the\n"
        "sampling distribution — the signal-to-noise ratio is simply too\n"
        "low.  A startup needs at minimum several hundred days of data\n"
        "(or a formal Bayesian prior) before a crash-rate estimate can be\n"
        "trusted for financial planning.\n"
    )
