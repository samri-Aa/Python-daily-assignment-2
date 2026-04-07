# Statistical Engineering & Simulation Engine

This project is a statistics engine and Monte Carlo simulator I built entirely from scratch using nothing but Python's standard library. No NumPy, no pandas, no shortcuts — just pure Python math implemented by hand.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Mathematical Logic](#mathematical-logic)
4. [Setup Instructions](#setup-instructions)
5. [Running the Project](#running-the-project)
6. [Testing](#testing)
7. [Acceptance Criteria Checklist](#acceptance-criteria-checklist)

---

## Project Overview

The project is split into two core pieces that work together.

The first is `StatEngine` — a class that takes a list of numbers and lets you compute all the standard descriptive statistics: mean, median, mode, variance, standard deviation, and outlier detection. I wrote every formula by hand, including Bessel's correction for sample variance and the z-score method for outliers. It also handles bad input gracefully, so things like empty lists, mixed types, or None values won't silently break anything.

The second is a Monte Carlo server crash simulation. The idea is simple: a startup's server has a 4.5% chance of crashing on any given day. I simulate that over different time windows (30 days, 1 year, 1,000 days, 10,000 days) to show how the observed crash rate gets closer and closer to the true 4.5% as the sample size grows. This is the Law of Large Numbers in action.

The salary dataset (`data/sample_salaries.json`) has 50 intentionally skewed startup salaries — ranging from junior staff around $42k all the way up to a CEO at $12M — to demonstrate why relying only on the mean can be really misleading.

---

## Folder Structure

```
statistical_engine/
│
├── data/
│   └── sample_salaries.json       # 50 mock startup salaries, heavily skewed
│
├── src/
│   ├── __init__.py                # Makes src a package, exposes key classes
│   ├── stat_engine.py             # The StatEngine class — all stats from scratch
│   └── monte_carlo.py             # Server crash simulation and LLN demo
│
├── tests/
│   ├── __init__.py
│   └── test_stat_engine.py        # 56 unit tests covering all edge cases
│
├── README.md
└── main.py                        # Run this to see the full analysis and simulation
```

---

## Mathematical Logic

### Variance and Bessel's Correction

There are two versions of variance depending on whether your data is the full population or just a sample.

**Population Variance (σ²)** is used when you have every data point — you divide by N:

```
σ² = Σ(xᵢ − μ)² / N
```

**Sample Variance (s²)** is used when your data is a sample from a bigger population. The trick here is that your sample mean is calculated from the same data, which causes the plain ÷N formula to slightly underestimate the real spread. Dividing by N−1 (called Bessel's correction) fixes that bias:

```
s² = Σ(xᵢ − x̄)² / (N − 1)
```

Standard deviation in both cases is just the square root of variance.

### Median — Even vs. Odd

I sort the data first, then check whether the count is odd or even:

| n | What I do | Example |
|---|---|---|
| **Odd** | Take the middle element: `data[n // 2]` | `[1, 2, 3, 4, 5]` → **3** |
| **Even** | Average the two middle elements: `(data[n//2 − 1] + data[n//2]) / 2` | `[1, 2, 3, 4]` → **2.5** |

### Outlier Detection (Z-Score)

For each data point I calculate how many standard deviations away from the mean it sits:

```
z = (xᵢ − x̄) / s
```

If the absolute z-score is greater than the threshold (default is 2.0), the point gets flagged as an outlier. At threshold 2, about 95% of normally distributed data falls inside the boundary, so only the genuinely extreme values get caught.

### Law of Large Numbers

The LLN says that the more trials you run, the closer your observed result will get to the true theoretical probability. In my simulation:

| Days | Simulated Prob | Theoretical | Error |
|------|----------------|-------------|-------|
| 30   | varies widely  | 4.50%       | high  |
| 10,000 | ≈ 4.50%     | 4.50%       | < 0.5% |

At 30 days you might see 0 crashes or 3 crashes — both are plausible, but neither tells you much. At 10,000 days the noise averages out and you land very close to 4.5% every time.

---

## Setup Instructions

You only need Python 3.8 or higher. There are no packages to install.

```bash
# Clone the repo
git clone https://github.com/samrawit-alemeshet/statistical-engine.git
cd statistical-engine/statistical_engine

# Run the analysis
python main.py
```

No `pip install`, no virtual environment, nothing extra.

---

## Running the Project

```bash
python main.py
```

The output walks through four things:
1. Full statistical summary of the salary dataset
2. Outlier analysis with interpretation
3. Explanation of why the mean is misleading for skewed data
4. Monte Carlo simulation results across 30, 365, 1,000, and 10,000 days with LLN commentary

---

## Testing

```bash
# Run all tests with detailed output
python -m unittest discover -s tests -v

# Or run directly
python tests/test_stat_engine.py
```

There are 56 tests total. Here's what they cover:

| Category | What's tested |
|---|---|
| Instantiation & cleaning | Valid input, numeric strings rejected, None rejected, booleans rejected, empty lists |
| `get_mean()` | Simple cases, negatives, floats, 100-element dataset |
| `get_median()` | Odd n, even n, duplicates, single element, reverse-sorted input |
| `get_mode()` | Unimodal, multimodal, all-unique message |
| `get_variance()` | Known population result (4.0), Bessel's correction, s² > σ², single-element edge case |
| `get_standard_deviation()` | Known value (2.0), √variance identity, always non-negative |
| `get_outliers()` | Outliers caught, no false positives, bad threshold raises, zero std dev handled |
| Monte Carlo | Return type, value range, LLN convergence at n=100k, seed reproducibility |
| Error messages | EmptyDataError says "empty", InvalidDataTypeError lists the bad values |

---

## Acceptance Criteria Checklist

- [x] **Empty list handling** — `StatEngine([])` raises `EmptyDataError` with a clear message
- [x] **Mixed type handling** — `StatEngine([1, 2, '3', None])` raises `InvalidDataTypeError` and names the bad values
- [x] **Boolean rejection** — `True` and `False` are explicitly blocked (they're secretly ints in Python)
- [x] **Multimodal mode** — `get_mode()` returns all tied modes as a sorted list
- [x] **All-unique mode** — Returns `"No mode: all values are unique."` instead of crashing or returning an empty list
- [x] **Sample variance (Bessel's)** — Divides by N−1, verified against the known result 32/7
- [x] **Population variance** — Divides by N, verified against the known result 4.0
- [x] **s² > σ² always holds** — Confirmed by test for any n > 1
- [x] **Median odd n** — Returns the single middle element
- [x] **Median even n** — Returns the average of the two central elements
- [x] **Standard deviation = √variance** — Verified to 12 decimal places
- [x] **Outlier threshold validation** — `threshold ≤ 0` raises `ValueError`
- [x] **Identical values produce no outliers** — When std dev is 0, the method returns an empty list safely
- [x] **Monte Carlo LLN convergence** — At n = 100,000, simulated probability lands within 0.5% of 4.50%
- [x] **Reproducible simulation** — The same seed always produces the same result
- [x] **Zero external dependencies** — Only uses `math`, `random`, `typing`, `unittest`, `json`, `os`, and `sys`

---

## Author

**Samrawit Alemeshet**

This was built as part of a Statistical Engineering & Simulation Assessment. Every formula is implemented from first principles — I didn't use any statistical libraries because the whole point was to understand the math well enough to write it myself.
