# Heston Micro

A small MA52109 programming project: sample Heston parameters, generate two
synthetic underlying/variance histories, describe them numerically, and plot
them using reusable NumPy and Matplotlib functions.

**These are underlying histories, not option-price paths.** Option pricing and
posterior inference are deliberately outside this micro-project.

## Installation and demonstration

From this directory, with Python 3.10 or newer:

```bash
python -m pip install -e .
python demo/demo_heston.py
python -m unittest discover -s tests -v
```

The demo saves data to `data/` and four plots to `demo/figures/`. Add `--show`
to display windows. Paths are resolved from the demo file, so after installation
the script can also be launched from a different working directory.

For the notebook, install optional notebook tools if needed:

```bash
python -m pip install -e ".[notebook]"
jupyter notebook demo/two_heston_samples.ipynb
```

With uv, the equivalent setup is `uv sync --extra notebook`, then
`uv run python demo/demo_heston.py`. Select that project's kernel in your editor.
Jupyter is only the optional presentation environment; package runtime code
depends on the standard library, NumPy and Matplotlib only.

## Package layout

```text
heston_micro/
    __init__.py
    data_generation.py     # prior, grid, increments, update, paths, file I/O
    visualisations.py      # line, histogram, scatter and drawdown plots
demo/
    demo_heston.py          # terminal demonstration
    two_heston_samples.ipynb
    figures/
data/                      # bundled synthetic CSV and NPZ datasets
tests/
pyproject.toml
README.md
```

## How the simulation works

Each row receives its own parameter vector `[v_0, kappa, theta, xi, rho]`.
Parameters stay fixed along that row's history. Positive parameters are sampled
from normals in log coordinates, truncated at one underlying-normal standard
deviation either side of the mean, and exponentiated. Rho uses a truncated normal
in its original coordinate. Rejection sampling, not clipping, enforces bounds.

For each step, two independent standard normals produce correlated Brownian
increments. All rows advance together, while time advances sequentially.
Writing `v_plus = max(v, 0)`, the updates are:

```text
x_next = x + (drift - 0.5*v_plus)*dt + sqrt(v_plus)*dW_S
v_next = v + kappa*(theta - v_plus)*dt + xi*sqrt(v_plus)*dW_v
S_next = exp(x_next)
```

The internal variance is the raw full-truncation Euler state and can be negative.
The exported variance is its nonnegative part; it cannot be used to reconstruct
an exact numerical restart. The default drift is zero and initial price is one.
These can be interpreted as risk-neutral dynamics with zero rates and dividends,
but are not calibrated historical market data. Variance is not volatility:
volatility is its square root. No Feller-condition rejection is applied.

## Main functions

| Function | Purpose / shape |
|---|---|
| `make_prior_spec()` | Transformed bounds and normal parameters, each `(5, 2)` |
| `sample_parameters(n_samples, rng)` | Physical parameter matrix `(n_samples, 5)` |
| `make_euler_grid(maturities, steps_per_year)` | Times, dt, maturity indices; rejects off-grid maturities |
| `correlated_increments(n_paths, dt, rho, rng)` | Two vectors of increments `(n_paths,)` |
| `und_and_var_step(...)` | One joint log-price/raw-variance update |
| `simulate_heston_paths(parameters, time_grid, rng)` | Two arrays `(n_samples, n_times)` |
| `generate_samples(...)` | Convenient reproducible `HestonSample` container |
| `sample.log_returns()` / `sample.summary()` | Vectorised descriptive statistics |
| `save_sample(...)` / `load_sample(...)` | Compressed numeric dataset round trip |
| `make_strike_grid(...)` | Retained reference-grid utility; does not price options |

Example:

```python
from heston_micro import generate_samples
from heston_micro.visualisations import plot_paths
import matplotlib.pyplot as plt

sample = generate_samples(n_samples=2, seed=42, steps_per_year=1000)
fig, axes = plot_paths(sample)
plt.show()
```

`HestonSample` replaces the unused per-parameter dataclass in the draft: it stores
the completed dataset and supplies useful analysis methods. Random generators
are explicit arguments in low-level functions, not shared mutable defaults.

## Plots and interpretation

- Line plots compare price and effective variance histories.
- Histograms compare within-path one-step log returns using shared bins.
- Scatter plots show price/variance states coloured by observation time.
- Drawdowns show distance below each path's previous sampled maximum.

Two histories demonstrate the code, not a statistically representative model
comparison. Parameters AND random shocks differ between rows, so visual
differences cannot be attributed to a single parameter. Level scatter is not
an estimate of rho. Histograms pool time increments, not independent parameter
draws. The finite-step scheme has discretisation bias; no financial accuracy
or predictive-performance claim is made.

## Course connections

- LecNotesA section 2.1: modules and imports; 2.3: file handling; 2.4: exceptions.
- LecNotesA section 3.1: arrays, slicing, masks, broadcasting and RNGs.
- LecNotesA section 3.2: line/scatter/histogram plots and Figure/Axes objects.
- LecNotesB: review changes, commit the package and identify a reproducible
  submission commit. No remote repository is changed by these scripts.

The financial model and truncated-prior design come from the student's supplied
draft and project discussion, not from claims that the lectures teach Heston.
This completed draft was prepared with AI assistance from that code. Review it,
understand it and disclose assistance according to the course's requirements.

## Reproducibility and checks

The demo records its seed, NumPy version, grid and model settings in
`data/metadata.json`. Repeating the same seed and configuration in the same
environment reproduces the sample. Changing batch size changes RNG assignment;
exact results are not promised across arbitrary versions or batch layouts.

Tests cover bounded-prior sampling, grid alignment, Brownian moments and
correlation, raw negative-variance handling, a deterministic limit, constant-
variance price expectation, reproducibility and saved-data round trips.

## Submission

This folder is a self-contained local package; it has not been published or
submitted. Copy it to your chosen submission repository/branch, review the
changes, run the demo and tests, then commit and push. Use the exact final commit
URL if required by Lab 02. Keep the bundled synthetic `data/` files tracked;
do not apply the research repository's blanket data exclusion to this demo.
