"""Sample Heston parameters, simulate histories, and save synthetic datasets.

Rows represent different parameter draws, not Monte Carlo replicates of one
fixed parameter vector. Time is in years; v denotes variance, not volatility.
Only NumPy and the Python standard library are required by this module.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np

PARAM_NAMES = ["v_0", "kappa", "theta", "xi", "rho"]
PARAM_RANGES = {
    "v_0": (0.05, 0.50),
    "kappa": (0.05, 5.00),
    "theta": (0.01, 0.50),
    "xi": (0.05, 1.50),
    "rho": (-0.95, 0.10),
}
LOG_PARAMS = ["v_0", "kappa", "theta", "xi"]
MATURITIES = np.array([0.05, 0.10, 0.25, 0.50, 1.00, 2.00])
Z_GRID = np.array([-2.0, -1.25, -0.5, 0.0, 0.5, 1.25, 2.0])


def _positive_integer(value: int, name: str) -> None:
    """Reject non-integer counts, including booleans."""

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")


def make_prior_spec(
    ranges: dict[str, tuple[float, float]] | None = None,
    log_params: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct Gaussian specifications in transformed coordinates.

    Parameters
    ----------
    ranges : dict, optional
        Physical (lower, upper) bounds for every name in PARAM_NAMES.
        Defaults to PARAM_RANGES; log-transformed bounds must be positive.
    log_params : list of str, optional
        Parameters transformed with log; defaults to the first four names.

    Returns
    -------
    transformed_bounds : np.ndarray
        Shape (5, 2), with lower and upper transformed bounds.
    norm_params : np.ndarray
        Shape (5, 2), with columns mu and sigma of the UNTRUNCATED normal.
        The sampler subsequently truncates this normal to the bounds.
        The standard deviation after truncation is smaller than sigma.
    """

    ranges = PARAM_RANGES if ranges is None else ranges
    log_params = LOG_PARAMS if log_params is None else log_params

    if set(ranges) != set(PARAM_NAMES) or not set(log_params) <= set(
        PARAM_NAMES
    ):
        raise ValueError(
            "Prior keys must match PARAM_NAMES; log names must be known."
        )

    bounds = np.asarray(
        [ranges[name] for name in PARAM_NAMES], dtype=np.float64
    )

    if bounds.shape != (5, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("Bounds must be a finite (5, 2) array.")

    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError(
            "Every lower bound must be smaller than its upper bound."
        )

    mask = np.array([name in log_params for name in PARAM_NAMES])

    if np.any(bounds[mask] <= 0):
        raise ValueError("Log-transformed bounds must be strictly positive.")

    transformed = bounds.copy()  # Do not overwrite the physical bounds.
    transformed[mask, :] = np.log(transformed[mask, :])
    lo, hi = transformed[:, 0], transformed[:, 1]
    norm_params = np.column_stack(((lo + hi) / 2, (hi - lo) / 2))

    return transformed, norm_params


def sample_parameters(
    n_samples: int,
    rng: np.random.Generator,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Draw independent Heston parameter vectors from the bounded prior.

    Parameters
    ----------
    n_samples : int
        Positive number of training examples to generate.
    rng : np.random.Generator
        Caller-owned generator; reuse it rather than reseeding inside loops.
    ranges : dict, optional
        Physical bounds in PARAM_NAMES order, using the fixed LOG_PARAMS rule.

    Returns
    -------
    parameters : np.ndarray
        Shape (n_samples, 5), in physical coordinates. Each row contains
        v_0, kappa, theta, xi, rho. These parameters stay fixed along its path.
    """

    _positive_integer(n_samples, "n_samples")
    bounds, normal = make_prior_spec(ranges)
    # Rejection sampling preserves the truncated-normal distribution.
    # Clipping normal draws would instead put artificial mass on the bounds.
    z = rng.standard_normal((n_samples, 5))
    rejected = np.abs(z) > 1.0

    while np.any(rejected):
        z[rejected] = rng.standard_normal(np.count_nonzero(rejected))
        rejected = np.abs(z) > 1.0

    parameters = normal[:, 0] + normal[:, 1] * z
    mask = np.array([name in LOG_PARAMS for name in PARAM_NAMES])
    parameters[:, mask] = np.exp(parameters[:, mask])

    return parameters


def make_strike_grid(
    maturities: np.ndarray,
    z_grid: np.ndarray,
    norm_params: np.ndarray | None = None,
    S_0: float = 1.0,
    drift: float = 0.0,
) -> np.ndarray:
    """Return reference strikes, shape (n_maturities, n_z), not option prices.

    maturities is a positive 1-D array in years; z_grid is a finite 1-D
    array of standardised offsets. norm_params is the (5, 2) prior matrix
    returned by make_prior_spec, or the default specification when omitted.
    S_0 is positive initial spot; drift is the annual reference growth rate.
    A common prior-centre volatility defines the grid, not each hidden v_0.
    """

    maturities = np.asarray(maturities, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)

    if (
        maturities.ndim != 1
        or maturities.size == 0
        or np.any(maturities <= 0)
        or not np.all(np.isfinite(maturities))
    ):
        raise ValueError(
            "maturities must be a nonempty positive finite vector."
        )

    if z_grid.ndim != 1 or z_grid.size == 0 or not np.all(np.isfinite(z_grid)):
        raise ValueError("z_grid must be a nonempty finite vector.")

    if not np.isfinite(S_0) or S_0 <= 0 or not np.isfinite(drift):
        raise ValueError("S_0 must be positive and drift must be finite.")

    normal = (
        make_prior_spec()[1]
        if norm_params is None
        else np.asarray(norm_params)
    )

    if normal.shape != (5, 2) or not np.all(np.isfinite(normal)):
        raise ValueError("norm_params must be finite with shape (5, 2).")

    sigma_ref = np.sqrt(np.exp(normal[0, 0]))
    T = maturities[:, None]  # Column: one row per maturity.
    z = z_grid[None, :]  # Row: one column per offset.

    return S_0 * np.exp(drift * T + sigma_ref * np.sqrt(T) * z)


def make_euler_grid(
    maturities: np.ndarray,
    steps_per_year: int = 2000,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Construct a uniform Euler grid without silently rounding maturities.

    Parameters
    ----------
    maturities : np.ndarray
        Nonempty, strictly increasing positive maturities in years.
        Each must align with the selected time-step resolution.
    steps_per_year : int
        Positive number of equal time steps in one year.

    Returns
    -------
    time_grid : np.ndarray
        Times from zero to the final maturity, shape (n_steps + 1,).
    dt : float
        Time-step length in years: 1 / steps_per_year.
    maturity_indices : np.ndarray
        Integer indices locating the supplied maturities on time_grid.
    """

    _positive_integer(steps_per_year, "steps_per_year")
    maturities = np.asarray(maturities, dtype=float)

    if (
        maturities.ndim != 1
        or maturities.size == 0
        or not np.all(np.isfinite(maturities))
    ):
        raise ValueError("maturities must be a nonempty finite vector.")

    if np.any(maturities <= 0) or np.any(np.diff(maturities) <= 0):
        raise ValueError(
            "maturities must be positive and strictly increasing."
        )

    dt = 1.0 / steps_per_year
    scaled = maturities * steps_per_year

    if not np.allclose(scaled, np.rint(scaled), rtol=0, atol=1e-8):
        raise ValueError(
            "Maturities do not align with this grid; increase steps_per_year."
        )

    indices = np.rint(scaled).astype(int)

    return np.arange(indices[-1] + 1) * dt, dt, indices


def correlated_increments(
    n_paths: int,
    dt: float,
    rho: np.ndarray | float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one step of correlated Brownian increments.

    Parameters
    ----------
    n_paths : int
        Positive number of paths; one random draw pair per path.
    dt : float
        Positive time-step length in years.
    rho : np.ndarray or float
        Correlation per path, shape (n_paths,), or a shared scalar, in [-1, 1].
    rng : np.random.Generator
        Caller-owned generator. Do not reset its seed at every time step.

    Returns
    -------
    dW_S, dW_v : np.ndarray
        Two 1-D arrays, each shape (n_paths,). Both have theoretical mean
        zero and variance dt; each pair has correlation rho for its path.
    """

    _positive_integer(n_paths, "n_paths")
    rho = np.asarray(rho, dtype=float)

    if (
        rho.shape not in ((), (n_paths,))
        or not np.all(np.isfinite(rho))
        or np.any(np.abs(rho) > 1)
    ):
        raise ValueError(
            "rho must be a scalar or (n_paths,) vector in [-1, 1]."
        )

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite.")

    z = rng.standard_normal(
        (n_paths, 2)
    )  # Rows are paths; columns are shocks.
    sqrt_dt = np.sqrt(dt)
    dW_S = sqrt_dt * z[:, 0]
    dW_v = sqrt_dt * (rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1])

    return dW_S, dW_v


def und_and_var_step(
    x: np.ndarray,
    v: np.ndarray,
    kappa: np.ndarray,
    theta: np.ndarray,
    xi: np.ndarray,
    drift: float,
    dt: float,
    dW_S: np.ndarray,
    dW_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance log-price and raw variance by one full-truncation Euler step.

    Parameters
    ----------
    x, v : np.ndarray
        Current log-price and raw numerical variance, shape (n_paths,).
    kappa, theta, xi : np.ndarray
        Mean-reversion speed, long-run variance and volatility of variance,
        each shape (n_paths,). The simulator validates these inputs once.
    drift : float
        Shared annual price drift; use r - q for risk-neutral dynamics.
    dt : float
        Positive time-step length in years.
    dW_S, dW_v : np.ndarray
        Brownian increments of shape (n_paths,), already scaled by sqrt(dt).

    Returns
    -------
    x_next, v_next : np.ndarray
        Updated log-prices and RAW variance states, each shape (n_paths,).
        Raw variance can be negative. Keep it for the next numerical step.
    """

    v_positive = np.maximum(v, 0.0)
    sqrt_v = np.sqrt(v_positive)
    # Both updates use the old variance, not the newly updated variance.
    x_next = x + (drift - 0.5 * v_positive) * dt + sqrt_v * dW_S
    v_next = v + kappa * (theta - v_positive) * dt + xi * sqrt_v * dW_v

    return x_next, v_next


def simulate_heston_paths(
    parameters: np.ndarray,
    time_grid: np.ndarray,
    rng: np.random.Generator,
    S_0: float = 1.0,
    drift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one joint history per parameter row, vectorised across rows.

    Parameters
    ----------
    parameters : np.ndarray
        Shape (n_paths, 5), columns v_0, kappa, theta, xi, rho.
        First four columns must be nonnegative; rho must be in [-1, 1].
    time_grid : np.ndarray
        Uniform increasing times in years starting at zero; at least 2 entries.
    rng : np.random.Generator
        Generator for fresh shocks at every step.
    S_0 : float
        Shared positive initial price.
    drift : float
        Shared annual price drift; this demo uses zero.

    Returns
    -------
    prices : np.ndarray
        Underlying prices, shape (n_paths, n_times), including initial values.
    variances : np.ndarray
        Effective nonnegative variances with the same shape. Internal raw
        Euler states are retained during simulation but not returned.
        These outputs are underlying histories, NOT option-price histories.
    """

    parameters = np.asarray(parameters, dtype=np.float64)
    times = np.asarray(time_grid, dtype=np.float64)

    if (
        parameters.ndim != 2
        or parameters.shape[0] == 0
        or parameters.shape[1] != 5
    ):
        raise ValueError(
            "parameters must have shape (n_paths, 5), with n_paths > 0."
        )

    if (
        not np.all(np.isfinite(parameters))
        or np.any(parameters[:, :4] < 0)
        or np.any(np.abs(parameters[:, 4]) > 1)
    ):
        raise ValueError(
            "Invalid Heston parameters: check finiteness, signs and rho."
        )

    if (
        times.ndim != 1
        or times.size < 2
        or not np.all(np.isfinite(times))
        or times[0] != 0
    ):
        raise ValueError("time_grid must be a finite vector starting at zero.")

    steps = np.diff(times)

    if np.any(steps <= 0) or not np.allclose(
        steps, steps[0], rtol=1e-9, atol=1e-12
    ):
        raise ValueError("time_grid must be strictly increasing and uniform.")

    if not np.isfinite(S_0) or S_0 <= 0 or not np.isfinite(drift):
        raise ValueError("S_0 must be positive and drift finite.")

    n_paths = len(parameters)
    v_0, kappa, theta, xi, rho = parameters.T
    x = np.full(n_paths, np.log(S_0))
    v = v_0.copy()
    prices = np.empty((n_paths, len(times)), dtype=np.float64)
    variances = np.empty_like(prices)
    prices[:, 0], variances[:, 0] = S_0, v_0

    # Time is sequential; all parameter rows advance simultaneously.

    for step in range(1, len(times)):
        dW_S, dW_v = correlated_increments(n_paths, steps[0], rho, rng)
        x, v = und_and_var_step(
            x, v, kappa, theta, xi, drift, steps[0], dW_S, dW_v
        )
        prices[:, step] = np.exp(x)
        variances[:, step] = np.maximum(v, 0.0)

    if (
        not np.all(np.isfinite(prices))
        or np.any(prices <= 0)
        or not np.all(np.isfinite(variances))
    ):
        raise FloatingPointError(
            "Simulation overflow/underflow: inspect parameters and resolution."
        )

    return prices, variances


@dataclass
class HestonSample:
    """Dataset container: parameter rows, time columns, and generation metadata.

    parameters: (n_samples, 5); times: (n_times,); prices and variances:
    (n_samples, n_times). seed, S_0 and drift document reproducibility.
    Variances are effective clipped outputs, not raw restart states.
    """

    parameters: np.ndarray
    times: np.ndarray
    prices: np.ndarray
    variances: np.ndarray
    seed: int
    S_0: float
    drift: float

    def log_returns(self) -> np.ndarray:
        """Return successive log-price differences, shape (n_samples, n_times-1)."""

        return np.diff(np.log(self.prices), axis=1)

    def summary(self) -> np.ndarray:
        """Return columns terminal S, total log return, mean v and max drawdown.

        Drawdown is the largest fractional fall from a previous sampled price
        peak. Mean variance is a simple average of the stored grid values.
        These are descriptive path statistics, not estimates of prior moments.
        """

        running_peak = np.maximum.accumulate(self.prices, axis=1)
        drawdown = np.max(1 - self.prices / running_peak, axis=1)

        return np.column_stack(
            (
                self.prices[:, -1],
                np.log(self.prices[:, -1] / self.prices[:, 0]),
                self.variances.mean(axis=1),
                drawdown,
            )
        )


def generate_samples(
    n_samples: int = 2,
    maturities: np.ndarray | None = None,
    steps_per_year: int = 1000,
    seed: int = 42,
    S_0: float = 1.0,
    drift: float = 0.0,
) -> HestonSample:
    """Generate a reproducible small dataset with one parameter draw per row.

    n_samples controls rows; maturities defaults to [1.0] years and must align
    with steps_per_year. seed initialises a fresh local generator. S_0 and
    drift are shared initial spot and annual price drift. Returns HestonSample
    containing parameters, full paths, times and generation metadata.
    """

    rng = np.random.default_rng(seed)
    parameters = sample_parameters(n_samples, rng)
    times, _, _ = make_euler_grid(
        np.array([1.0]) if maturities is None else maturities, steps_per_year
    )
    prices, variances = simulate_heston_paths(
        parameters, times, rng, S_0, drift
    )

    return HestonSample(parameters, times, prices, variances, seed, S_0, drift)


def save_sample(sample: HestonSample, path: str | Path) -> Path:
    """Save a sample and its metadata to a compressed .npz file; return its path.

    sample is a HestonSample. path must end in .npz; missing parent directories
    are created. Numeric arrays avoid pickle and retain float64 precision.
    """

    path = Path(path)

    if path.suffix != ".npz":
        raise ValueError("Dataset path must end in .npz.")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        parameters=sample.parameters,
        times=sample.times,
        prices=sample.prices,
        variances=sample.variances,
        seed=sample.seed,
        S_0=sample.S_0,
        drift=sample.drift,
        parameter_names=np.array(PARAM_NAMES),
        schema_version=1,
    )

    return path


def load_sample(path: str | Path) -> HestonSample:
    """Load a .npz dataset made by save_sample without enabling pickle."""

    with np.load(path, allow_pickle=False) as data:
        if (
            int(data["schema_version"]) != 1
            or data["parameter_names"].tolist() != PARAM_NAMES
        ):
            raise ValueError("Unsupported dataset schema or parameter order.")

        return HestonSample(
            data["parameters"],
            data["times"],
            data["prices"],
            data["variances"],
            int(data["seed"]),
            float(data["S_0"]),
            float(data["drift"]),
        )


if __name__ == "__main__":
    sample = generate_samples()
    print("Columns:", PARAM_NAMES)
    print(sample.parameters)
    print(
        "Price and variance shapes:",
        sample.prices.shape,
        sample.variances.shape,
    )
