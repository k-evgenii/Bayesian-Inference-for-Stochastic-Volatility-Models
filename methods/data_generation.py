"""Synthetic data generation for the Heston model."""

from dataclasses import dataclass
import numpy as np

# ======================================================================
# Heston parameter definitions
# ======================================================================
# Parameter order and prior bounds
PARAM_NAMES = [
    "v_0",
    "kappa",
    "theta",
    "xi",
    "rho"
]


PARAM_RANGES = {
    "v_0":   (0.05, 0.50),
    "kappa": (0.05, 5.00),
    "theta": (0.01, 0.50),
    "xi":    (0.05, 1.50),
    "rho":   (-0.95, 0.10),
}


LOG_PARAMS = [
    "v_0",
    "kappa",
    "theta",
    "xi",
]


#Writing options as a dataclass for reusability
# Prior container
@dataclass(frozen=True)
class ParameterPrior:
    mu: float
    sigma: float
    lower: float
    upper: float
    transform: str


#Functions

# ======================================================================
# prior specification
# ======================================================================
def make_prior_spec(
    ranges: dict[str, tuple[float, float]] = PARAM_RANGES,
    log_params: list[str] = LOG_PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct prior specifications from physical parameter bounds.
    
    Positive parameters
    -------------------

    For

        V in {v_0, kappa, theta, xi},

    define

        phi = log(V).

    We want the supplied parameter range

        [lo, hi]

    to correspond exactly to one standard deviation around the
    Gaussian mean in phi-space:

        log(lo) = mu - sigma
        log(hi) = mu + sigma

    Solving these two equations gives

        mu = [log(lo) + log(hi)] / 2

    and

        sigma = [log(hi) - log(lo)] / 2.


    rho
    ---

    rho is not transformed:

        phi_rho = rho.

    Therefore

        lo = mu - sigma
        hi = mu + sigma,

    giving

        mu = (lo + hi) / 2

        sigma = (hi - lo) / 2.


    Returns
    -------
    prior_spec : dict

        Dictionary containing, for each parameter:

            mu
            sigma
            transform
            lower
            upper

        Example:

            PRIOR_SPEC["v_0"]

        returns something of the form

            {
                "mu": ...,
                "sigma": ...,
                "transform": "log",
                "lower": 0.05,
                "upper": 0.50,
            }
    """
    
    #replacing the loop with vectorisation since nothing changes
    #the view is (1,5) matrix [True, True, True, True, False]
    log_bool_mask = np.array([name in log_params for name in PARAM_NAMES])

    #making the bounds into array
    #(5,2) : 
    bounds = np.array(
        [ranges[name] for name in PARAM_NAMES],
        dtype=float,
    )
    #Applying bool mask to select which params should be log transformed
    transformed_bounds = bounds.copy()
    transformed_bounds[log_bool_mask, :] = np.log(transformed_bounds[log_bool_mask, :])

    lo = transformed_bounds[:, 0]
    hi  = transformed_bounds[:, 1]

    mu = (lo + hi) / 2
    sigma = (hi - lo) / 2

    norm_params = np.empty(
        (len(PARAM_NAMES), 2),
        dtype = np.float64
    ) 
    
    norm_params[:, 0] = mu
    norm_params[:, 1] = sigma


    return transformed_bounds, norm_params


PRIOR_SPEC = make_prior_spec()


# ======================================================================
# defining the maturity grid
# ======================================================================
MATURITIES = np.array([
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
], dtype=float)


Z_GRID = np.array([
    -2.0,
    -1.25,
    -0.5,
     0.0,
     0.5,
     1.25,
     2.0,
], dtype=float)


def make_strike_grid(
    maturities: np.ndarray,
    z_grid: np.ndarray,
    norm_params: np.ndarray = PRIOR_SPEC[1],
    S_0: np.float64 = np.float64(1.0),
    drift: np.float64 = np.float64(0.0),
)-> None:    #np.ndarray:

    v0_centre = np.exp(norm_params[0,0])
    sigma_ref = np.sqrt(v0_centre)


    T = maturities[:, None]                             #shape (6,1)
    z = z_grid[None, :]                                 #shape (1,7)

    K_centre = S_0 * np.exp(drift * T)                  # (6, 1)
    exponent = np.sqrt(T) * z * sigma_ref               # (6, 7) 
    K = K_centre * np.exp(exponent)

    return K



# ======================================================================
# Euler-Maruyama time grid
# ======================================================================
def make_euler_grid(
    maturities: np.ndarray,
    steps_per_year: int = 2000,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Construct the Euler-Maruyama time grid.

    Returns
    -------
    time_grid:
        Full Euler grid from 0 to the largest maturity.

    dt:
        Euler time-step size.

    maturity_indices:
        Indices of the requested maturities on the Euler grid.
    """
    
    T_max = float(
        np.max(maturities)
    )

    dt = 1.0 / steps_per_year

    n_steps = int(
        round(T_max * steps_per_year)
    )

    time_grid = (
        np.arange(n_steps + 1) * dt
    )

    maturity_indices = np.rint(
        maturities / dt
    ).astype(int)

    return (
        time_grid,
        dt,
        maturity_indices,
    )


# ======================================================================
# Generate correlated Brownian increments
# ======================================================================
RNG = np.random.default_rng(seed=42)

def correlated_increments(
    n_paths: int,
    dt: float,
    rho: np.ndarray,
    rng: np.random.Generator = RNG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated Brownian increments for one time step.

    Each simulated path receives a price increment and a variance
    increment. Different paths use independent random draws.(and each row in the tuple represents differnt path)

    Parameters
    ----------
    n_paths : int
        Number of paths simulated simultaneously. Must be positive.

    sqrt_dt : float
        Square root of the time-step length: sqrt(dt).
        Scales standard normal draws into Brownian increments.
        Must be positive.

    rho : float
        Correlation between the price and variance increments.
        Mathematically valid between [-1, 1].
        This project's prior restricts sampling to [-0.95, 0.10].

    rng : np.random.Generator
        NumPy random-number generator used to produce the draws.
        Create it outside this function and reuse it across time steps.

    Returns
    -------
    dW_S : np.ndarray
        Brownian increments driving log-price, shape (n_paths,).
        Each increment has mean zero and variance dt.

    dW_v : np.ndarray
        Brownian increments driving variance, shape (n_paths,).
        Each increment has mean zero and variance dt.
        Its correlation with the corresponding dW_S increment is rho.

    Notes
    -----
    The stated moments and correlation describe the distribution.
    A finite sample will only approximate them.
    """
    z = rng.standard_normal((n_paths, 2))

    sqrt_dt = np.sqrt(dt)
    
    dW_S = sqrt_dt * z[:, 0]

    #make sure that rho is between -1 and 1.
    if not np.all((rho >= -1.0) & (rho <= 1.0)):
        raise ValueError("Every rho value must be between -1 and 1.")

    dW_v = sqrt_dt * (
        rho * z[:, 0]
        + np.sqrt(1.0 - rho ** 2) * z[:, 1]
    )

    return dW_S, dW_v


# ======================================================================
# Underlying(S_t) and variance(v_t) step function
# ======================================================================
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
    """
    Advance all Heston paths by one full-truncation Euler step.

    Parameters
    ----------
    x : np.ndarray
        Current log-prices, shape (n_paths,): x = log(S).
    v : np.ndarray
        Current raw numerical variance states, shape (n_paths,).
        These may become negative under the discretisation.
    kappa : np.ndarray
        Mean-reversion speeds, shape (n_paths,).
    theta : np.ndarray
        Long-run variance levels, shape (n_paths,).
    xi : np.ndarray
        Volatility-of-variance coefficients, shape (n_paths,).
    drift : float
        Shared underlying-price drift.
        For risk-neutral pricing, use interest rate minus dividend yield.
    dt : float
        Time-step length, using the same time units as the parameters.
    dW_S : np.ndarray
        Price Brownian increments, shape (n_paths,).
        Already include the sqrt(dt) scaling.
    dW_v : np.ndarray
        Variance Brownian increments, shape (n_paths,).
        Already include the sqrt(dt) scaling.

    Returns
    -------
    x_next : np.ndarray
        Updated log-prices, shape (n_paths,).
    v_next : np.ndarray
        Updated raw numerical variance states, shape (n_paths,).
    """
    # Use nonnegative variance in both drift and diffusion coefficients.
    v_positive = np.maximum(v, 0.0)

    # Convert effective variance into volatility.
    sqrt_v = np.sqrt(v_positive)

    # Update log-price using the current variance and price increment.
    x_next = (
        x
        + (drift - 0.5 * v_positive) * dt
        + sqrt_v * dW_S
    )

    # Update variance using that SAME current variance.
    # Retain raw v in the first term: this is full-truncation Euler.
    v_next = (
        v
        + kappa * (theta - v_positive) * dt
        + xi * sqrt_v * dW_v
    )

    return x_next, v_next

if __name__ == "__main__":
    