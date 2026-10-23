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
    print(v0_centre)
    sigma_ref = np.sqrt(v0_centre)
    print(sigma_ref)


    T = maturities[:, None]                             #shape (6,1)
    print(T)
    z = z_grid[None, :]                                 #shape (1,7)
    print(z)

    K_centre = S_0 * np.exp(drift * T)                  # (6, 1)
    print(K_centre)
    exponent = np.sqrt(T) * z * sigma_ref               # (6, 7) 
    print(exponent)
    K = K_centre * np.exp(exponent)
    print(K)


if __name__ == "__main__":
    make_prior_spec()
    make_strike_grid(maturities= MATURITIES, z_grid= Z_GRID, )
