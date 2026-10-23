"""Reusable Matplotlib plots; each function returns Figure and Axes objects.

No function calls show() or changes global plotting settings. The caller can
display, customise, save and close figures, following the lecture OO interface.
"""

import numpy as np
import matplotlib.pyplot as plt
from .data_generation import HestonSample

COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def plot_paths(sample: HestonSample) -> tuple:
    """Plot underlying prices and effective variances against time.

    Parameters
    ----------
    sample : HestonSample
        Dataset with rows as samples and columns as observation times.

    Returns
    -------
    fig, axes : Figure, np.ndarray
        Two vertically stacked axes sharing time in years.
    """

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, layout="constrained"
    )

    for i in range(len(sample.parameters)):
        colour = COLOURS[i % len(COLOURS)]
        label = f"Sample {i + 1}"
        axes[0].plot(
            sample.times, sample.prices[i], color=colour, label=label, lw=1.3
        )
        axes[1].plot(
            sample.times,
            sample.variances[i],
            color=colour,
            label=label,
            lw=1.3,
        )
        # A faint reference line shows each path's long-run variance parameter.
        axes[1].axhline(
            sample.parameters[i, 2], color=colour, ls="--", alpha=0.5
        )

    axes[0].set(
        title=(
            "Two synthetic Heston histories"
            if len(sample.parameters) == 2
            else "Synthetic Heston histories"
        ),
        ylabel="Underlying price S(t)",
    )
    axes[1].set(
        xlabel="Time (years)",
        ylabel="Variance v(t)",
        title="Effective variance; dashed lines show each theta",
    )

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()

    return fig, axes


def plot_return_histograms(sample: HestonSample, bins: int = 30) -> tuple:
    """Plot per-step log-return density histograms, one panel per sample.

    sample supplies histories; bins is the positive histogram bin count.
    Returns (fig, axes), where axes is a 1-D array. Shared bins support visual
    comparison. Time increments are not independent parameter draws, and a
    histogram of one path is not an estimated terminal-price distribution.
    """

    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer.")

    returns = sample.log_returns()
    edges = np.histogram_bin_edges(returns.ravel(), bins=bins)
    fig, panels = plt.subplots(
        1,
        len(returns),
        figsize=(5 * len(returns), 3.8),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes = panels[0]

    for i, ax in enumerate(axes):
        ax.hist(
            returns[i],
            bins=edges,
            density=True,
            color=COLOURS[i % len(COLOURS)],
            alpha=0.8,
            edgecolor="white",
        )
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set(title=f"Sample {i + 1}: one-step returns", xlabel="Log return")

    axes[0].set_ylabel("Density")

    return fig, axes


def plot_price_variance(sample: HestonSample) -> tuple:
    """Scatter underlying price against variance, colouring points by time.

    sample supplies the arrays. Returns (fig, axes) with one panel per sample.
    The trajectory's level relationship is NOT the Brownian shock correlation
    rho. A shared colour scale indicates time in years.
    """

    fig, panels = plt.subplots(
        1,
        len(sample.parameters),
        figsize=(5 * len(sample.parameters), 4),
        squeeze=False,
        layout="constrained",
    )
    axes = panels[0]

    for i, ax in enumerate(axes):
        points = ax.scatter(
            sample.prices[i],
            sample.variances[i],
            c=sample.times,
            cmap="viridis",
            s=9,
            alpha=0.75,
            vmin=sample.times[0],
            vmax=sample.times[-1],
        )
        ax.set(
            xlabel="Underlying price",
            ylabel="Variance",
            title=f"Sample {i + 1}: state trajectory",
        )

    fig.colorbar(points, ax=list(axes), label="Time (years)")

    return fig, axes


def plot_drawdowns(sample: HestonSample) -> tuple:
    """Plot percentage drawdown from each sample's running price maximum.

    sample supplies the prices and time grid. Returns (fig, ax). Drawdown is
    zero at a new peak and positive below that peak; no trading strategy is
    modelled by this descriptive statistic.
    """

    peak = np.maximum.accumulate(sample.prices, axis=1)
    drawdowns = 100 * (1 - sample.prices / peak)
    fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")

    for i, values in enumerate(drawdowns):
        ax.plot(
            sample.times,
            values,
            label=f"Sample {i + 1}",
            color=COLOURS[i % len(COLOURS)],
        )

    ax.set(
        xlabel="Time (years)",
        ylabel="Drawdown (%)",
        title="Distance below the running price peak",
    )
    ax.grid(alpha=0.2)
    ax.legend()

    return fig, ax
