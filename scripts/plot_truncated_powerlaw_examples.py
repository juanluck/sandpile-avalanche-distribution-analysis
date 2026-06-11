"""
Example script to fit a full-range truncated power law to individual
avalanche-duration frequency datasets and plot the empirical PDF
against the theoretical fits (pure power law vs. truncated power law).
"""

from pathlib import Path

import numpy as np
import powerlaw
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# Directory containing the experimental data
DATA_DIR = Path("./data/avex_data/exp1")

# List of filenames to analyze and plot
FILES_TO_PLOT = [
    "d0_r0.dat",
    "d45_r10.dat",
]


# ============================================================
# FUNCTIONS
# ============================================================

def load_frequency_file(path: Path):
    """
    Load a frequency data file containing two columns: value and frequency.
    Repeats values according to their frequencies to reconstruct the raw data sample.

    Parameters
    ----------
    path : Path
        Path to the space-separated text file (.dat) containing the frequency distribution.

    Returns
    -------
    data : np.ndarray
        Reconstructed 1D array of raw values repeated by count.
    values : np.ndarray
        Array of unique values where counts > 0.
    counts : np.ndarray
        Array of frequencies corresponding to each unique value.
    """
    # Load 2D numerical table from text file
    table = np.loadtxt(path, dtype=int)
    table = np.atleast_2d(table)

    values = table[:, 0]
    counts = table[:, 1]

    # Keep only strictly positive values and non-zero counts
    mask = (values > 0) & (counts > 0)
    values = values[mask]
    counts = counts[mask]

    # Reconstruct raw data array by repeating values based on their counts
    data = np.repeat(values, counts)

    return data, values, counts


def plot_full_truncated_powerlaw(file_name: str):
    """
    Fits pure and truncated power laws on the full range (xmin=min(data), xmax=max(data))
    of a single file's reconstructed sample, prints stats, and plots the PDF comparison.

    Parameters
    ----------
    file_name : str
        Filename of the data file located in DATA_DIR.
    """
    path = DATA_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    # Load and reconstruct the data
    data, values, counts = load_frequency_file(path)

    # Full range boundary setup
    xmin = int(np.min(data))
    xmax = int(np.max(data))

    # Fit discrete power law models using powerlaw package
    fit_full = powerlaw.Fit(
        data,
        xmin=xmin,
        xmax=xmax,
        discrete=True,
        verbose=False,
    )

    # Extract fitted distribution instances
    truncated_power_law = fit_full.truncated_power_law
    pure_power_law = fit_full.power_law

    # Retrieve parameters
    alpha_truncated = truncated_power_law.alpha
    lambda_truncated = truncated_power_law.Lambda
    alpha_power_law = pure_power_law.alpha

    # Perform nested likelihood ratio test: power_law vs truncated_power_law
    likelihood_ratio, p_value = fit_full.distribution_compare(
        "power_law",
        "truncated_power_law",
    )

    # Print summary statistics
    print()
    print(f"File: {file_name}")
    print(f"n = {len(data)}")
    print(f"Full range: xmin={xmin}, xmax={xmax}")
    print(f"power_law alpha = {alpha_power_law}")
    print(f"truncated_power_law alpha = {alpha_truncated}")
    print(f"truncated_power_law lambda = {lambda_truncated}")
    print(f"power_law vs truncated_power_law: R = {likelihood_ratio}, p = {p_value}")

    # Normalize counts to form empirical probability density function (PDF)
    empirical_pdf = counts / counts.sum()

    # Generate the log-log visualization plot
    plt.figure(figsize=(7, 5))

    # Plot empirical data points
    plt.scatter(values, empirical_pdf, marker="o", label="Empirical data")

    # Overlay fitted truncated power-law PDF
    truncated_power_law.plot_pdf(
        linestyle="--",
        linewidth=2,
        label=fr"Truncated power law: $\alpha={alpha_truncated:.3f}$, $\lambda={lambda_truncated:.3g}$",
    )

    # Overlay fitted pure power-law PDF
    pure_power_law.plot_pdf(
        linestyle=":",
        linewidth=2,
        label=fr"Pure power law: $\alpha={alpha_power_law:.3f}$",
    )

    # Formatting axes and legends
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Avalanche duration")
    plt.ylabel("Probability")
    plt.title(f"Full-range truncated power-law fit: {file_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    Main driver function: iterate through the configured files and plot their fits.
    """
    for file_name in FILES_TO_PLOT:
        plot_full_truncated_powerlaw(file_name)


if __name__ == "__main__":
    main()
