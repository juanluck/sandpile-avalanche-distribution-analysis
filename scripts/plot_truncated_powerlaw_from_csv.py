"""
Plot truncated power-law exponent (alpha) and exponential cutoff (lambda)
evolution parameters from pre-calculated summary or parameter CSV files.
Generates line plots with uncertainty bands for configured rewiring percentages
across varying network degradation levels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# Directory containing the input CSV files
INPUT_DIR = Path("./truncated_powerlaw_full_range_results")

# Paths to the input CSV data files
SUMMARY_CSV = INPUT_DIR / "truncated_powerlaw_full_range_summary.csv"
PARAMETERS_CSV = INPUT_DIR / "truncated_powerlaw_full_range_parameters.csv"

# Output paths for the generated figures (PNG and PDF)
OUTPUT_FIGURE = INPUT_DIR / "truncated_powerlaw_full_range_alpha_lambda_from_csv.png"
OUTPUT_FIGURE_PDF = INPUT_DIR / "truncated_powerlaw_full_range_alpha_lambda_from_csv.pdf"

# Rewiring rates (percentages) to filter and display in the plot
R_VALUES = [0, 10]

# Network degradation range (percentage of nodes removed)
D_MIN = 0
D_MAX = 75

# Color mapping representing each rewiring rate
COLOR_BY_REWIRING = {0: "red", 10: "blue"}


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Load data from the summary CSV if it exists. Otherwise, fall back
    to load and aggregate data from the raw parameter CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame containing aggregated mean and std statistics for alpha and lambda.

    Raises
    ------
    FileNotFoundError
        If neither the summary CSV nor the parameter CSV is found.
    """
    if SUMMARY_CSV.exists():
        print(f"Reading summary CSV: {SUMMARY_CSV}")
        return normalize_summary_columns(pd.read_csv(SUMMARY_CSV))

    if PARAMETERS_CSV.exists():
        print(f"Summary CSV not found. Reading parameter CSV: {PARAMETERS_CSV}")
        dataframe = normalize_parameter_columns(pd.read_csv(PARAMETERS_CSV))
        return build_summary_from_parameters(dataframe)

    raise FileNotFoundError(
        f"Could not find either:\n"
        f"  {SUMMARY_CSV}\n"
        f"or:\n"
        f"  {PARAMETERS_CSV}"
    )


def normalize_summary_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns of the summary DataFrame to standard names ('degradation', 'rewiring')
    if they are stored under shorthand names ('d', 'r').

    Parameters
    ----------
    dataframe : pd.DataFrame
        Loaded summary DataFrame.

    Returns
    -------
    pd.DataFrame
        Normalized summary DataFrame.

    Raises
    ------
    ValueError
        If any of the required columns are missing after normalization.
    """
    dataframe = dataframe.copy()

    # Normalize degradation column name
    if "d" in dataframe.columns and "degradation" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"d": "degradation"})

    # Normalize rewiring column name
    if "r" in dataframe.columns and "rewiring" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"r": "rewiring"})

    required_columns = {
        "degradation",
        "rewiring",
        "alpha_mean",
        "alpha_std",
        "lambda_mean",
        "lambda_std",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"The summary CSV is missing required columns: {missing}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return dataframe


def normalize_parameter_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns of the raw parameters DataFrame to standard names.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Loaded parameters DataFrame.

    Returns
    -------
    pd.DataFrame
        Normalized parameters DataFrame.

    Raises
    ------
    ValueError
        If any of the required columns are missing after normalization.
    """
    dataframe = dataframe.copy()

    # Normalize degradation column name
    if "d" in dataframe.columns and "degradation" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"d": "degradation"})

    # Normalize rewiring column name
    if "r" in dataframe.columns and "rewiring" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"r": "rewiring"})

    # Normalize lambda parameter column name
    if "Lambda_full" in dataframe.columns and "lambda_full" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"Lambda_full": "lambda_full"})

    required_columns = {
        "degradation",
        "rewiring",
        "alpha_truncated_full",
        "lambda_full",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"The parameter CSV is missing required columns: {missing}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return dataframe


def build_summary_from_parameters(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw fit parameter results (across multiple runs/experiments)
    to calculate the mean and standard deviation of alpha and lambda.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Normalized parameter DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated summary DataFrame.
    """
    return (
        dataframe.groupby(["degradation", "rewiring"])
        .agg(
            alpha_mean=("alpha_truncated_full", "mean"),
            alpha_std=("alpha_truncated_full", "std"),
            lambda_mean=("lambda_full", "mean"),
            lambda_std=("lambda_full", "std"),
        )
        .reset_index()
        .sort_values(["rewiring", "degradation"])
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_mean_with_band(
    axis,
    summary: pd.DataFrame,
    rewiring_value: int,
    y_mean: str,
    y_std: str,
    label: str,
    color: str,
):
    """
    Helper function to plot a line representing the mean and a shaded band
    representing the standard deviation.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Matplotlib axis object to plot onto.
    summary : pd.DataFrame
        Aggregated summary DataFrame.
    rewiring_value : int
        Rewiring percentage to filter.
    y_mean : str
        Name of the column containing the mean values.
    y_std : str
        Name of the column containing the standard deviation values.
    label : str
        Label for the legend.
    color : str
        Color of the line and band.
    """
    # Filter by rewiring value and sort by degradation level
    subset = summary[summary["rewiring"] == rewiring_value].sort_values("degradation")

    # Restrict to configured degradation limits
    subset = subset[
        (subset["degradation"] >= D_MIN)
        & (subset["degradation"] <= D_MAX)
    ]

    x = subset["degradation"].to_numpy()
    mean = subset[y_mean].to_numpy()
    std = subset[y_std].fillna(0).to_numpy()

    # Plot the mean line
    axis.plot(x, mean, linewidth=2.5, color=color, label=label)
    # Fill the standard deviation uncertainty band
    axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def make_figure(summary: pd.DataFrame):
    """
    Create a 1x2 grid figure displaying the evolution of the alpha exponent
    and lambda cutoff parameters under different degradation and rewiring conditions,
    and save the output files.

    Parameters
    ----------
    summary : pd.DataFrame
        Normalized and aggregated summary DataFrame.
    """
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True)
    x_ticks = np.arange(D_MIN, D_MAX + 1, 10)

    # Left Panel: Alpha Exponent
    axis = axes[0]
    for rewiring_value in R_VALUES:
        plot_mean_with_band(
            axis=axis,
            summary=summary,
            rewiring_value=rewiring_value,
            y_mean="alpha_mean",
            y_std="alpha_std",
            label=f"RR={rewiring_value}%",
            color=COLOR_BY_REWIRING.get(rewiring_value, "black"),
        )

    axis.set_title(r"Full-range truncated power-law exponent $\alpha$")
    axis.set_xlabel("Nodes removed (%)")
    axis.set_ylabel(r"$\alpha$")
    axis.legend()
    axis.set_xlim(D_MIN, D_MAX)
    axis.set_xticks(x_ticks)
    axis.grid(True, which="major", alpha=0.3)
    axis.minorticks_off()

    # Right Panel: Lambda Parameter
    axis = axes[1]
    for rewiring_value in R_VALUES:
        plot_mean_with_band(
            axis=axis,
            summary=summary,
            rewiring_value=rewiring_value,
            y_mean="lambda_mean",
            y_std="lambda_std",
            label=f"RR={rewiring_value}%",
            color=COLOR_BY_REWIRING.get(rewiring_value, "black"),
        )

    axis.set_title(r"Exponential cutoff parameter $\lambda$")
    axis.set_xlabel("Nodes removed (%)")
    axis.set_ylabel(r"$\lambda$")
    axis.legend()
    axis.set_xlim(D_MIN, D_MAX)
    axis.set_xticks(x_ticks)
    axis.grid(True, which="major", alpha=0.3)
    axis.minorticks_off()

    # Add overall title and layout styling
    figure.suptitle(
        "Full-range truncated power-law fits of avalanche-duration distributions",
        fontsize=13,
    )

    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")

    plt.show()

    print(f"Figure saved to: {OUTPUT_FIGURE}")
    print(f"PDF saved to:    {OUTPUT_FIGURE_PDF}")


def main():
    """
    Main execution pipeline: loads the dataset, filters it, and triggers plotting.
    """
    summary = load_data()

    # Filter data to keep only configured rewiring values and degradation levels
    summary = summary[summary["rewiring"].isin(R_VALUES)]
    summary = summary[
        (summary["degradation"] >= D_MIN)
        & (summary["degradation"] <= D_MAX)
    ]

    make_figure(summary)


if __name__ == "__main__":
    main()
