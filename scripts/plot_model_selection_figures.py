"""
Generate visualization figures for distribution model selection results.
Reads CSV data containing the best-fitting distributions across experiments,
aggregates distribution winning frequencies (optionally in degradation level bins),
and plots:
  - Combined stacked bar charts grouped by rewiring rate.
  - Individual stacked bar charts for each rewiring rate.
  - Line plots representing winning frequencies for rewiring rates or distributions.
  - A modal winner scatter plot showing the dominant distribution at each d and r.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# Primary input file path
INPUT_CSV = "./distribution_results/all_experiments/best_distributions_all.csv"
# Fallback input file path if primary is missing
FALLBACK_INPUT_CSV = "./distribution_results/all_experiments/best_distributions_by_file.csv"

# Directory where output plots will be saved
OUTPUT_DIR = Path("./distribution_results/all_experiments/model_selection_figures")

# Rewiring values (percentages) to analyze
R_VALUES = [0, 10]
# Degradation values (percentage of nodes removed) to analyze
D_VALUES = list(range(0, 76))

# Set of candidate probability distributions
DISTRIBUTIONS = [
    "power_law",
    "lognormal",
    "exponential",
    "truncated_power_law",
    "stretched_exponential",
    "lognormal_positive",
]

# Mapping distribution names to user-friendly labels for figure legends
DISTRIBUTION_LABELS = {
    "power_law": "Power law",
    "lognormal": "Lognormal",
    "exponential": "Exponential",
    "truncated_power_law": "Truncated power law",
    "stretched_exponential": "Stretched exponential",
    "lognormal_positive": "Positive lognormal",
}

# Explicit color scheme for distributions to ensure consistency across plots
DISTRIBUTION_COLORS = {
    "lognormal": "#1f77b4",
    "truncated_power_law": "#ff7f0e",
    "stretched_exponential": "#2ca02c",
    "power_law": "#d62728",
    "exponential": "#9467bd",
    "lognormal_positive": "#8c564b",
}

# Color mapping representing each rewiring rate
REWIRING_COLORS = {
    0: "red",
    10: "blue",
}


# ============================================================
# DATA PREPARATION
# ============================================================

def get_input_csv() -> str:
    """
    Locate the input CSV file. Prefers the primary path, but falls back
    to the alternative file if needed.

    Returns
    -------
    str
        Path to the available CSV file.

    Raises
    ------
    FileNotFoundError
        If neither file is found.
    """
    if Path(INPUT_CSV).exists():
        return INPUT_CSV

    if Path(FALLBACK_INPUT_CSV).exists():
        return FALLBACK_INPUT_CSV

    raise FileNotFoundError(
        f"Could not find input CSV:\n"
        f"  {INPUT_CSV}\n"
        f"or fallback:\n"
        f"  {FALLBACK_INPUT_CSV}"
    )


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the shorthand column names to 'd' and 'r' in the DataFrame.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns.

    Raises
    ------
    ValueError
        If required columns are missing after normalization.
    """
    dataframe = dataframe.copy()

    # Map 'degradation' column to 'd'
    if "degradation" in dataframe.columns and "d" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"degradation": "d"})

    # Map 'rewiring' column to 'r'
    if "rewiring" in dataframe.columns and "r" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"rewiring": "r"})

    required_columns = {"d", "r", "best_distribution"}
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Available columns are: {list(dataframe.columns)}"
        )

    return dataframe


def build_summary(input_csv: str) -> pd.DataFrame:
    """
    Read the model selection output, filter data by configured r and d limits,
    and calculate winning frequencies (percentages) for each distribution
    grouped by degradation level and rewiring rate.

    Parameters
    ----------
    input_csv : str
        Path to the model selection CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame listing winning distribution percentages per (d, r).
    """
    dataframe = pd.read_csv(input_csv)
    dataframe = normalize_columns(dataframe)

    # Apply configuration filters
    dataframe = dataframe[dataframe["r"].isin(R_VALUES)]
    dataframe = dataframe[dataframe["d"].isin(D_VALUES)]

    # Calculate total runs per degradation and rewiring level
    totals = dataframe.groupby(["d", "r"]).size().reset_index(name="total")

    # Count wins per distribution, degradation, and rewiring level
    counts = (
        dataframe.groupby(["d", "r", "best_distribution"])
        .size()
        .reset_index(name="count")
    )

    # Reindex to ensure all permutations are represented (filling missing counts with 0)
    full_index = pd.MultiIndex.from_product(
        [D_VALUES, R_VALUES, DISTRIBUTIONS],
        names=["d", "r", "best_distribution"],
    )

    counts = (
        counts.set_index(["d", "r", "best_distribution"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Compute percentage frequency for each distribution
    summary = counts.merge(totals, on=["d", "r"], how="left")
    summary = summary.dropna(subset=["total"])

    summary["percentage"] = 100.0 * summary["count"] / summary["total"]
    return summary.sort_values(["r", "d", "best_distribution"]).reset_index(drop=True)


def bin_summary(summary: pd.DataFrame, bin_width: int) -> pd.DataFrame:
    """
    Group degradation values ('d') into larger bins to smooth plotting results.

    Parameters
    ----------
    summary : pd.DataFrame
        The un-binned summary DataFrame.
    bin_width : int
        Width of the degradation bins. If <= 1, no binning is done.

    Returns
    -------
    pd.DataFrame
        Binned DataFrame with averaged percentages.
    """
    if bin_width <= 1:
        return summary.copy()

    data = summary.copy()
    # Compute the bin start value for each degradation level
    data["d_bin"] = (data["d"] // bin_width) * bin_width

    return (
        data.groupby(["d_bin", "r", "best_distribution"])
        .agg(percentage=("percentage", "mean"))
        .reset_index()
        .rename(columns={"d_bin": "d"})
    )


def make_pivot_for_rewiring(
    summary: pd.DataFrame,
    rewiring_value: int,
    bin_width: int = 1,
) -> pd.DataFrame:
    """
    Reshape the summary DataFrame into a pivot table (degradation 'd' as rows,
    distributions as columns, percentage as values) for a specific rewiring rate.

    Parameters
    ----------
    summary : pd.DataFrame
        Summary DataFrame.
    rewiring_value : int
        Rewiring rate to pivot.
    bin_width : int
        Bin size to group degradation values.

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame containing only distributions that won at least once.
    """
    data = bin_summary(summary, bin_width=bin_width)
    subset = data[data["r"] == rewiring_value].copy()

    # Pivot to form matrix of degradation vs distribution percentages
    pivot = (
        subset.pivot_table(
            index="d",
            columns="best_distribution",
            values="percentage",
            fill_value=0.0,
        )
        .sort_index()
    )

    # Ensure all configured distributions are columns
    pivot = pivot.reindex(columns=DISTRIBUTIONS, fill_value=0.0)
    # Exclude distributions that have 0% win rate across all degradation levels
    return pivot.loc[:, (pivot > 0).any(axis=0)]


# ============================================================
# PLOT UTILITIES
# ============================================================

def save_figure(figure, output_file: Path):
    """
    Helper function to create output folders and save a figure as PNG and PDF.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure instance to save.
    output_file : Path
        Target filename path (expected to end in .png).
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    figure.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")


def draw_stacked_bars(axis, pivot: pd.DataFrame, bar_width: float = 1.0):
    """
    Draw a stacked bar chart onto the specified axes.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Plot axis.
    pivot : pd.DataFrame
        Pivoted DataFrame containing percentage distributions.
    bar_width : float
        Width of the bars.
    """
    x_positions = np.arange(len(pivot.index))
    bottom = np.zeros(len(pivot.index))

    # Iterate through distributions and stack their frequencies
    for distribution in pivot.columns:
        values = pivot[distribution].to_numpy()
        color = DISTRIBUTION_COLORS.get(distribution, None)
        label = DISTRIBUTION_LABELS.get(distribution, distribution)

        # Plot bars for the current distribution
        bars = axis.bar(
            x_positions,
            values,
            bottom=bottom,
            width=bar_width,
            align="center",
            label=label,
            color=color,
            edgecolor=color,
            linewidth=0,
            antialiased=False,
        )

        # Disable antialiasing on bars to avoid sub-pixel rendering gaps
        for bar in bars:
            bar.set_antialiased(False)

        bottom += values


def set_degradation_ticks(axis, degradation_values):
    """
    Format x-axis tick positions and labels for degradation values.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Plot axis.
    degradation_values : list
        List of degradation levels mapped to the bar index.
    """
    tick_positions = [
        index for index, value in enumerate(degradation_values) if value % 10 == 0
    ]
    tick_labels = [str(value) for value in degradation_values if value % 10 == 0]
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels, rotation=0)


def unique_legend(handles, labels):
    """
    Filter legend handles and labels to remove duplicate entries.

    Parameters
    ----------
    handles : list
        Legend handles.
    labels : list
        Legend labels.

    Returns
    -------
    unique_handles : list
        Unique legend handles.
    unique_labels : list
        Unique legend labels.
    """
    seen = set()
    unique_handles = []
    unique_labels = []

    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)

    return unique_handles, unique_labels


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_stacked_bars_combined_by_rewiring(
    summary: pd.DataFrame,
    output_dir: Path,
    bin_width: int = 5,
):
    """
    Plot combined stacked bar charts on a single grid with shared x and y axes,
    separated vertically by rewiring rate.

    Parameters
    ----------
    summary : pd.DataFrame
        Winning distribution summary DataFrame.
    output_dir : Path
        Base output directory.
    bin_width : int
        Degradation bin width.
    """
    figure_dir = output_dir / "stacked_bars_combined_by_rewiring"
    figure_dir.mkdir(parents=True, exist_ok=True)

    available_r_values = [r for r in R_VALUES if r in summary["r"].unique()]

    figure, axes = plt.subplots(
        len(available_r_values),
        1,
        figsize=(11, 6.5),
        sharex=True,
        sharey=True,
    )

    if len(available_r_values) == 1:
        axes = [axes]

    last_pivot = None

    # Draw stacked bar charts for each rewiring value
    for axis, rewiring_value in zip(axes, available_r_values):
        pivot = make_pivot_for_rewiring(summary, rewiring_value, bin_width)

        if pivot.empty:
            continue

        last_pivot = pivot
        draw_stacked_bars(axis, pivot, bar_width=1.0)

        axis.set_title(f"RR={rewiring_value}%", loc="left", fontweight="bold")
        axis.set_ylabel("Winning frequency (%)")
        axis.set_ylim(0, 100)
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_axisbelow(True)

    if last_pivot is not None:
        set_degradation_ticks(axes[-1], list(last_pivot.index))
        axes[-1].set_xlabel("Nodes removed (%)")

    # Add shared legend at the top of the combined figure
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = unique_legend(handles, labels)

    figure.legend(
        handles,
        labels,
        title="Winning distribution",
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )

    figure.suptitle("Composition of best-fitting distributions", y=1.08, fontsize=13)
    figure.tight_layout()

    output_file = figure_dir / f"winning_distribution_composition_combined_bin{bin_width}.png"
    save_figure(figure, output_file)
    plt.close(figure)


def plot_stacked_bars_by_rewiring(
    summary: pd.DataFrame,
    output_dir: Path,
    bin_width: int = 1,
):
    """
    Plot and save individual stacked bar charts for each rewiring rate.

    Parameters
    ----------
    summary : pd.DataFrame
        Winning distribution summary DataFrame.
    output_dir : Path
        Base output directory.
    bin_width : int
        Degradation bin width.
    """
    figure_dir = output_dir / "stacked_bars_by_rewiring"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for rewiring_value in R_VALUES:
        pivot = make_pivot_for_rewiring(summary, rewiring_value, bin_width)

        if pivot.empty:
            continue

        figure, axis = plt.subplots(figsize=(12, 5.5))
        draw_stacked_bars(axis, pivot, bar_width=1.0)

        axis.set_title(f"Composition of best-fitting distributions, RR={rewiring_value}%")
        axis.set_xlabel("Nodes removed (%)")
        axis.set_ylabel("Winning frequency (%)")
        axis.set_ylim(0, 100)
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_axisbelow(True)

        set_degradation_ticks(axis, list(pivot.index))

        handles, labels = axis.get_legend_handles_labels()
        handles, labels = unique_legend(handles, labels)

        axis.legend(
            handles,
            labels,
            title="Winning distribution",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        figure.tight_layout()

        output_file = figure_dir / f"winning_distribution_composition_RR{rewiring_value}_bin{bin_width}.png"
        save_figure(figure, output_file)
        plt.close(figure)


def plot_lines_by_rewiring(summary: pd.DataFrame, output_dir: Path):
    """
    Plot winning frequencies of different distributions as line plots over degradation levels
    for each individual rewiring rate.

    Parameters
    ----------
    summary : pd.DataFrame
        Winning distribution summary DataFrame.
    output_dir : Path
        Base output directory.
    """
    figure_dir = output_dir / "lines_by_rewiring"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for rewiring_value in R_VALUES:
        subset = summary[summary["r"] == rewiring_value].copy()
        figure, axis = plt.subplots(figsize=(9, 5.5))

        for distribution in DISTRIBUTIONS:
            line = subset[subset["best_distribution"] == distribution].sort_values("d")

            if line.empty or line["percentage"].sum() == 0:
                continue

            axis.plot(
                line["d"],
                line["percentage"],
                linewidth=2,
                label=DISTRIBUTION_LABELS.get(distribution, distribution),
                color=DISTRIBUTION_COLORS.get(distribution, None),
            )

        axis.set_xlabel("Nodes removed (%)")
        axis.set_ylabel("Winning frequency (%)")
        axis.set_title(f"Best-fitting distribution frequency, RR={rewiring_value}%")
        axis.set_xticks(np.arange(0, max(D_VALUES) + 1, 10))
        axis.set_xlim(min(D_VALUES), max(D_VALUES))
        axis.set_ylim(0, 100)
        axis.grid(True, alpha=0.3)
        axis.legend(
            title="Winning distribution",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        figure.tight_layout()

        output_file = figure_dir / f"best_distribution_frequency_RR{rewiring_value}.png"
        save_figure(figure, output_file)
        plt.close(figure)


def plot_lines_by_distribution(summary: pd.DataFrame, output_dir: Path):
    """
    Plot line plots showing the winning frequency evolution of each candidate distribution,
    comparing different rewiring rates.

    Parameters
    ----------
    summary : pd.DataFrame
        Winning distribution summary DataFrame.
    output_dir : Path
        Base output directory.
    """
    figure_dir = output_dir / "lines_by_distribution"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for distribution in DISTRIBUTIONS:
        subset = summary[summary["best_distribution"] == distribution].copy()

        if subset.empty or subset["percentage"].sum() == 0:
            continue

        figure, axis = plt.subplots(figsize=(8, 5))

        for rewiring_value in R_VALUES:
            line = subset[subset["r"] == rewiring_value].sort_values("d")

            if line.empty:
                continue

            axis.plot(
                line["d"],
                line["percentage"],
                linewidth=2,
                color=REWIRING_COLORS.get(rewiring_value, None),
                label=f"RR={rewiring_value}%",
            )

        axis.set_xlabel("Nodes removed (%)")
        axis.set_ylabel("Winning frequency (%)")
        axis.set_title(f"Winning frequency: {DISTRIBUTION_LABELS.get(distribution, distribution)}")
        axis.set_xticks(np.arange(0, max(D_VALUES) + 1, 10))
        axis.set_xlim(min(D_VALUES), max(D_VALUES))
        axis.set_ylim(0, 100)
        axis.grid(True, alpha=0.3)
        axis.legend()

        figure.tight_layout()

        output_file = figure_dir / f"winning_frequency_{distribution}.png"
        save_figure(figure, output_file)
        plt.close(figure)


def plot_modal_winner_map(summary: pd.DataFrame, output_dir: Path):
    """
    Create a scatter plot matrix showing the modal winning distribution (most frequent winner)
    for each degradation and rewiring parameter pair. Point sizes represent the win percentage.

    Parameters
    ----------
    summary : pd.DataFrame
        Winning distribution summary DataFrame.
    output_dir : Path
        Base output directory.
    """
    figure_dir = output_dir / "modal_winner_map"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve the row with the maximum win percentage for each (d, r) configuration
    idx = summary.groupby(["d", "r"])["percentage"].idxmax()
    modal = summary.loc[idx].copy().sort_values(["r", "d"])

    figure, axis = plt.subplots(figsize=(11, 4.8))

    # Scatter plot for each distribution group
    for distribution in DISTRIBUTIONS:
        subset = modal[modal["best_distribution"] == distribution]

        if subset.empty:
            continue

        # Point size scales with the win percentage of the modal distribution
        sizes = 100 + 500 * subset["percentage"] / 100.0

        axis.scatter(
            subset["d"],
            subset["r"],
            s=sizes,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.8,
            label=DISTRIBUTION_LABELS.get(distribution, distribution),
            color=DISTRIBUTION_COLORS.get(distribution, None),
        )

    # Label points with their winning percentage
    for _, row in modal.iterrows():
        axis.text(row["d"], row["r"], f"{row['percentage']:.0f}%", ha="center", va="center", fontsize=8)

    axis.set_xlabel("Nodes removed (%)")
    axis.set_ylabel("Rewiring")
    axis.set_title("Modal best-fitting distribution by degradation and rewiring")
    axis.set_xticks(np.arange(0, max(D_VALUES) + 1, 10))
    axis.set_yticks(R_VALUES)
    axis.set_yticklabels([f"RR={r}%" for r in R_VALUES])
    axis.set_xlim(min(D_VALUES) - 1, max(D_VALUES) + 1)
    axis.grid(True, axis="x", alpha=0.25)
    axis.legend(title="Modal winner", bbox_to_anchor=(1.02, 1), loc="upper left")

    figure.tight_layout()

    output_file = figure_dir / "modal_best_fitting_distribution_map.png"
    save_figure(figure, output_file)
    plt.close(figure)


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main driver execution. Sets up folders, reads data, aggregates, and draws all figures.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_csv = get_input_csv()
    summary = build_summary(input_csv)

    # Save summary table to CSV for reference
    summary_file = OUTPUT_DIR / "distribution_winning_percentages.csv"
    summary.to_csv(summary_file, index=False)

    # Combined stacked bar charts (binned at 5% and 1%)
    plot_stacked_bars_combined_by_rewiring(summary, OUTPUT_DIR, bin_width=5)
    plot_stacked_bars_combined_by_rewiring(summary, OUTPUT_DIR, bin_width=1)

    # Separate stacked bar charts by rewiring (binned at 5% and 1%)
    plot_stacked_bars_by_rewiring(summary, OUTPUT_DIR, bin_width=5)
    plot_stacked_bars_by_rewiring(summary, OUTPUT_DIR, bin_width=1)

    # Frequency evolution line plots
    plot_lines_by_rewiring(summary, OUTPUT_DIR)
    plot_lines_by_distribution(summary, OUTPUT_DIR)
    
    # Modal best-fit distribution grid map
    plot_modal_winner_map(summary, OUTPUT_DIR)

    print("Analysis completed.")
    print(f"Input CSV used:         {input_csv}")
    print(f"Summary table saved to: {summary_file}")
    print(f"Figures saved to:       {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
