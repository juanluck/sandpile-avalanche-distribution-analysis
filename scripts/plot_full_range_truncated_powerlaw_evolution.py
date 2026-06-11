"""
Parallel fitting of the full-range truncated power law (TPL) model to avalanche
duration distributions across multiple experiments, and generation of figures representing the
evolution of the power law exponent (alpha) and exponential cutoff (lambda) as functions
of network degradation and rewiring.
Also evaluates the nested likelihood-ratio test (LRT) comparing the truncated power law 
against the pure power law.
"""

from __future__ import annotations

import os

# Avoid CPU oversubscription if NumPy/SciPy use multithreaded BLAS internally.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw


# ============================================================
# CONFIGURATION
# ============================================================

# Root folder containing experimental folders (exp1, exp2, etc.)
DATA_ROOT: Optional[Path] = None

# Range of experiments to analyze
EXPERIMENTS = range(1, 26)

# Rewiring rates (percentages) to look for
R_VALUES = [0, 10]

# Range of degradation rates (percentage of nodes removed) to consider
D_MIN = 0
D_MAX = 75

# Number of parallel subprocesses for process pool
N_WORKERS = 8

# Significance p-value threshold for likelihood ratio comparisons
P_THRESHOLD = 0.05

# Full-range fit configuration:
#
# USE_XMAX = False (recommended): unbounded fit, p(t) ∝ t^(-alpha) e^(-lambda t)
#   on [xmin, +inf). This matches the equation as written in the paper.
# USE_XMAX = True (legacy behaviour): bounded fit on [xmin, xmax=max(data)].
#   NOTE: setting xmax changes the *normalization* of every candidate
#   (bounded support), so alpha and lambda are parameters of a truncated
#   distribution in a second sense, and the fit becomes sensitive to the
#   single largest observed avalanche. Run both and compare: if the
#   alpha(d) and lambda(d) curves are qualitatively stable, report the
#   unbounded fit and mention the robustness check in the paper.
USE_XMAX = False

# Plot the naive pure power-law exponent (dashed) on the alpha panel,
# to visualize how much it diverges from the TPL descriptor.
PLOT_PURE_PL_ALPHA = True

# Uncertainty band plotting mode:
# "percentile" (16th-84th percentiles, recommended due to skewness)
# "std" (mean +/- one standard deviation, legacy behaviour)
BAND_MODE = "percentile"

# Output directory and output file targets
OUTPUT_DIR = Path("./truncated_powerlaw_full_range_results")
OUTPUT_PARAMETERS_CSV = OUTPUT_DIR / "truncated_powerlaw_full_range_parameters.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "truncated_powerlaw_full_range_summary.csv"
OUTPUT_FIGURE = OUTPUT_DIR / "truncated_powerlaw_full_range_alpha_lambda.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "truncated_powerlaw_full_range_alpha_lambda.pdf"
OUTPUT_FIGURE_LRT = OUTPUT_DIR / "truncated_powerlaw_full_range_lrt_fraction.png"
OUTPUT_FIGURE_LRT_PDF = OUTPUT_DIR / "truncated_powerlaw_full_range_lrt_fraction.pdf"

# Clean up datasets by dropping <= 0 values
DROP_NONPOSITIVE_VALUES = True


# ============================================================
# UTILITIES
# ============================================================

def detect_data_root() -> Path:
    """
    Search candidate folders to find where the experimental folders are stored.

    Returns
    -------
    Path
        Path to the experimental root folder.

    Raises
    ------
    FileNotFoundError
        If no experimental root folder can be found.
    """
    if DATA_ROOT is not None:
        return DATA_ROOT

    # Scan default paths
    for candidate in [Path("./data"), Path("./data/avex_data")]:
        if (candidate / "exp1").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find ./data/exp1 or ./data/avex_data/exp1. "
        "Please edit DATA_ROOT at the beginning of the script."
    )


def alphanumeric_key(text: str):
    """
    Key function to sort filenames containing numeric characters in natural order.
    E.g. d2_r0.dat < d10_r0.dat.
    """
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def parse_filename(path: Path) -> Tuple[int, int]:
    """
    Extract degradation (d) and rewiring (r) parameter values from a filename.
    Expected pattern: d{degradation}_r{rewiring}.dat

    Parameters
    ----------
    path : Path
        File path.

    Returns
    -------
    Tuple[int, int]
        (degradation percentage, rewiring percentage).

    Raises
    ------
    ValueError
        If the filename format cannot be recognized.
    """
    match = re.match(r"d(?P<degradation>\d+)_r(?P<rewiring>\d+)\.dat$", path.name)

    if match is None:
        raise ValueError(f"Unrecognized filename: {path.name}")

    return int(match.group("degradation")), int(match.group("rewiring"))


def load_frequency_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a frequency file, filter positive values and non-zero counts, and
    reconstruct a raw data 1D sample by repeating values based on their counts.

    Parameters
    ----------
    path : Path
        File path to load.

    Returns
    -------
    data : np.ndarray
        1D array of repeated raw values.
    values : np.ndarray
        Unique positive values.
    counts : np.ndarray
        Frequencies corresponding to unique values.

    Raises
    ------
    ValueError
        If the file contains invalid shapes or negative frequencies.
    """
    table = np.loadtxt(path, dtype=int)
    table = np.atleast_2d(table)

    if table.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns.")

    values = table[:, 0]
    counts = table[:, 1]

    if np.any(counts < 0):
        raise ValueError(f"{path} contains negative frequencies.")

    if DROP_NONPOSITIVE_VALUES:
        mask = values > 0
        values = values[mask]
        counts = counts[mask]

    mask = counts > 0
    values = values[mask]
    counts = counts[mask]

    if len(values) == 0:
        raise ValueError(f"{path} contains no positive values with frequency > 0.")

    data = np.repeat(values, counts)

    if len(data) == 0:
        raise ValueError(f"{path} produces an empty sample.")

    return data, values, counts


# ============================================================
# FULL-RANGE TRUNCATED POWER-LAW FIT
# ============================================================

def analyze_one_file(job: Tuple[int, Path]) -> Dict[str, object]:
    """
    Process a single data file: load, fit pure and truncated power laws to
    the full range of data, run nested LRT, and collect stats.

    Parameters
    ----------
    job : Tuple[int, Path]
        Tuple of (experiment_id, file_path).

    Returns
    -------
    Dict[str, object]
        Dictionary of analysis result parameters.
    """
    experiment, path = job

    degradation, rewiring = parse_filename(path)
    data, _, _ = load_frequency_file(path)

    # Full range bounds setup
    xmin = int(np.min(data))
    xmax = int(np.max(data))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fit_kwargs = dict(xmin=xmin, discrete=True, verbose=False)
        if USE_XMAX:
            fit_kwargs["xmax"] = xmax

        # Fit models using powerlaw package
        fit = powerlaw.Fit(data, **fit_kwargs)

        truncated_power_law = fit.truncated_power_law
        pure_power_law = fit.power_law

        # Extract exponents and cutoff lambda
        alpha_truncated = float(truncated_power_law.alpha)
        lambda_truncated = float(truncated_power_law.Lambda)
        alpha_power_law = float(pure_power_law.alpha)

        # Nested likelihood-ratio test (chi^2, 1 d.o.f.): does the cutoff
        # parameter lambda significantly improve on the pure power law?
        # powerlaw auto-detects this nested pair, but we make it explicit
        # for robustness across package versions.
        likelihood_ratio, p_value = fit.distribution_compare(
            "power_law",
            "truncated_power_law",
            nested=True,
        )
        likelihood_ratio = float(likelihood_ratio)
        p_value = float(p_value)

    # Sign convention of distribution_compare(A, B): R < 0 favors B (truncated power law).
    tpl_beats_pl = bool(
        np.isfinite(likelihood_ratio)
        and np.isfinite(p_value)
        and (likelihood_ratio < 0)
        and (p_value < P_THRESHOLD)
    )

    return {
        "experiment": experiment,
        "file": path.name,
        "degradation": degradation,
        "rewiring": rewiring,
        "sample_size": int(len(data)),
        "xmin_full": xmin,
        "xmax_full": xmax if USE_XMAX else np.nan,
        "use_xmax": USE_XMAX,
        "alpha_truncated_full": alpha_truncated,
        "lambda_full": lambda_truncated,
        "alpha_power_law_full": alpha_power_law,
        "loglikelihood_ratio_power_law_vs_truncated": likelihood_ratio,
        "p_value_power_law_vs_truncated": p_value,
        "tpl_beats_pl": tpl_beats_pl,
    }


def collect_jobs(data_root: Path) -> List[Tuple[int, Path]]:
    """
    Scan experimental directories to compile a list of job configurations
    matching configured rewiring (r) and degradation (d) limits.

    Parameters
    ----------
    data_root : Path
        Root path to scan.

    Returns
    -------
    List[Tuple[int, Path]]
        List of tuples matching (experiment_id, data_file_path).
    """
    jobs = []

    for experiment in EXPERIMENTS:
        folder = data_root / f"exp{experiment}"

        if not folder.is_dir():
            print(f"Warning: {folder} does not exist. Skipping it.")
            continue

        # Sort files in natural alphanumeric order
        for path in sorted(folder.glob("*.dat"), key=lambda p: alphanumeric_key(p.name)):
            try:
                degradation, rewiring = parse_filename(path)
            except ValueError:
                print(f"Warning: incompatible filename skipped: {path.name}")
                continue

            if rewiring not in R_VALUES:
                continue

            if not (D_MIN <= degradation <= D_MAX):
                continue

            jobs.append((experiment, path))

    return jobs


def worker(job: Tuple[int, Path]) -> Dict[str, object]:
    """
    Process wrapper function executed within process pool workers. Catch exceptions
    to prevent process pool failures.

    Parameters
    ----------
    job : Tuple[int, Path]
        Tuple of (experiment_id, file_path).

    Returns
    -------
    Dict[str, object]
        A dictionary containing execution status and results or errors.
    """
    experiment, path = job

    try:
        return {
            "ok": True,
            "experiment": experiment,
            "file": path.name,
            "result": analyze_one_file(job),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "experiment": experiment,
            "file": path.name,
            "result": None,
            "error": str(exc),
        }


def run_parallel(jobs: List[Tuple[int, Path]]) -> pd.DataFrame:
    """
    Distribute jobs to parallel worker processes using ProcessPoolExecutor.
    Displays progress and logs analysis failures.

    Parameters
    ----------
    jobs : List[Tuple[int, Path]]
        List of (experiment_id, path) jobs.

    Returns
    -------
    pd.DataFrame
        DataFrame of raw results sorted by parameters.
    """
    results = []
    errors = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(worker, job): job for job in jobs}
        total = len(futures)

        for index, future in enumerate(as_completed(futures), start=1):
            experiment, path = futures[future]
            output = future.result()

            if output["ok"]:
                results.append(output["result"])
                print(f"[{index}/{total}] OK exp{experiment}/{path.name}")
            else:
                errors.append(
                    {
                        "experiment": output["experiment"],
                        "file": output["file"],
                        "error": output["error"],
                    }
                )
                print(f"[{index}/{total}] ERROR exp{experiment}/{path.name}: {output['error']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save details of any errors encountered
    if errors:
        errors_file = OUTPUT_DIR / "errors_full_range.csv"
        pd.DataFrame(errors).to_csv(errors_file, index=False)
        print(f"Errors saved to: {errors_file}")

    if not results:
        raise RuntimeError("No valid results were obtained.")

    dataframe = pd.DataFrame(results)
    return dataframe.sort_values(["rewiring", "degradation", "experiment"]).reset_index(drop=True)


def percentile_16(series: pd.Series) -> float:
    """Helper to calculate the 16th percentile (lower bound for 1-sigma equivalent)."""
    return float(np.percentile(series.dropna(), 16))


def percentile_84(series: pd.Series) -> float:
    """Helper to calculate the 84th percentile (upper bound for 1-sigma equivalent)."""
    return float(np.percentile(series.dropna(), 84))


def summarize_by_degradation_and_rewiring(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics (means, standard deviations, medians, percentiles)
    across all runs grouped by degradation and rewiring levels.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw results DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated summary stats DataFrame.
    """
    return (
        dataframe.groupby(["degradation", "rewiring"])
        .agg(
            number_of_runs=("experiment", "count"),
            alpha_mean=("alpha_truncated_full", "mean"),
            alpha_std=("alpha_truncated_full", "std"),
            alpha_median=("alpha_truncated_full", "median"),
            alpha_p16=("alpha_truncated_full", percentile_16),
            alpha_p84=("alpha_truncated_full", percentile_84),
            lambda_mean=("lambda_full", "mean"),
            lambda_std=("lambda_full", "std"),
            lambda_median=("lambda_full", "median"),
            lambda_p16=("lambda_full", percentile_16),
            lambda_p84=("lambda_full", percentile_84),
            alpha_pl_mean=("alpha_power_law_full", "mean"),
            alpha_pl_median=("alpha_power_law_full", "median"),
            # Fraction of runs in which the truncated power law beats the
            # pure power law significantly (nested LRT, p < P_THRESHOLD).
            # This -- not a median of p-values -- is the interpretable
            # aggregate of the test across runs.
            tpl_beats_pl_fraction=("tpl_beats_pl", "mean"),
            median_loglikelihood_ratio_power_law_vs_truncated=(
                "loglikelihood_ratio_power_law_vs_truncated",
                "median",
            ),
        )
        .reset_index()
        .sort_values(["rewiring", "degradation"])
    )


# ============================================================
# FIGURES
# ============================================================

def plot_with_band(
    axis,
    summary: pd.DataFrame,
    rewiring_value: int,
    parameter: str,
    label: str,
    color: str,
):
    """
    Plot parameter evolution line with shaded uncertainty band onto target axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Plot axis.
    summary : pd.DataFrame
        Aggregated summary DataFrame.
    rewiring_value : int
        Rewiring percentage to display.
    parameter : str
        Parameter name ('alpha' or 'lambda').
    label : str
        Label for legend.
    color : str
        Visual color representing the series.
    """
    subset = summary[summary["rewiring"] == rewiring_value].sort_values("degradation")
    x = subset["degradation"].to_numpy()

    if BAND_MODE == "percentile":
        center = subset[f"{parameter}_median"].to_numpy()
        low = subset[f"{parameter}_p16"].to_numpy()
        high = subset[f"{parameter}_p84"].to_numpy()
    elif BAND_MODE == "std":
        center = subset[f"{parameter}_mean"].to_numpy()
        std = subset[f"{parameter}_std"].fillna(0).to_numpy()
        low = center - std
        high = center + std
    else:
        raise ValueError(f"Unknown BAND_MODE: {BAND_MODE}")

    axis.plot(x, center, linewidth=2, color=color, label=label)
    axis.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)


def make_figure(summary: pd.DataFrame):
    """
    Create and save a 1x2 panel figure showing the evolution of the alpha exponent
    and lambda cutoff parameters across degradation levels, using uncertainty bands.

    Parameters
    ----------
    summary : pd.DataFrame
        Aggregated summary DataFrame.
    """
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True)

    color_by_rewiring = {0: "red", 10: "blue"}
    x_ticks = np.arange(0, D_MAX + 1, 10)

    # --- Left Panel: Alpha Exponent ---
    axis = axes[0]
    for rewiring_value in R_VALUES:
        color = color_by_rewiring.get(rewiring_value, "black")
        plot_with_band(
            axis, summary,
            rewiring_value=rewiring_value,
            parameter="alpha",
            label=f"RR={rewiring_value}%",
            color=color,
        )
        # Optionally overlay pure power law alpha values to show divergence
        if PLOT_PURE_PL_ALPHA:
            subset = summary[summary["rewiring"] == rewiring_value].sort_values("degradation")
            center_col = "alpha_pl_median" if BAND_MODE == "percentile" else "alpha_pl_mean"
            axis.plot(
                subset["degradation"], subset[center_col],
                linewidth=1.2, linestyle="--", color=color, alpha=0.7,
                label=f"pure PL, RR={rewiring_value}%",
            )

    axis.set_title(r"Full-range truncated power-law exponent $\alpha$")
    axis.set_xlabel("Nodes removed (%)")
    axis.set_ylabel(r"$\alpha$")
    axis.legend(fontsize=8)

    # --- Right Panel: Lambda parameter ---
    axis = axes[1]
    for rewiring_value in R_VALUES:
        plot_with_band(
            axis, summary,
            rewiring_value=rewiring_value,
            parameter="lambda",
            label=f"RR={rewiring_value}%",
            color=color_by_rewiring.get(rewiring_value, "black"),
        )

    axis.set_title(r"Exponential cutoff parameter $\lambda$")
    axis.set_xlabel("Nodes removed (%)")
    axis.set_ylabel(r"$\lambda$")
    axis.set_ylim(bottom=0)  # lambda is non-negative by definition
    axis.legend(fontsize=8)

    # Formatting both axes
    for axis in axes:
        axis.set_xlim(D_MIN, D_MAX)
        axis.set_xticks(x_ticks)
        axis.minorticks_off()
        axis.grid(True, axis="both", which="major", alpha=0.3)
        axis.grid(False, which="minor")

    band_label = "median, 16th-84th percentile" if BAND_MODE == "percentile" \
        else "mean $\\pm$ 1 s.d."
    support = "bounded support" if USE_XMAX else "unbounded support"
    figure.suptitle(
        "Full-range truncated power-law fits of avalanche-duration "
        f"distributions ({band_label}; {support})",
        fontsize=12,
    )

    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
    plt.show()


def make_lrt_figure(summary: pd.DataFrame):
    """
    Plot and save the fraction of runs where the truncated power law significantly
    beats the pure power law (nested LRT) as a function of network degradation.

    Parameters
    ----------
    summary : pd.DataFrame
        Aggregated summary DataFrame.
    """
    figure, axis = plt.subplots(figsize=(5.4, 4.2))

    color_by_rewiring = {0: "red", 10: "blue"}

    for rewiring_value in R_VALUES:
        subset = summary[summary["rewiring"] == rewiring_value].sort_values("degradation")
        axis.plot(
            subset["degradation"],
            100.0 * subset["tpl_beats_pl_fraction"],
            linewidth=2,
            color=color_by_rewiring.get(rewiring_value, "black"),
            label=f"RR={rewiring_value}%",
        )

    axis.set_xlabel("Nodes removed (%)")
    axis.set_ylabel("Runs where TPL significantly\nbeats pure PL (%)")
    axis.set_title(
        f"Nested likelihood-ratio test ($p < {P_THRESHOLD}$)", fontsize=11
    )
    axis.set_xlim(D_MIN, D_MAX)
    axis.set_ylim(0, 105)
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE_LRT, dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_FIGURE_LRT_PDF, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main entry point: detects root data folder, collects fit jobs, executes them in parallel,
    saves results to CSV files, and generates figures.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_root = detect_data_root()

    print(f"DATA_ROOT = {data_root.resolve()}")
    print(f"N_WORKERS = {N_WORKERS}")
    print(f"USE_XMAX  = {USE_XMAX}")
    print(f"BAND_MODE = {BAND_MODE}")
    print("Fit type: full-range truncated power law")

    jobs = collect_jobs(data_root)

    if not jobs:
        raise RuntimeError("No files matched the selected filters.")

    print(f"Files to analyze: {len(jobs)}")

    # Execute fit jobs in parallel
    dataframe = run_parallel(jobs)
    dataframe.to_csv(OUTPUT_PARAMETERS_CSV, index=False)

    # Compute and save summary aggregations
    summary = summarize_by_degradation_and_rewiring(dataframe)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    # Create parameter evolution and LRT figures
    make_figure(summary)
    make_lrt_figure(summary)

    # Print summary of LRT statistics for publication placeholders
    print()
    print("Analysis completed.")
    print(f"Full parameter CSV: {OUTPUT_PARAMETERS_CSV}")
    print(f"Summary CSV:        {OUTPUT_SUMMARY_CSV}")
    print(f"Figures:            {OUTPUT_FIGURE}")
    print(f"                    {OUTPUT_FIGURE_LRT}")
    print()
    for rewiring_value in R_VALUES:
        subset = dataframe[dataframe["rewiring"] == rewiring_value]
        fraction = 100.0 * subset["tpl_beats_pl"].mean()
        print(
            f"RR={rewiring_value}%: TPL significantly beats pure PL "
            f"(nested LRT, p<{P_THRESHOLD}) in {fraction:.1f}% of runs "
            f"(all degradation levels pooled)."
        )


if __name__ == "__main__":
    main()
