"""
Parallel model selection for avalanche-duration distributions across multiple runs.
Fits candidate probability distributions (power law, lognormal, exponential, 
stretched exponential, truncated power law) to experimental avalanche data.
Computes and compares information criteria (AIC, AICc, BIC) and performs nested
likelihood-ratio tests (LRT) to identify the best-fitting distribution model
under various network degradation and rewiring conditions.
"""

from __future__ import annotations

import os

# Avoid CPU oversubscription if NumPy/SciPy use multithreaded BLAS internally.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import powerlaw


# ============================================================
# CONFIGURATION
# ============================================================

# Path to the experimental data directory. If None, it will be auto-detected.
DATA_ROOT: Optional[Path] = None

# Range of experiment indices to analyze
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

# Output folder for saving model selection CSV outputs
OUTPUT_DIR = Path("./distribution_results/all_experiments")

# Explicit candidate set: the five distributions reported in the paper.
# (Do NOT use None: depending on the installed powerlaw version, the
# supported set also includes lognormal_positive and others, which would
# silently add undocumented candidates to the model selection.)
DISTRIBUTIONS: List[str] = [
    "power_law",
    "lognormal",
    "exponential",
    "stretched_exponential",
    "truncated_power_law",
]

# Number of free parameters of each candidate (needed for AIC/AICc/BIC).
N_PARAMS: Dict[str, int] = {
    "power_law": 1,
    "exponential": 1,
    "lognormal": 2,
    "stretched_exponential": 2,
    "truncated_power_law": 2,
    "lognormal_positive": 2,
}

# Criterion used for the 'best_distribution' column (and hence Figure 9).
# One of: "AICc", "AIC", "BIC", "loglik" (legacy behaviour, no penalty).
# AICc is recommended for finite sample sizes.
SELECTION_CRITERION = "AICc"

# Fit range behavior:
# "tail": powerlaw estimates xmin automatically.
# "full": force xmin=min(data), xmax=max(data).
FIT_MODE = "tail"

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
        raise ValueError(f"{path} must contain at least two columns: value frequency")

    values = table[:, 0]
    counts = table[:, 1]

    if np.any(counts < 0):
        raise ValueError(f"{path} contains negative frequencies")

    if DROP_NONPOSITIVE_VALUES:
        mask = values > 0
        values = values[mask]
        counts = counts[mask]

    mask = counts > 0
    values = values[mask]
    counts = counts[mask]

    if len(values) == 0:
        raise ValueError(f"{path} contains no positive values with frequency > 0")

    data = np.repeat(values, counts)

    if len(data) == 0:
        raise ValueError(f"{path} produces an empty sample")

    return data, values, counts


def create_fit(data: np.ndarray) -> powerlaw.Fit:
    """
    Configure and instantiate the powerlaw.Fit object based on FIT_MODE.

    Parameters
    ----------
    data : np.ndarray
        Raw reconstructed dataset array.

    Returns
    -------
    powerlaw.Fit
        Configured Fit instance.

    Raises
    ------
    ValueError
        If FIT_MODE is not recognized.
    """
    if FIT_MODE == "tail":
        # Estimate xmin automatically (fitting heavy-tail portion)
        return powerlaw.Fit(data, discrete=True, verbose=False)

    if FIT_MODE == "full":
        # Force fitting to cover the complete data range
        return powerlaw.Fit(
            data,
            xmin=int(np.min(data)),
            xmax=int(np.max(data)),
            discrete=True,
            verbose=False,
        )

    raise ValueError(f"Unknown FIT_MODE: {FIT_MODE}")


def tail_data(fit: powerlaw.Fit, data: np.ndarray) -> np.ndarray:
    """
    Extract the subset of data actually used by the powerlaw model fits
    (satisfying xmin and optional xmax boundaries).

    Parameters
    ----------
    fit : powerlaw.Fit
        Instantiated fit model.
    data : np.ndarray
        Complete reconstructed raw data array.

    Returns
    -------
    np.ndarray
        Data subset within fit boundaries.
    """
    tail = data[data >= fit.xmin]
    xmax = getattr(fit, "xmax", None)
    if xmax is not None:
        tail = tail[tail <= xmax]
    return tail


# ============================================================
# MODEL SELECTION (information criteria + likelihood-ratio tests)
# ============================================================

def information_criteria(loglik: float, k: int, n: int) -> Tuple[float, float, float]:
    """
    Calculate AIC, AICc, and BIC information criteria scores.
    AICc returns infinity if sample size n is too small (n <= k + 1).

    Parameters
    ----------
    loglik : float
        Fitted log-likelihood value.
    k : int
        Number of free model parameters.
    n : int
        Number of sample data points.

    Returns
    -------
    Tuple[float, float, float]
        (AIC, AICc, BIC) scores.
    """
    aic = 2 * k - 2 * loglik
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else math.inf
    bic = k * math.log(n) - 2 * loglik
    return aic, aicc, bic


def score_distributions(fit: powerlaw.Fit, data: np.ndarray) -> pd.DataFrame:
    """
    Compute log-likelihood, AIC, AICc, and BIC scores for all candidate models
    evaluated over the identical data subset.

    Parameters
    ----------
    fit : powerlaw.Fit
        Powerlaw Fit instance.
    data : np.ndarray
        Complete reconstructed raw data array.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing fits and criteria scores for each candidate distribution.
    """
    tail = tail_data(fit, data)
    n = int(tail.size)
    rows = []

    for name in DISTRIBUTIONS:
        try:
            dist = getattr(fit, name)
            # Sum individual log-likelihoods
            loglik = float(np.sum(dist.loglikelihoods(tail)))
            if not np.isfinite(loglik):
                raise ValueError("non-finite log-likelihood")
            k = N_PARAMS[name]
            aic, aicc, bic = information_criteria(loglik, k, n)
            rows.append(dict(distribution=name, k=k, n=n, loglik=loglik,
                             AIC=aic, AICc=aicc, BIC=bic, error=""))
        except Exception as exc:
            # Handle fit errors gracefully
            rows.append(dict(distribution=name, k=N_PARAMS.get(name, np.nan),
                             n=n, loglik=np.nan, AIC=np.nan, AICc=np.nan,
                             BIC=np.nan, error=str(exc)))

    return pd.DataFrame(rows)


def select_winner(scores: pd.DataFrame, criterion: str) -> Tuple[str, str]:
    """
    Determine the best and runner-up candidate distributions based on the chosen criterion.
    Lower criteria scores are preferred, except for log-likelihood where higher is better.

    Parameters
    ----------
    scores : pd.DataFrame
        Candidate model scores DataFrame.
    criterion : str
        Selection criterion (e.g. 'AICc', 'BIC', 'loglik').

    Returns
    -------
    Tuple[str, str]
        (best distribution name, runner-up distribution name).

    Raises
    ------
    RuntimeError
        If fewer than two candidates fit successfully.
    """
    valid = scores[scores["error"] == ""].copy()
    if len(valid) < 2:
        raise RuntimeError("Fewer than two valid distributions were available")

    # Ascending sort is preferred for information criteria (lower is better)
    ascending = criterion != "loglik"
    valid = valid.sort_values(criterion, ascending=ascending)
    return valid.iloc[0]["distribution"], valid.iloc[1]["distribution"]


def compare_pair(fit: powerlaw.Fit, dist1: str, dist2: str) -> Tuple[float, float]:
    """
    Compare two fitted distributions using a likelihood-ratio test.
    Automatically applies nested correction (nested chi-squared test, 1 d.o.f.)
    if one candidate name is nested in the other (e.g., power law vs. truncated power law).

    Parameters
    ----------
    fit : powerlaw.Fit
        Powerlaw Fit instance.
    dist1 : str
        First distribution name.
    dist2 : str
        Second distribution name.

    Returns
    -------
    Tuple[float, float]
        (likelihood ratio R, significance p-value).
    """
    nested = (dist1 in dist2) or (dist2 in dist1)
    likelihood_ratio, p_value = fit.distribution_compare(dist1, dist2, nested=nested)
    return float(likelihood_ratio), float(p_value)


# ============================================================
# ANALYSIS
# ============================================================

def analyze_one_file(path: Path, experiment: int) -> Dict[str, object]:
    """
    Perform complete model selection pipeline on a single file: loading, fitting,
    calculating criteria, identifying winners, and running significance tests.

    Parameters
    ----------
    path : Path
        File path.
    experiment : int
        Experiment ID.

    Returns
    -------
    Dict[str, object]
        Analysis results record.
    """
    degradation, rewiring = parse_filename(path)
    data, values, _ = load_frequency_file(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = create_fit(data)
        scores = score_distributions(fit, data)

        # Retrieve winner and runner-up based on configured criterion
        best, second = select_winner(scores, SELECTION_CRITERION)

        # Significance test of winner vs. runner-up
        likelihood_ratio, p_value = compare_pair(fit, best, second)
        significant = bool((likelihood_ratio > 0) and (p_value < P_THRESHOLD))

        # Explicit nested LRT: power law vs. truncated power law
        try:
            R_pl_tpl, p_pl_tpl = compare_pair(fit, "power_law",
                                              "truncated_power_law")
        except Exception:
            R_pl_tpl, p_pl_tpl = np.nan, np.nan

    # Identify winning model under each different information criterion
    valid = scores[scores["error"] == ""]
    winners_ic = {}
    for crit in ("AIC", "AICc", "BIC"):
        winners_ic[crit] = valid.sort_values(crit).iloc[0]["distribution"]
    winner_loglik = valid.sort_values("loglik", ascending=False).iloc[0]["distribution"]

    # String documenting full ranked list of candidate models
    ranking_text = "; ".join(
        f"{row.distribution}:{getattr(row, SELECTION_CRITERION):.6g}"
        for row in valid.sort_values(
            SELECTION_CRITERION,
            ascending=(SELECTION_CRITERION != "loglik")).itertuples()
    )

    return {
        "experiment": experiment,
        "file": path.name,
        "d": degradation,
        "r": rewiring,
        "n": int(len(data)),
        "n_tail": int(valid["n"].iloc[0]) if len(valid) else np.nan,
        "value_min": int(np.min(values)),
        "value_max": int(np.max(values)),
        "fit_mode": FIT_MODE,
        "selection_criterion": SELECTION_CRITERION,
        "xmin_fit": fit.xmin,
        "xmax_fit": getattr(fit, "xmax", np.nan),
        # --- columns kept for compatibility with downstream scripts ---
        "best_distribution": best,
        "second_distribution": second,
        "R_best_vs_second": likelihood_ratio,
        "p_best_vs_second": p_value,
        "significant": significant,
        "winner_significant": best if significant else "undecided",
        "ranking": ranking_text,
        # --- new columns: winners under each criterion ---
        "winner_AIC": winners_ic["AIC"],
        "winner_AICc": winners_ic["AICc"],
        "winner_BIC": winners_ic["BIC"],
        "winner_loglik": winner_loglik,
        # --- new columns: nested LRT power_law vs truncated_power_law ---
        # R < 0 and p < 0.05 means the truncated power law is
        # significantly better than the pure power law.
        "LRT_R_pl_vs_tpl": R_pl_tpl,
        "LRT_p_pl_vs_tpl": p_pl_tpl,
        "tpl_beats_pl_significantly": bool(
            np.isfinite(R_pl_tpl) and np.isfinite(p_pl_tpl)
            and (R_pl_tpl < 0) and (p_pl_tpl < P_THRESHOLD)
        ),
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

        # Sort files alphanumeric-wise
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
    Wrapper function executed within parallel processes. Catch exceptions
    to prevent process pool failures.

    Parameters
    ----------
    job : Tuple[int, Path]
        Tuple of (experiment_id, file_path).

    Returns
    -------
    Dict[str, object]
        Execution status and results or errors.
    """
    experiment, path = job

    try:
        return {
            "ok": True,
            "experiment": experiment,
            "file": path.name,
            "result": analyze_one_file(path, experiment),
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

    # Save execution errors to CSV file
    if errors:
        errors_file = OUTPUT_DIR / "errors.csv"
        pd.DataFrame(errors).to_csv(errors_file, index=False)
        print(f"Errors saved to: {errors_file}")

    if not results:
        raise RuntimeError("No valid results were obtained")

    dataframe = pd.DataFrame(results)
    return dataframe.sort_values(["r", "d", "experiment"]).reset_index(drop=True)


def summarize_percentages(
    dataframe: pd.DataFrame,
    winner_column: str,
    include_undecided: bool,
) -> pd.DataFrame:
    """
    Compile summary statistics reporting winning percentages of distribution models
    across degradation levels and rewiring rates.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame of raw results.
    winner_column : str
        Column containing the winning distribution names.
    include_undecided : bool
        If True, include 'undecided' category counts.

    Returns
    -------
    pd.DataFrame
        Summary percentages DataFrame.
    """
    d_values = sorted(dataframe["d"].unique())
    r_values = sorted(dataframe["r"].unique())
    distributions = sorted(dataframe[winner_column].dropna().unique())

    if include_undecided and "undecided" not in distributions:
        distributions.append("undecided")

    totals = dataframe.groupby(["d", "r"]).size().reset_index(name="total")

    counts = (
        dataframe.groupby(["d", "r", winner_column])
        .size()
        .reset_index(name="count")
        .rename(columns={winner_column: "distribution"})
    )

    # Reindex to ensure all permutations are covered
    full_index = pd.MultiIndex.from_product(
        [d_values, r_values, distributions],
        names=["d", "r", "distribution"],
    )

    counts = (
        counts.set_index(["d", "r", "distribution"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    summary = counts.merge(totals, on=["d", "r"], how="left")
    summary["percentage"] = 100.0 * summary["count"] / summary["total"]

    return summary.sort_values(["distribution", "r", "d"]).reset_index(drop=True)


def summarize_lrt_placeholders(
    dataframe: pd.DataFrame,
    strict_threshold: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute aggregate statistics for the manuscript placeholders [X] (regular lattice)
    and [Y] (weakly rewired case):
        fraction of runs in which the truncated power law significantly
        beats the pure power law (nested LRT), aggregated over all
        degraded configurations (d > 0).

    Parameters
    ----------
    dataframe : pd.DataFrame
        Complete results DataFrame.
    strict_threshold : float
        Stricter secondary p-value threshold for robustness checks.

    Returns
    -------
    placeholders : pd.DataFrame
        Pooled fractions for both normal (0.05) and strict (0.01) p-value limits.
    by_d_r : pd.DataFrame
        Fraction details for every individual (d, r) pair.
    """
    # Exclude the intact network (d=0): manuscript refers to "degraded configurations"
    degraded = dataframe[dataframe["d"] > 0].copy()

    # Filter to valid LRT computations
    valid = (
        np.isfinite(degraded["LRT_R_pl_vs_tpl"])
        & np.isfinite(degraded["LRT_p_pl_vs_tpl"])
    )
    degraded = degraded[valid]

    # Evaluate significance under stricter threshold
    degraded["tpl_sig_strict"] = (
        (degraded["LRT_R_pl_vs_tpl"] < 0)
        & (degraded["LRT_p_pl_vs_tpl"] < strict_threshold)
    )

    rows = []
    groups = [(f"RR={r}%", degraded[degraded["r"] == r])
              for r in sorted(degraded["r"].unique())]
    groups.append(("pooled (all RR)", degraded))

    # Calculate fraction metrics
    for label, subset in groups:
        n_runs = len(subset)
        n_sig = int(subset["tpl_beats_pl_significantly"].sum())
        n_sig_strict = int(subset["tpl_sig_strict"].sum())
        rows.append(dict(
            group=label,
            n_runs=n_runs,
            n_significant=n_sig,
            fraction_significant_pct=100.0 * n_sig / n_runs if n_runs else np.nan,
            p_threshold=P_THRESHOLD,
            n_significant_strict=n_sig_strict,
            fraction_significant_strict_pct=(
                100.0 * n_sig_strict / n_runs if n_runs else np.nan
            ),
            strict_threshold=strict_threshold,
            note="d=0 excluded; R<0 means TPL favored",
        ))

    placeholders = pd.DataFrame(rows)

    # Detailed statistics breakdown per (d, r) configuration
    by_d_r = (
        degraded.groupby(["d", "r"])
        .agg(
            n_runs=("tpl_beats_pl_significantly", "size"),
            n_significant=("tpl_beats_pl_significantly", "sum"),
            fraction_significant_pct=(
                "tpl_beats_pl_significantly",
                lambda s: 100.0 * s.mean(),
            ),
            fraction_significant_strict_pct=(
                "tpl_sig_strict",
                lambda s: 100.0 * s.mean(),
            ),
            median_LRT_R=("LRT_R_pl_vs_tpl", "median"),
        )
        .reset_index()
        .sort_values(["r", "d"])
    )

    return placeholders, by_d_r


def main():
    """
    Main driver execution pipeline. Scans paths, distributes fits in parallel,
    runs criteria and likelihood tests, saves summary CSV files, and prints publication placeholders.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_root = detect_data_root()

    print(f"DATA_ROOT = {data_root.resolve()}")
    print(f"N_WORKERS = {N_WORKERS}")
    print(f"FIT_MODE = {FIT_MODE}")
    print(f"SELECTION_CRITERION = {SELECTION_CRITERION}")

    jobs = collect_jobs(data_root)

    if not jobs:
        raise RuntimeError("No files matched the selected filters")

    print(f"Files to analyze: {len(jobs)}")

    # Execute jobs using process pool
    dataframe = run_parallel(jobs)

    output_all = OUTPUT_DIR / "best_distributions_all.csv"
    output_by_file = OUTPUT_DIR / "best_distributions_by_file.csv"

    # Save raw fit outcomes
    dataframe.to_csv(output_all, index=False)
    dataframe.to_csv(output_by_file, index=False)

    # Calculate and save winning percentage summaries
    summarize_percentages(
        dataframe,
        winner_column="best_distribution",
        include_undecided=False,
    ).to_csv(OUTPUT_DIR / "distribution_percentages_raw.csv", index=False)

    summarize_percentages(
        dataframe,
        winner_column="winner_significant",
        include_undecided=True,
    ).to_csv(OUTPUT_DIR / "distribution_percentages_significant.csv", index=False)

    # Save summaries for individual information criteria
    for crit in ("AIC", "AICc", "BIC", "loglik"):
        summarize_percentages(
            dataframe,
            winner_column=f"winner_{crit}",
            include_undecided=False,
        ).to_csv(OUTPUT_DIR / f"distribution_percentages_{crit}.csv",
                 index=False)

    # Calculate overall comparison agreement metrics
    agree = (dataframe["winner_AIC"] == dataframe["winner_BIC"]).mean()
    tpl_sig = dataframe["tpl_beats_pl_significantly"].mean()
    legacy_diff = (dataframe["winner_loglik"]
                   != dataframe[f"winner_{SELECTION_CRITERION}"]).mean()

    # --- Compute manuscript placeholders [X] / [Y] -------------------------
    placeholders, lrt_by_d_r = summarize_lrt_placeholders(dataframe)

    placeholders_file = OUTPUT_DIR / "lrt_placeholders_summary.csv"
    lrt_by_d_r_file = OUTPUT_DIR / "lrt_fraction_by_d_r.csv"
    placeholders.to_csv(placeholders_file, index=False)
    lrt_by_d_r.to_csv(lrt_by_d_r_file, index=False)

    print()
    print("Analysis completed.")
    print(f"Full results:       {output_all}")
    print(f"Full results alias: {output_by_file}")
    print(f"LRT placeholders:   {placeholders_file}")
    print(f"LRT detail (d, r):  {lrt_by_d_r_file}")
    print()
    print(f"AIC/BIC winner agreement:                  {100 * agree:.1f}% of files")
    print(f"TPL significantly beats pure PL (LRT):     {100 * tpl_sig:.1f}% of files (all, incl. d=0)")
    print(f"Files where penalty changes the winner vs "
          f"raw log-likelihood ranking:                {100 * legacy_diff:.1f}%")
    print()
    print("=" * 64)
    print("MANUSCRIPT PLACEHOLDERS (degraded configurations only, d > 0)")
    print("Fraction of runs where the exponential cutoff is statistically")
    print("required: nested LRT, R < 0 and p < threshold.")
    print("=" * 64)
    label_map = {"RR=0%": "[X]", "RR=10%": "[Y]"}
    for row in placeholders.itertuples():
        tag = label_map.get(row.group, "   ")
        print(f"{tag} {row.group:<16} "
              f"p<{row.p_threshold}: {row.fraction_significant_pct:5.1f}% "
              f"({row.n_significant}/{row.n_runs})   |   "
              f"p<{row.strict_threshold}: "
              f"{row.fraction_significant_strict_pct:5.1f}% "
              f"({row.n_significant_strict}/{row.n_runs})")
    print("=" * 64)
    # --- Sanity check on the LRT values themselves -----------------
    print()
    print("LRT sanity check (R should be negative with varied")
    print("magnitudes; the largest p-values should be tiny):")
    print(dataframe[["LRT_R_pl_vs_tpl", "LRT_p_pl_vs_tpl"]].describe())
    print()
    print("5 runs with the largest p-values:")
    print(dataframe.nlargest(5, "LRT_p_pl_vs_tpl")[
        ["d", "r", "experiment", "LRT_R_pl_vs_tpl", "LRT_p_pl_vs_tpl"]
    ])


if __name__ == "__main__":
    main()
