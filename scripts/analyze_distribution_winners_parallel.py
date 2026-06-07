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
import powerlaw


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT: Optional[Path] = None
EXPERIMENTS = range(1, 26)
R_VALUES = [0, 10]
D_MIN = 0
D_MAX = 75
N_WORKERS = 8
P_THRESHOLD = 0.05

OUTPUT_DIR = Path("./distribution_results/all_experiments")

# None = use all distributions supported by the installed powerlaw version.
DISTRIBUTIONS: Optional[List[str]] = None

# "tail": powerlaw estimates xmin automatically.
# "full": force xmin=min(data), xmax=max(data).
FIT_MODE = "tail"

DROP_NONPOSITIVE_VALUES = True


# ============================================================
# UTILITIES
# ============================================================

def detect_data_root() -> Path:
    if DATA_ROOT is not None:
        return DATA_ROOT

    for candidate in [Path("./data"), Path("./data/avex_data")]:
        if (candidate / "exp1").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not find ./data/exp1 or ./data/avex_data/exp1. "
        "Please edit DATA_ROOT at the beginning of the script."
    )


def alphanumeric_key(text: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def parse_filename(path: Path) -> Tuple[int, int]:
    match = re.match(r"d(?P<degradation>\d+)_r(?P<rewiring>\d+)\.dat$", path.name)

    if match is None:
        raise ValueError(f"Unrecognized filename: {path.name}")

    return int(match.group("degradation")), int(match.group("rewiring"))


def load_frequency_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    if FIT_MODE == "tail":
        return powerlaw.Fit(data, discrete=True, verbose=False)

    if FIT_MODE == "full":
        return powerlaw.Fit(
            data,
            xmin=int(np.min(data)),
            xmax=int(np.max(data)),
            discrete=True,
            verbose=False,
        )

    raise ValueError(f"Unknown FIT_MODE: {FIT_MODE}")


def get_supported_distributions(fit: powerlaw.Fit) -> List[str]:
    if DISTRIBUTIONS is not None:
        return DISTRIBUTIONS

    if hasattr(fit, "supported_distributions"):
        return list(fit.supported_distributions.keys())

    return [
        "power_law",
        "lognormal",
        "exponential",
        "truncated_power_law",
        "stretched_exponential",
        "lognormal_positive",
    ]


# ============================================================
# MODEL SELECTION
# ============================================================

def fit_available_distributions(fit: powerlaw.Fit, distributions: List[str]) -> List[str]:
    valid = []

    for distribution in distributions:
        try:
            getattr(fit, distribution)
            valid.append(distribution)
        except Exception:
            pass

    return valid


def rank_distributions(fit: powerlaw.Fit, distributions: List[str]) -> Dict[str, object]:
    """
    Rank candidate distributions using relative log-likelihood ratios.

    A baseline distribution is selected. For each alternative distribution:

        score(distribution) = log L(distribution) - log L(baseline)

    The largest score is treated as the best-supported distribution within
    this candidate set. The best and second-best distributions are then
    compared directly to obtain R and p.
    """
    valid = fit_available_distributions(fit, distributions)

    if len(valid) < 2:
        raise RuntimeError("Fewer than two valid distributions were available")

    baseline = valid[0]
    scores = {baseline: 0.0}

    for distribution in valid[1:]:
        try:
            likelihood_ratio, _ = fit.distribution_compare(distribution, baseline)
            scores[distribution] = float(likelihood_ratio)
        except Exception:
            continue

    if len(scores) < 2:
        raise RuntimeError("Not enough valid distribution comparisons")

    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best = ranking[0][0]
    second = ranking[1][0]

    likelihood_ratio, p_value = fit.distribution_compare(best, second)
    likelihood_ratio = float(likelihood_ratio)
    p_value = float(p_value)

    significant = bool((likelihood_ratio > 0) and (p_value < P_THRESHOLD))

    return {
        "valid_distributions": valid,
        "ranking": ranking,
        "best_distribution": best,
        "second_distribution": second,
        "R_best_vs_second": likelihood_ratio,
        "p_best_vs_second": p_value,
        "significant": significant,
        "winner_significant": best if significant else "undecided",
    }


# ============================================================
# ANALYSIS
# ============================================================

def analyze_one_file(path: Path, experiment: int) -> Dict[str, object]:
    degradation, rewiring = parse_filename(path)
    data, values, _ = load_frequency_file(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = create_fit(data)

    ranking_info = rank_distributions(fit, get_supported_distributions(fit))

    ranking_text = "; ".join(
        f"{name}:{score:.6g}" for name, score in ranking_info["ranking"]
    )

    return {
        "experiment": experiment,
        "file": path.name,
        "d": degradation,
        "r": rewiring,
        "n": int(len(data)),
        "value_min": int(np.min(values)),
        "value_max": int(np.max(values)),
        "fit_mode": FIT_MODE,
        "xmin_fit": fit.xmin,
        "xmax_fit": getattr(fit, "xmax", np.nan),
        "best_distribution": ranking_info["best_distribution"],
        "second_distribution": ranking_info["second_distribution"],
        "R_best_vs_second": ranking_info["R_best_vs_second"],
        "p_best_vs_second": ranking_info["p_best_vs_second"],
        "significant": ranking_info["significant"],
        "winner_significant": ranking_info["winner_significant"],
        "ranking": ranking_text,
    }


def collect_jobs(data_root: Path) -> List[Tuple[int, Path]]:
    jobs = []

    for experiment in EXPERIMENTS:
        folder = data_root / f"exp{experiment}"

        if not folder.is_dir():
            print(f"Warning: {folder} does not exist. Skipping it.")
            continue

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
    d_values = sorted(dataframe["d"].unique())
    r_values = sorted(dataframe["r"].unique())
    distributions = sorted(dataframe["best_distribution"].dropna().unique())

    if include_undecided and "undecided" not in distributions:
        distributions.append("undecided")

    totals = dataframe.groupby(["d", "r"]).size().reset_index(name="total")

    counts = (
        dataframe.groupby(["d", "r", winner_column])
        .size()
        .reset_index(name="count")
        .rename(columns={winner_column: "distribution"})
    )

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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_root = detect_data_root()

    print(f"DATA_ROOT = {data_root.resolve()}")
    print(f"N_WORKERS = {N_WORKERS}")
    print(f"FIT_MODE = {FIT_MODE}")

    jobs = collect_jobs(data_root)

    if not jobs:
        raise RuntimeError("No files matched the selected filters")

    print(f"Files to analyze: {len(jobs)}")

    dataframe = run_parallel(jobs)

    output_all = OUTPUT_DIR / "best_distributions_all.csv"
    output_by_file = OUTPUT_DIR / "best_distributions_by_file.csv"

    dataframe.to_csv(output_all, index=False)
    dataframe.to_csv(output_by_file, index=False)

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

    print()
    print("Analysis completed.")
    print(f"Full results:       {output_all}")
    print(f"Full results alias: {output_by_file}")


if __name__ == "__main__":
    main()
