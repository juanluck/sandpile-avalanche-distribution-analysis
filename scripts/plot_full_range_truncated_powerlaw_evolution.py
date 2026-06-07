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

DATA_ROOT: Optional[Path] = None
EXPERIMENTS = range(1, 26)
R_VALUES = [0, 10]
D_MIN = 0
D_MAX = 75
N_WORKERS = 8

OUTPUT_DIR = Path("./truncated_powerlaw_full_range_results")
OUTPUT_PARAMETERS_CSV = OUTPUT_DIR / "truncated_powerlaw_full_range_parameters.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "truncated_powerlaw_full_range_summary.csv"
OUTPUT_FIGURE = OUTPUT_DIR / "truncated_powerlaw_full_range_alpha_lambda.png"
OUTPUT_FIGURE_PDF = OUTPUT_DIR / "truncated_powerlaw_full_range_alpha_lambda.pdf"

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
    experiment, path = job

    degradation, rewiring = parse_filename(path)
    data, _, _ = load_frequency_file(path)

    xmin = int(np.min(data))
    xmax = int(np.max(data))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fit = powerlaw.Fit(
            data,
            xmin=xmin,
            xmax=xmax,
            discrete=True,
            verbose=False,
        )

        truncated_power_law = fit.truncated_power_law
        pure_power_law = fit.power_law

        alpha_truncated = float(truncated_power_law.alpha)
        lambda_truncated = float(truncated_power_law.Lambda)
        alpha_power_law = float(pure_power_law.alpha)

        likelihood_ratio, p_value = fit.distribution_compare(
            "power_law",
            "truncated_power_law",
        )

    return {
        "experiment": experiment,
        "file": path.name,
        "degradation": degradation,
        "rewiring": rewiring,
        "sample_size": int(len(data)),
        "xmin_full": xmin,
        "xmax_full": xmax,
        "alpha_truncated_full": alpha_truncated,
        "lambda_full": lambda_truncated,
        "alpha_power_law_full": alpha_power_law,
        "loglikelihood_ratio_power_law_vs_truncated": float(likelihood_ratio),
        "p_value_power_law_vs_truncated": float(p_value),
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
        errors_file = OUTPUT_DIR / "errors_full_range.csv"
        pd.DataFrame(errors).to_csv(errors_file, index=False)
        print(f"Errors saved to: {errors_file}")

    if not results:
        raise RuntimeError("No valid results were obtained.")

    dataframe = pd.DataFrame(results)
    return dataframe.sort_values(["rewiring", "degradation", "experiment"]).reset_index(drop=True)


def summarize_by_degradation_and_rewiring(dataframe: pd.DataFrame) -> pd.DataFrame:
    return (
        dataframe.groupby(["degradation", "rewiring"])
        .agg(
            number_of_runs=("experiment", "count"),
            alpha_mean=("alpha_truncated_full", "mean"),
            alpha_std=("alpha_truncated_full", "std"),
            alpha_median=("alpha_truncated_full", "median"),
            lambda_mean=("lambda_full", "mean"),
            lambda_std=("lambda_full", "std"),
            lambda_median=("lambda_full", "median"),
            median_loglikelihood_ratio_power_law_vs_truncated=(
                "loglikelihood_ratio_power_law_vs_truncated",
                "median",
            ),
            median_p_value_power_law_vs_truncated=(
                "p_value_power_law_vs_truncated",
                "median",
            ),
        )
        .reset_index()
        .sort_values(["rewiring", "degradation"])
    )


def plot_mean_with_band(
    axis,
    summary: pd.DataFrame,
    rewiring_value: int,
    y_mean: str,
    y_std: str,
    label: str,
    color: str,
):
    subset = summary[summary["rewiring"] == rewiring_value].sort_values("degradation")

    x = subset["degradation"].to_numpy()
    mean = subset[y_mean].to_numpy()
    std = subset[y_std].fillna(0).to_numpy()

    axis.plot(x, mean, linewidth=2, color=color, label=label)
    axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def make_figure(summary: pd.DataFrame):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True)

    color_by_rewiring = {0: "red", 10: "blue"}
    x_ticks = np.arange(0, D_MAX + 1, 10)

    axis = axes[0]
    for rewiring_value in R_VALUES:
        plot_mean_with_band(
            axis,
            summary,
            rewiring_value=rewiring_value,
            y_mean="alpha_mean",
            y_std="alpha_std",
            label=f"RR={rewiring_value}%",
            color=color_by_rewiring.get(rewiring_value, "black"),
        )

    axis.set_title(r"Full-range truncated power-law exponent $\alpha$")
    axis.set_xlabel("Nodes removed (%)")
    axis.set_ylabel(r"$\alpha$")
    axis.legend()

    axis = axes[1]
    for rewiring_value in R_VALUES:
        plot_mean_with_band(
            axis,
            summary,
            rewiring_value=rewiring_value,
            y_mean="lambda_mean",
            y_std="lambda_std",
            label=f"RR={rewiring_value}%",
            color=color_by_rewiring.get(rewiring_value, "black"),
        )

    axis.set_title(r"Exponential cutoff parameter $\lambda$")
    axis.set_xlabel("Nodes removed (%)")
    axis.set_ylabel(r"$\lambda$")
    axis.legend()

    for axis in axes:
        axis.set_xlim(D_MIN, D_MAX)
        axis.set_xticks(x_ticks)
        axis.minorticks_off()
        axis.grid(True, axis="both", which="major", alpha=0.3)
        axis.grid(False, which="minor")

    figure.suptitle(
        "Full-range truncated power-law fits of avalanche-duration distributions",
        fontsize=13,
    )

    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
    plt.show()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_root = detect_data_root()

    print(f"DATA_ROOT = {data_root.resolve()}")
    print(f"N_WORKERS = {N_WORKERS}")
    print("Fit type: full-range truncated power law")

    jobs = collect_jobs(data_root)

    if not jobs:
        raise RuntimeError("No files matched the selected filters.")

    print(f"Files to analyze: {len(jobs)}")

    dataframe = run_parallel(jobs)
    dataframe.to_csv(OUTPUT_PARAMETERS_CSV, index=False)

    summary = summarize_by_degradation_and_rewiring(dataframe)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    make_figure(summary)

    print()
    print("Analysis completed.")
    print(f"Full parameter CSV: {OUTPUT_PARAMETERS_CSV}")
    print(f"Summary CSV:        {OUTPUT_SUMMARY_CSV}")
    print(f"Figure:             {OUTPUT_FIGURE}")
    print(f"PDF:                {OUTPUT_FIGURE_PDF}")


if __name__ == "__main__":
    main()
