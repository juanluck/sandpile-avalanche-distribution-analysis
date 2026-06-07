from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("./truncated_powerlaw_full_range_results")

SUMMARY_CSV = INPUT_DIR / "truncated_powerlaw_full_range_summary.csv"
PARAMETERS_CSV = INPUT_DIR / "truncated_powerlaw_full_range_parameters.csv"

OUTPUT_FIGURE = INPUT_DIR / "truncated_powerlaw_full_range_alpha_lambda_from_csv.png"
OUTPUT_FIGURE_PDF = INPUT_DIR / "truncated_powerlaw_full_range_alpha_lambda_from_csv.pdf"

R_VALUES = [0, 10]
D_MIN = 0
D_MAX = 75

COLOR_BY_REWIRING = {0: "red", 10: "blue"}


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
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
    dataframe = dataframe.copy()

    if "d" in dataframe.columns and "degradation" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"d": "degradation"})

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
    dataframe = dataframe.copy()

    if "d" in dataframe.columns and "degradation" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"d": "degradation"})

    if "r" in dataframe.columns and "rewiring" not in dataframe.columns:
        dataframe = dataframe.rename(columns={"r": "rewiring"})

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
    subset = summary[summary["rewiring"] == rewiring_value].sort_values("degradation")

    subset = subset[
        (subset["degradation"] >= D_MIN)
        & (subset["degradation"] <= D_MAX)
    ]

    x = subset["degradation"].to_numpy()
    mean = subset[y_mean].to_numpy()
    std = subset[y_std].fillna(0).to_numpy()

    axis.plot(x, mean, linewidth=2.5, color=color, label=label)
    axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def make_figure(summary: pd.DataFrame):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True)
    x_ticks = np.arange(D_MIN, D_MAX + 1, 10)

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
    summary = load_data()

    summary = summary[summary["rewiring"].isin(R_VALUES)]
    summary = summary[
        (summary["degradation"] >= D_MIN)
        & (summary["degradation"] <= D_MAX)
    ]

    make_figure(summary)


if __name__ == "__main__":
    main()
