from pathlib import Path

import numpy as np
import powerlaw
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("./data/avex_data/exp1")

FILES_TO_PLOT = [
    "d0_r0.dat",
    "d45_r10.dat",
]


# ============================================================
# FUNCTIONS
# ============================================================

def load_frequency_file(path: Path):
    table = np.loadtxt(path, dtype=int)
    table = np.atleast_2d(table)

    values = table[:, 0]
    counts = table[:, 1]

    mask = (values > 0) & (counts > 0)
    values = values[mask]
    counts = counts[mask]

    data = np.repeat(values, counts)

    return data, values, counts


def plot_full_truncated_powerlaw(file_name: str):
    path = DATA_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    data, values, counts = load_frequency_file(path)

    xmin = int(np.min(data))
    xmax = int(np.max(data))

    fit_full = powerlaw.Fit(
        data,
        xmin=xmin,
        xmax=xmax,
        discrete=True,
        verbose=False,
    )

    truncated_power_law = fit_full.truncated_power_law
    pure_power_law = fit_full.power_law

    alpha_truncated = truncated_power_law.alpha
    lambda_truncated = truncated_power_law.Lambda
    alpha_power_law = pure_power_law.alpha

    likelihood_ratio, p_value = fit_full.distribution_compare(
        "power_law",
        "truncated_power_law",
    )

    print()
    print(f"File: {file_name}")
    print(f"n = {len(data)}")
    print(f"Full range: xmin={xmin}, xmax={xmax}")
    print(f"power_law alpha = {alpha_power_law}")
    print(f"truncated_power_law alpha = {alpha_truncated}")
    print(f"truncated_power_law lambda = {lambda_truncated}")
    print(f"power_law vs truncated_power_law: R = {likelihood_ratio}, p = {p_value}")

    empirical_pdf = counts / counts.sum()

    plt.figure(figsize=(7, 5))

    plt.scatter(values, empirical_pdf, marker="o", label="Empirical data")

    truncated_power_law.plot_pdf(
        linestyle="--",
        linewidth=2,
        label=fr"Truncated power law: $\alpha={alpha_truncated:.3f}$, $\lambda={lambda_truncated:.3g}$",
    )

    pure_power_law.plot_pdf(
        linestyle=":",
        linewidth=2,
        label=fr"Pure power law: $\alpha={alpha_power_law:.3f}$",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Avalanche duration")
    plt.ylabel("Probability")
    plt.title(f"Full-range truncated power-law fit: {file_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    for file_name in FILES_TO_PLOT:
        plot_full_truncated_powerlaw(file_name)


if __name__ == "__main__":
    main()
