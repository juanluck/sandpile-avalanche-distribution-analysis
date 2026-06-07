# Sandpile Robustness Analysis

Utilities for analyzing avalanche-duration distributions in degraded Bak--Tang--Wiesenfeld (BTW) sandpile simulations on rewired networks.

The repository contains Python scripts used to:

1. compare candidate distributions for avalanche-duration data;
2. summarize the winning distribution across independent runs;
3. generate model-selection figures;
4. fit full-range truncated power laws;
5. regenerate figures directly from generated CSV files.

The data are **included** in this repository but should be substituted by those of your experiments.

## Suggested repository name

Recommended name:

```text
sandpile-robustness-rewiring
```

Other possible names:

```text
btw-sandpile-robustness
sandpile-node-failure-analysis
rewired-sandpile-degradation
soc-sandpile-robustness
```

## Expected data layout

The scripts expect avalanche-duration frequency files with two columns:

```text
duration frequency
```

Expected folder structure:

```text
data/
└── avex_data/
    ├── exp1/
    │   ├── d0_r0.dat
    │   ├── d0_r10.dat
    │   ├── d45_r10.dat
    │   └── ...
    ├── exp2/
    │   └── ...
    └── exp25/
        └── ...
```

Filename convention:

```text
d<degradation>_r<rewiring>.dat
```

## Environment

```bash
conda create -n powerlawfit python=3.12 -y
conda activate powerlawfit
pip install -r requirements.txt
```

If you run into NumPy/Matplotlib binary compatibility issues, use a conservative NumPy version:

```bash
pip install "numpy<2"
```

## Main scripts

### Distribution model selection

```bash
python scripts/analyze_distribution_winners_parallel.py
```

Writes:

```text
distribution_results/all_experiments/best_distributions_all.csv
distribution_results/all_experiments/best_distributions_by_file.csv
```

### Model-selection figures

```bash
python scripts/plot_model_selection_figures.py
```

Generates figures under:

```text
distribution_results/all_experiments/model_selection_figures/
```

### Full-range truncated power-law fits

```bash
python scripts/plot_full_range_truncated_powerlaw_evolution.py
```

Fits:

```text
p(t) ∝ t^(-alpha) exp(-lambda t)
```

over the full observed range and writes:

```text
truncated_powerlaw_full_range_results/truncated_powerlaw_full_range_parameters.csv
truncated_powerlaw_full_range_results/truncated_powerlaw_full_range_summary.csv
truncated_powerlaw_full_range_results/truncated_powerlaw_full_range_alpha_lambda.png
```

The parameters `alpha` and `lambda` should be interpreted as effective phenomenological descriptors, not as universal critical exponents.

### Regenerate truncated-power-law figure from CSV

```bash
python scripts/plot_truncated_powerlaw_from_csv.py
```

### Visual examples

```bash
python scripts/plot_truncated_powerlaw_examples.py
```

## Outputs

Most generated outputs are ignored by Git by default, except `.gitkeep` files.

## License

This repository is distributed under the MIT License.
