# `jols_fit_xy()`, `lm()`, and `feols()` Benchmarks

This folder contains a standalone benchmark runner for comparing the compiled Julia-backed OLS entrypoint `jols_fit_xy()` against both base R's formula interface `lm()` and `fixest::feols()`.

## What it does

- benchmarks four fixed `n x p` scenarios with `p = 8`
- validates that `jols_fit_xy()`, `lm()`, and `feols()` return matching coefficients before recording timings
- writes raw timing data and summarized medians to `results/`
- writes a grouped bar chart and a scaling line chart to `figures/`

## Caveat

This benchmark compares:

- `jols_fit_xy(X, y)`, a low-level matrix API
- `lm(y ~ ., data = df)`, the full formula/data-frame API
- `fixest::feols(y ~ ., data = df)`, a high-performance econometrics-focused formula API

That makes this useful as an end-user latency comparison, not a solver-only comparison. `feols()` is a stronger baseline than `lm()`, and `lm.fit()` would still be the cleaner low-level reference if you want to isolate only the numerical kernel.

## Run It

Install the source tree under `statlibR/` as package `metricsjl`, point `METRICSJL_BACKEND_LIB` at a built backend library, and run:

```bash
R CMD INSTALL -l /tmp/metricsjl-lib statlibR
R_LIBS=/tmp/metricsjl-lib \
METRICSJL_BACKEND_LIB=/absolute/path/to/libstatlibbackend.dylib \
Rscript statlibR/inst/benchmarks/benchmark_ols_vs_lm.R \
  --output-dir statlibR/inst/benchmarks \
  --reps 12 \
  --batch-size 5
```

The script also requires `fixest`.

On Linux, swap the library filename to `libstatlibbackend.so`.

## Outputs

- `results/ols_vs_lm_timings.csv`: raw per-run timings for all three methods
- `results/ols_vs_lm_summary.csv`: cold and warm timing summaries
- `results/ols_vs_lm_speedup.csv`: warm median timing table and ratios against `jols_fit_xy()`
- `results/benchmark_metadata.txt`: backend path and runtime metadata
- `figures/ols_vs_lm_bar.png`: grouped warm-runtime bar chart
- `figures/ols_vs_lm_line.png`: warm-runtime scaling line chart

## `augsynth`, `metricsjl`, and `gsynth` Benchmark

This folder also supports a synthetic-control benchmark that generates its own simulated panel data and can sweep one dimension at a time (donors, pre-treatment periods, or post-treatment periods).

- `metricsjl::augsynth()` (compiled backend)
- `augsynth::augsynth()` (reference R implementation)
- `gsynth::gsynth()` (if available; optional)

Run:

```bash
R CMD INSTALL -l /tmp/metricsjl-lib statlibR
R_LIBS=/tmp/metricsjl-lib \
METRICSJL_BACKEND_LIB=/absolute/path/to/libstatlibbackend.dylib \
Rscript statlibR/inst/benchmarks/benchmark_augsynth_vs_gsynth.R \
  --output-dir statlibR/inst/benchmarks \
  --figures-dir /tmp/metricsjl-bench-figs \
  --reps 20
```

Use `--skip-gsynth` if `gsynth` is not installed.

To include `gsynth`, install it once first:

```r
install.packages("gsynth")
```

### Sweep examples

```bash
Rscript statlibR/inst/benchmarks/benchmark_augsynth_vs_gsynth.R \
  --figures-dir /tmp/bench-donors \
  --sweep donors \
  --sweep-values 10,20,40 \
  --pre-periods 12 \
  --post-periods 6 \
  --reps 20
```

```bash
Rscript statlibR/inst/benchmarks/benchmark_augsynth_vs_gsynth.R \
  --figures-dir /tmp/bench-pre \
  --sweep pre \
  --sweep-values 8,12,16 \
  --donors 30 \
  --post-periods 6 \
  --reps 20
```

```bash
Rscript statlibR/inst/benchmarks/benchmark_augsynth_vs_gsynth.R \
  --figures-dir /tmp/bench-post \
  --sweep post \
  --sweep-values 4,8,12 \
  --donors 30 \
  --pre-periods 12 \
  --reps 20
```

### Inference options

Inference timing is included by default for `jackknife` and `conformal`.

```bash
Rscript statlibR/inst/benchmarks/benchmark_augsynth_vs_gsynth.R \
  --sweep donors \
  --sweep-values 20,40,60 \
  --inference jackknife,conformal \
  --reps 20
```

- `--inference jackknife,conformal` (or `--inference both`): both inference modes
- `--inference jackknife`: jackknife only
- `--inference conformal`: conformal only
- `--skip-inference`: estimation only

In both cases, benchmarking uses `summary(fit, inf = TRUE, inf_type = "<mode>")` on
`metricsjl::augsynth()` and `augsynth::augsynth()`.

## Factorial `augsynth` Benchmark

For a broader and easier-to-review comparison, use `benchmark_augsynth_factorial.R`.

This script is deliberately simpler than the sweep runner:

- one explicit `2 x 2 x 2 x 2` grid
- factors: donors, pre-periods, post-periods, and noise level
- times estimation, jackknife, and conformal separately
- writes one CSV per output table plus one grouped bar chart per phase

The fourth factor is `noise_sd` because conformal runtime can move when intervals get wider, so it is useful to vary one "easy vs hard inference" dimension in addition to the raw panel sizes.

Current default grid:

- donors: `10`, `80`
- pre-periods: `20`, `120`
- post-periods: `20`, `50`
- noise sd: `0.5`, `1.5`

Review the exact 16 specs without running the benchmark:

```bash
Rscript statlibR/inst/benchmarks/benchmark_augsynth_factorial.R --print-specs
```

Run the full benchmark:

```bash
R CMD INSTALL -l /tmp/metricsjl-lib statlibR
R_LIBS=/tmp/metricsjl-lib \
METRICSJL_BACKEND_LIB=/absolute/path/to/libstatlibbackend.dylib \
Rscript statlibR/inst/benchmarks/benchmark_augsynth_factorial.R \
  --output-dir statlibR/inst/benchmarks \
  --reps 3
```

To include the patched-R benchmark clone as a third conformal method, install the
local temp package and pass its package name:

```bash
R CMD INSTALL -l /tmp/metricsjl-lib tmp/augsynth-upstream-fast
R_LIBS=/tmp/metricsjl-lib \
METRICSJL_BACKEND_LIB=/absolute/path/to/libstatlibbackend.dylib \
Rscript statlibR/inst/benchmarks/benchmark_augsynth_factorial.R \
  --output-dir statlibR/inst/benchmarks \
  --reps 3 \
  --fast-r-package augsynthfast
```

This patched clone only changes the pointwise `iid` conformal path. It is faster
because it replaces the Monte Carlo single-post approximation with the exact
single-post permutation distribution, so it should not be expected to match
upstream `iid` pointwise intervals exactly.

Outputs:

- `results_factorial/augsynth_factorial_specs.csv`
- `results_factorial/augsynth_factorial_timings.csv`
- `results_factorial/augsynth_factorial_summary.csv`
- `results_factorial/augsynth_factorial_diagnostics.csv`
- `figures_factorial/augsynth_factorial_estimate_bar.png`
- `figures_factorial/augsynth_factorial_jackknife_bar.png`
- `figures_factorial/augsynth_factorial_conformal_bar.png`

## `augsynth` Agreement Plot

Use `plot_augsynth_agreement.R` to visually compare `metricsjl::augsynth()` and upstream `augsynth::augsynth()` on a small battery of simulated datasets.

The script:

- simulates 9 datasets with varying donor counts, pre-period counts, post-period counts, and noise levels
- fits both `metricsjl` and upstream `augsynth`
- computes conformal intervals with deterministic `type = "block"` and `conformal_mode = "reference"` by default
- writes one 3 x 3 panel figure with treated outcome, synthetic path, conformal interval, and treatment-date vline
- writes a diagnostics CSV with max absolute differences in synthetic paths and CI bounds so any remaining inference-gap is explicit

Review the spec table:

```bash
Rscript statlibR/inst/benchmarks/plot_augsynth_agreement.R --print-specs
```

Run it:

```bash
R CMD INSTALL -l /tmp/metricsjl-lib statlibR
R_LIBS=/tmp/metricsjl-lib \
METRICSJL_BACKEND_LIB=/absolute/path/to/libstatlibbackend.dylib \
Rscript statlibR/inst/benchmarks/plot_augsynth_agreement.R \
  --output-dir statlibR/inst/benchmarks
```

Outputs:

- `results_agreement/augsynth_agreement_specs.csv`
- `results_agreement/augsynth_agreement_diagnostics.csv`
- `figures_agreement/augsynth_agreement_grid.png`

## `augsynth` vs `gsynth` Outputs

- `results/augsynth_vs_gsynth_timings.csv`: per-iteration raw timings in milliseconds
- `results/augsynth_vs_gsynth_summary.csv`: averaged summaries and speedup against `metricsjl`
- `results/augsynth_vs_gsynth_speedup.csv`: warm-median speedup table
- `results/benchmark_metadata.txt`: run metadata, package versions, and backend path
- `figures/augsynth_vs_gsynth_<sweep>_bar.png`: grouped bar chart for selected sweep dimension

## Current Warm-Median Results

The exact raw numbers are in `results/ols_vs_lm_speedup.csv` and `results/ols_vs_lm_summary.csv`.

| Scenario | `jols_fit_xy()` | `lm()` | `feols()` | `lm() / jols_fit_xy()` | `feols() / jols_fit_xy()` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `n=500, p=8` | `0.20 ms` | `0.80 ms` | `3.30 ms` | `4.00x` | `16.50x` |
| `n=1000, p=8` | `0.20 ms` | `0.80 ms` | `3.30 ms` | `4.00x` | `16.50x` |
| `n=5000, p=8` | `0.60 ms` | `1.60 ms` | `4.00 ms` | `2.67x` | `6.67x` |
| `n=10000, p=8` | `0.80 ms` | `2.90 ms` | `4.80 ms` | `3.62x` | `6.00x` |

On this machine, `feols()` is slower for these cases because the workload is just plain dense OLS with no fixed effects, so its richer formula/econometrics machinery does more setup than the low-level matrix call and even more than `lm()`. That does not imply `feols()` is weak overall; it means this specific benchmark is stressing a narrower path than the one `fixest` is optimized to dominate.

### Woodbury note

Woodbury-style updates are typically most useful when the donor matrix is strongly rectangular (many more pre-periods than donors). As the problem becomes more square, the speed advantage tends to shrink, so a future pass could switch strategies by problem shape.

![Warm runtime bar chart](figures/ols_vs_lm_bar.png)

![Warm runtime scaling line chart](figures/ols_vs_lm_line.png)
