# fastaugsynth

`fastaugsynth` is an experimental R clone of [augsynth](https://github.com/ebenmichael/augsynth), with a compiled Julia backend for speed.

This repository was built with heavy AI assistance. The current package should be viewed as a direct AI-assisted Julia/R port of `augsynth`, with a small number of explicit algorithmic changes for speed and validation.

## Scope

Current surface:

- `augsynth()`
- `summary.augsynth()` for jackknife and conformal inference
- `predict.augsynth()`

Not implemented:

- multisynth / staggered-adoption paths
- multi-outcome support

## Installation

```r
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")

remotes::install_github(
  "jamesbrandecon/fastaugsynth"
)

library(fastaugsynth)

if (!fastaugsynth::backend_status()$exists) {
  fastaugsynth::backend_install()
}
```

<!-- For the public repo, `remotes::install_github()` should not require a special PAT. -->

`backend_install()` downloads the prebuilt backend artifact for your platform from this repository's GitHub Actions runs. We have tested Linux, MacOS ARM, and Windows builds/installs. 

If you are testing a non-default branch, install that branch and point backend installation at the same ref:

```r
remotes::install_github("jamesbrandecon/fastaugsynth", ref = "<branch-name>")
Sys.setenv(FASTAUGSYNTH_BACKEND_REF = "<branch-name>")
fastaugsynth::backend_install(force = TRUE)
```

Useful environment variables:

- `FASTAUGSYNTH_BACKEND_LIB`
- `FASTAUGSYNTH_BACKEND_REF`
- `FASTAUGSYNTH_JULIA_THREADS`

If GitHub asks for authentication while downloading the backend artifact, the package will also use these optional auth sources:

- `FASTAUGSYNTH_GITHUB_PAT`
- `GITHUB_PAT`
- `gh auth token`
- `gitcreds`

## Quick Start
The functionality for `fastaugsynth` should be plug and play with any other existing `augsynth` code. As a result, you'll need to prepend the function calls with `fastaugsynth::` or be careful to only load the fast library to avoid namespace conflicts.
```r
fit <- fastaugsynth::augsynth(
  outcome ~ trt,
  unit = unit_id,
  time = period,
  data = panel,
  progfunc = "None",
  scm = TRUE,
  t_int = treatment_start
)

fastaugsynth::summary(fit, inf = TRUE, inf_type = "jackknife")
fastaugsynth::summary(fit, inf = TRUE, inf_type = "conformal")

# closer-to-upstream validation mode
fastaugsynth::summary(
  fit,
  inf = TRUE,
  inf_type = "conformal",
  type = "block",
  conformal_mode = "reference"
)
```

Increasing the number of Julia threads used will increase speed further in some cases. Embedded Julia thread count can be controlled with:

```r
Sys.setenv(FASTAUGSYNTH_JULIA_THREADS = "auto")
fastaugsynth:::backend_thread_count()
```

## Benchmarks

The package story is simplest in pictures. This donor-sweep benchmark fixes `180` pre-treatment periods and `90` post-treatment periods and compares `fastaugsynth` against upstream `augsynth` on estimation and jackknife inference:

![Donor sweep benchmark](docs/figures/fastaugsynth_donor_sweep.png)

This figure covers estimation and jackknife only; it does not include conformal inference.

On that donor sweep, the jackknife inference call currently ranges from about `4.2 ms` to `10.9 ms` for `fastaugsynth` versus about `81.7 ms` to `347.3 ms` for upstream `augsynth`.

A separate conformal-only comparison benchmarks direct `summary.augsynth(..., inf_type = "conformal")` calls on three simulated panels:

![Conformal benchmark](docs/figures/fastaugsynth_conformal_compare.png)

On those three cases, `fastaugsynth` currently takes about `6 ms`, `82 ms`, and `360 ms`, versus about `8.78 s`, `23.7 s`, and `118.4 s` for upstream `augsynth`.

Within `fastaugsynth` on that larger case, the isolated joint multi-post `iid` kernel is about `48.4 ms` on `1` Julia thread and about `12.8 ms` on `6` Julia threads.

Reproduction scripts live in [inst/benchmarks/README.md](inst/benchmarks/README.md).

## Agreement With Upstream

Default conformal outputs are not expected to match exactly. In `fast` mode, `fastaugsynth` uses exact single-post `iid` conformal p-values and an adaptive CI search, while upstream `augsynth` uses Monte Carlo `iid` p-values and a fixed grid search. Those choices can move pointwise p-values and interval endpoints slightly, especially under `type = "iid"`.

The validation panel below therefore uses `type = "block"` and `conformal_mode = "reference"` to remove the `iid` Monte Carlo noise and align the conformal search path with upstream.

This 3x3 validation panel compares `fastaugsynth::augsynth()` and upstream `augsynth::augsynth()` across 9 simulated datasets using:

- `type = "block"`
- `conformal_mode = "reference"`

![Agreement plot](docs/figures/fastaugsynth_agreement_grid.png)

For that run:

- max synthetic-path difference: `1.82e-6`
- max ATT-path difference: `1.82e-6`
- max counterfactual CI lower-bound difference: `1.33e-6`
- max counterfactual CI upper-bound difference: `3.77e-6`

The figure inputs and diagnostics are checked into [docs/benchmarks/agreement/](docs/benchmarks/agreement/).

## Why It Is Faster

Current implementation differences that matter for runtime:

1. Single-post `iid` conformal tests use the exact permutation distribution instead of Monte Carlo approximation.
2. SCM/QP subproblems are cached and warm-started in Julia rather than repeatedly rebuilt from R.
3. The fast conformal path can use adaptive CI search.
4. The whole hot inference loop stays in compiled code.
5. Multi-post `iid` work can use Julia threading.

## Relationship To Upstream `augsynth`

The package intentionally keeps the familiar upstream function names and general API shape:

- `augsynth()`
- `summary.augsynth()`
- `predict.augsynth()`

But it is not the upstream package. It is best understood as:

- a separate implementation
- heavily AI-assisted
- validated against upstream
- still experimental

If you care most about close behavioral matching rather than maximum speed, use validation settings like `type = "block"` and `conformal_mode = "reference"`.

## Repository Layout

- [backend/](backend): Julia backend and PackageCompiler build
- [R/](R), [src/](src), [tests/](tests), and [inst/](inst): R package source for `fastaugsynth`
- [inst/benchmarks/](inst/benchmarks): benchmark runners and benchmark notes
- [docs/benchmarks/agreement/](docs/benchmarks/agreement/): agreement figure inputs and diagnostics

## CI

The GitHub Actions workflow builds the backend artifact and checks that the R package can install and run against that artifact without requiring Julia on the runtime worker.
