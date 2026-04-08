# jlrstats

Prototype workspace for an R statistical library with a compiled Julia backend.

## Phase 1 implemented prototype

This repository now includes a minimal end-to-end Phase 1 architecture:

- `backend/`: Julia backend with C-callable `fit_ols_dense` (OLS) and `fit_ridge_loocv_dense` (ridge + LOOCV tuning) ABIs.
- `statlibR/`: source tree for the `metricsjl` R package, with a thin C shim and low-level `jols_fit_xy(X, y)` plus `jridge_fit_xy(X, y, lambdas)` entrypoints.
- `.github/workflows/phase1-no-julia-runtime.yml`: CI pipeline that:
  1. builds the Julia shared library artifact in one job, and then
  2. validates R package installation and OLS fit correctness in a separate job **without Julia installed**.

## Quick layout

- Julia backend source: `backend/src/StatlibBackend.jl`
- Backend build scripts: `backend/build/build_backend.jl`, `backend/build/package_backend.sh`
- R package interface: `statlibR/R/jols_fit_xy.R` for installed package `metricsjl`
- R-to-backend C bridge: `statlibR/src/jols_bridge.c`
- R tests: `statlibR/tests/testthat/`

## CI guarantee for "machine without Julia"

The `verify-without-julia` job in the GitHub Actions workflow explicitly checks that `julia` is absent and then runs the R package against the prebuilt backend artifact.

## Planning

- Phase 1 architecture roadmap: [`docs/phase1-julia-backend-roadmap.md`](docs/phase1-julia-backend-roadmap.md)

## Triggering CI

The `phase1-no-julia-runtime` workflow triggers automatically on:

- any push to any branch
- any pull request event (`opened`, `synchronize`, `reopened`, `ready_for_review`) to any branch
- manual runs via **Actions → phase1-no-julia-runtime → Run workflow**
