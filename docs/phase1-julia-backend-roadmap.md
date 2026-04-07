# Phase 1 Roadmap: R Front End with Compiled Julia Backend

## Objective

Validate a practical architecture where an R package calls a **precompiled Julia backend** (via a thin C shim), so end users can run OLS **without Julia installed locally**.

This phase is a prototype to answer feasibility questions on:

- packaging and binary distribution
- cold vs warm runtime behavior
- copy overhead and boundary design
- developer ergonomics for future expansion

## Architectural decisions

### 1) R is the front end

R is responsible for user-facing concerns:

- data frame and matrix handling
- formula parsing (later phase)
- missing value and factor handling (later phase)
- print/summary methods (later phase)

### 2) Julia is the computation engine

Julia handles numerical work:

- OLS solve and diagnostics
- covariance/variance calculations
- later fixed-effects and higher-dimensional routines

### 3) Compiled library, no embedded Julia REPL

Use a compiled Julia shared library and stable C ABI. Avoid JuliaCall/live Julia session for the core architecture.

### 4) Coarse boundary crossings

Design one call per major operation:

- one call per OLS fit
- one call per vcov computation
- one call per future bootstrap batch

Avoid per-column/per-iteration cross-language calls.

### 5) Narrow ABI

Export a minimal interface with low-level types only:

- matrix/vector pointers
- scalar dimensions/options
- caller-allocated output buffers
- explicit status/error signaling

## Repositories

Use two repositories:

1. **Julia backend** (`statlib-julia-backend`)
   - owns ABI and compiled runtime
   - builds/release artifacts per platform
2. **R package** (`statlibR`)
   - user-facing R API
   - thin C bridge
   - backend download/install/load helpers

## Phase 1 scope

### In scope

- dense OLS with numeric `X` and `y`
- coefficient output (required)
- basic diagnostics and homoskedastic SEs (preferred)
- binary install workflow (at least one platform)

### Out of scope

- CRAN hardening
- formula interface and factor support
- fixed effects
- robust/cluster vcov
- micro-optimizations before measurement

## Phase 1 implementation checklist

1. **Julia backend package**
   - implement `fit_ols_dense` using QR-based solve
   - accept pointers + dimensions + output buffers
2. **Compiled library proof**
   - produce shared library bundle
   - verify from tiny native C test program
   - verify operation without local Julia install
3. **R package C shim**
   - validate dimensions/types
   - pass raw pointers to backend
   - return result list to R
4. **Initial R API**
   - implement low-level `jols_fit_xy(X, y)`
   - return coefficients and basic metadata
5. **Binary distribution workflow**
   - GitHub Actions builds release assets
   - first target: one platform; design for Linux/macOS/Windows
6. **Backend install helpers (R)**
   - `backend_install()` / `backend_status()` / `backend_path()` / `backend_version()`
   - detect OS/arch, download artifact, unpack in cache, verify files
7. **Correctness tests**
   - compare vs `lm.fit` over full-rank and shape variants
   - test intercept and no-intercept cases
8. **Timing tests**
   - report cold and warm timings separately
   - include tiny/medium/large matrix cases
9. **Optional SE support**
   - add residual variance, df, vcov, standard errors in same coarse call
10. **Go/no-go review for Phase 2**
   - evaluate install friction, startup cost, warm performance, complexity, binary size

## Success criteria

Phase 1 succeeds when all are true:

- user fits OLS from R without local Julia install
- backend binary download and load are reliable
- coefficients match `lm.fit` within numerical tolerance
- warm performance justifies continued development
- R/C/Julia boundary remains narrow and understandable

## Minimal milestone demo

1. Install R package from GitHub
2. Run `backend_install()`
3. Call `jols_fit_xy(X, y)`
4. Verify coefficients against `lm.fit`
5. Confirm same workflow on machine without Julia installed

## Performance interpretation

A thin C shim is not expected to be the bottleneck. Expected dominant costs:

- backend startup/initialization (cold path)
- matrix solve and linear algebra
- avoidable copying across boundaries

Primary optimization rule for this phase: **measure first, then optimize**.
