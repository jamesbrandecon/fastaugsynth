# Upstream `augsynth` Minimal Patch for Exact Single-Post `iid` Conformal

## Bottom line

Yes, there is still room to speed things up, but the next large win is narrow:

- The biggest cheap remaining win in the original R package is to stop using Monte Carlo resampling for the pointwise conformal CI subproblems where `post_length = 1` and `type = "iid"`.
- This does **not** solve the joint post-treatment null or linear-effect conformal path, because those are still multi-period problems with `post_length > 1`.
- For the standard pointwise conformal CIs, this is the main trick that produced the very large speedup.

If the goal is a minimal upstream patch, the good news is that this can be done entirely inside `R/inference.R` with no public API changes.

## Why the shortcut is valid

In upstream `augsynth`, conformal inference works by:

1. imposing a null effect `h0`,
2. refitting the model,
3. computing residuals over the pre-plus-post window,
4. building a null distribution by permuting those residuals.

For the pointwise CI problems in `conformal_inf()`, the code calls:

- `compute_permute_ci(..., post_length = 1, ...)`

So under `type = "iid"`, each random permutation only decides which **one** residual lands in the single post slot. That means the exact permutation distribution is just the set of test statistics obtained by putting each residual in that slot once.

For the default statistic, this collapses to:

```r
abs(resids)
```

More generally, for an arbitrary `stat_func`, the exact single-post distribution is:

```r
vapply(seq_along(resids), function(i) stat_func(resids[i]), numeric(1))
```

So the `ns` Monte Carlo loop is unnecessary for `post_length = 1`.

## Upstream locations to patch

The relevant upstream code in the cloned copy is:

- [tmp/augsynth-upstream/R/inference.R](/Users/jamesbrand/Dropbox/git_stuff/jlrstats/tmp/augsynth-upstream/R/inference.R#L137)
- [tmp/augsynth-upstream/R/inference.R](/Users/jamesbrand/Dropbox/git_stuff/jlrstats/tmp/augsynth-upstream/R/inference.R#L284)
- [tmp/augsynth-upstream/R/inference.R](/Users/jamesbrand/Dropbox/git_stuff/jlrstats/tmp/augsynth-upstream/R/inference.R#L350)

The pointwise CI loop is in `conformal_inf()`, but the minimal patch should go lower down in `compute_permute_test_stats()` so every single-post `iid` caller benefits automatically.

## Minimal code change

### Patch 1: add an exact single-post helper

Add this near the permutation helpers in `R/inference.R`:

```r
single_post_iid_test_stats <- function(resids, stat_func) {
  vapply(
    seq_along(resids),
    function(i) stat_func(resids[i]),
    numeric(1)
  )
}
```

### Patch 2: short-circuit the `iid` branch in `compute_permute_test_stats()`

Current upstream logic:

```r
if(type == "iid") {
  test_stats <- sapply(1:ns,
                      function(x) {
                        reorder <- sample(resids)
                        stat_func(reorder[(t0 + 1):tpost])
                      })
} else {
  ...
}
```

Replace it with:

```r
if(type == "iid") {
  if(post_length == 1) {
    test_stats <- single_post_iid_test_stats(resids, stat_func)
  } else {
    test_stats <- sapply(
      1:ns,
      function(x) {
        reorder <- sample(resids)
        stat_func(reorder[(t0 + 1):tpost])
      }
    )
  }
} else {
  test_stats <- sapply(
    1:tpost,
    function(j) {
      reorder <- resids[(0:tpost - 1 + j) %% tpost + 1]
      stat_func(reorder[(t0 + 1):tpost])
    }
  )
}
```

That is the smallest patch that changes behavior only where it should.

## Why this is enough

No other upstream functions need to change:

- `conformal_inf()` already calls `compute_permute_ci(..., post_length = 1, ...)` for each post period.
- `compute_permute_ci()` already calls `compute_permute_pval()`.
- `compute_permute_pval()` already calls `compute_permute_test_stats()`.

So once `compute_permute_test_stats()` uses the exact distribution for the single-post `iid` case, the pointwise conformal CIs inherit the speedup automatically.

This means:

- no new user-facing argument,
- no new summary method,
- no roxygen/API change required,
- no change to `block`,
- no change to the multi-period joint null test.

## What does **not** get faster

This patch does **not** speed up:

- the joint post-treatment null at the end of `conformal_inf()`, because that uses `post_length = ncol(wide_data$y)`,
- `conformal_inf_linear()`, which is inherently multi-period,
- multi-outcome conformal code,
- any `block` permutation path.

So if the question is "what remains expensive after this patch?", the answer is:

- `iid` conformal with `post_length > 1`

That is the remaining hard case.

## Expected impact

For standard single-period `augsynth::summary(..., inf_type = "conformal", type = "iid")`:

- pointwise CI work should drop dramatically,
- runtime should stop depending much on `ns` for those pointwise intervals,
- results should become deterministic for those pointwise p-values and bounds.

The total runtime will still include the final joint null test, so the whole call will not become "free", but this is still the highest-value minimal patch.

## Tests to add upstream

Upstream currently does not appear to have dedicated conformal inference tests in `tests/testthat`, so I would add a new file, for example:

- `tests/testthat/test_conformal_single_post_iid.R`

Minimum tests:

1. Exact single-post p-values no longer depend on `ns`

```r
test_that("single-post iid conformal is exact and independent of ns", {
  # construct a simple one-post-period problem
  # run conformal inference with ns = 16 and ns = 1024
  # expect identical pointwise p-values / intervals
})
```

2. Exact single-post shortcut matches brute-force enumeration

```r
test_that("single-post iid shortcut matches exact permutation distribution", {
  resids <- c(-2, -1, 0.5, 3)
  stat_func <- function(x) abs(x)

  exact <- single_post_iid_test_stats(resids, stat_func)
  brute <- sapply(seq_along(resids), function(i) stat_func(resids[i]))

  expect_equal(exact, brute)
})
```

3. Multi-period `iid` path is unchanged

```r
test_that("multi-period iid conformal still uses Monte Carlo path", {
  # use a problem with at least two post periods
  # verify function still runs and returns finite outputs
})
```

## If we wanted one more step beyond the minimal patch

The next-smallest upstream improvement would be to special-case the single-post default statistic even more aggressively:

```r
if(type == "iid" && post_length == 1 && is.null(stat_func)) {
  test_stats <- abs(resids)
}
```

That is slightly faster than the generic helper, but I would not start there. The generic helper above is cleaner and preserves compatibility with custom `stat_func`.

## Recommendation

If the goal is "smallest upstream patch with the biggest payoff", do exactly this:

1. add `single_post_iid_test_stats()`,
2. branch on `type == "iid" && post_length == 1` inside `compute_permute_test_stats()`,
3. add one focused `testthat` file covering exactness and `ns` invariance.

That captures the main trick without pulling in any Julia-specific machinery or changing upstream user-facing behavior.
