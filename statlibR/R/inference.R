# Single-period inference helpers adapted from upstream augsynth.

drop_time_t <- function(wide_data, Z, t_drop) {
  new_wide_data <- list()
  new_wide_data$trt <- wide_data$trt
  new_wide_data$X <- wide_data$X[, -t_drop, drop = FALSE]
  new_wide_data$y <- cbind(wide_data$X[, t_drop, drop = FALSE], wide_data$y)
  new_wide_data$Z <- Z

  X0 <- new_wide_data$X[new_wide_data$trt == 0, , drop = FALSE]
  x1 <- matrix(colMeans(new_wide_data$X[new_wide_data$trt == 1, , drop = FALSE]), ncol = 1)

  new_synth_data <- list(
    Z0 = t(X0),
    X0 = t(X0),
    Z1 = x1,
    X1 = x1
  )

  list(wide = new_wide_data, synth_data = new_synth_data, Z = Z)
}

time_jackknife_plus <- function(ascm, alpha = 0.05, conservative = FALSE) {
  wide_data <- ascm$data
  synth_data <- ascm$data$synth_data
  Z <- wide_data$Z

  t0 <- nrow(synth_data$Z0)
  tpost <- ncol(wide_data$y)
  t_final <- nrow(synth_data$Y0plot)

  jack_ests <- lapply(seq_len(t0), function(tdrop) {
    new_data <- drop_time_t(wide_data, Z, tdrop)
    new_ascm <- do.call(
      fit_augsynth_internal,
      c(
        list(
          wide = new_data$wide,
          synth_data = new_data$synth_data,
          Z = new_data$Z,
          progfunc = ascm$progfunc,
          scm = ascm$scm,
          fixedeff = ascm$fixedeff
        ),
        ascm$extra_args
      )
    )
    est <- predict(new_ascm, att = FALSE)[(t0 + 1):t_final]
    est <- c(est, mean(est))
    err <- c(
      colMeans(wide_data$X[wide_data$trt == 1, tdrop, drop = FALSE]) -
        predict(new_ascm, att = FALSE)[t0]
    )
    list(err, rbind(est + abs(err), est - abs(err), est + err, est))
  })

  held_out_errs <- vapply(jack_ests, `[[`, numeric(1), 1)
  jack_dist <- vapply(jack_ests, `[[`, matrix(0, nrow = 4, ncol = tpost + 1), 2)

  out <- list()
  att <- predict(ascm, att = TRUE)
  out$att <- c(att, mean(att[(t0 + 1):t_final]))
  out$heldout_att <- c(held_out_errs, att[(t0 + 1):t_final], mean(att[(t0 + 1):t_final]))

  if (conservative) {
    qerr <- stats::quantile(abs(held_out_errs), 1 - alpha)
    out$lb <- c(rep(NA_real_, t0), apply(jack_dist[4, , ], 1, min) - qerr)
    out$ub <- c(rep(NA_real_, t0), apply(jack_dist[4, , ], 1, max) + qerr)
  } else {
    out$lb <- c(rep(NA_real_, t0), apply(jack_dist[2, , ], 1, stats::quantile, alpha / 2))
    out$ub <- c(rep(NA_real_, t0), apply(jack_dist[1, , ], 1, stats::quantile, 1 - alpha / 2))
  }

  y1 <- predict(ascm, att = FALSE) + att
  y1 <- c(y1, mean(y1[(t0 + 1):t_final]))
  shifted_lb <- y1 - out$ub
  shifted_ub <- y1 - out$lb
  out$lb <- shifted_lb
  out$ub <- shifted_ub
  out$alpha <- alpha
  out
}

compute_permute_test_stats <- function(wide_data, ascm, h0,
                                       post_length, type,
                                       q, ns, stat_func) {
  new_wide_data <- wide_data
  t0 <- ncol(wide_data$X) - post_length
  tpost <- t0 + post_length

  treat_rows <- wide_data$trt == 1
  new_wide_data$X[treat_rows, (t0 + 1):tpost] <-
    sweep(new_wide_data$X[treat_rows, (t0 + 1):tpost, drop = FALSE], 2, h0, "-")

  X0 <- new_wide_data$X[new_wide_data$trt == 0, , drop = FALSE]
  x1 <- matrix(colMeans(new_wide_data$X[new_wide_data$trt == 1, , drop = FALSE]), ncol = 1)

  new_synth_data <- list(
    Z0 = t(X0),
    X0 = t(X0),
    Z1 = x1,
    X1 = x1
  )

  new_ascm <- do.call(
    fit_augsynth_internal,
    c(
      list(
        wide = new_wide_data,
        synth_data = new_synth_data,
        Z = wide_data$Z,
        progfunc = ascm$progfunc,
        scm = ascm$scm,
        fixedeff = ascm$fixedeff
      ),
      ascm$extra_args
    )
  )
  resids <- predict(new_ascm, att = TRUE)[1:tpost]

  if (is.null(stat_func)) {
    stat_func <- function(x) (sum(abs(x)^q) / sqrt(length(x)))^(1 / q)
  }

  if (type == "iid") {
    test_stats <- sapply(seq_len(ns), function(x) {
      reorder <- sample(resids)
      stat_func(reorder[(t0 + 1):tpost])
    })
  } else {
    test_stats <- sapply(seq_len(tpost), function(j) {
      reorder <- resids[((0:(tpost - 1)) + j) %% tpost + 1]
      stat_func(reorder[(t0 + 1):tpost])
    })
  }

  list(resids = resids, test_stats = test_stats, stat_func = stat_func)
}

compute_permute_pval <- function(wide_data, ascm, h0,
                                 post_length, type,
                                 q, ns, stat_func) {
  t0 <- ncol(wide_data$X) - post_length
  tpost <- t0 + post_length
  out <- compute_permute_test_stats(wide_data, ascm, h0, post_length, type, q, ns, stat_func)
  mean(out$stat_func(out$resids[(t0 + 1):tpost]) <= out$test_stats)
}

compute_permute_ci <- function(wide_data, ascm, grid,
                               post_length, alpha, type,
                               q, ns, stat_func) {
  grid <- c(grid, 0)
  ps <- sapply(grid, function(x) {
    compute_permute_pval(wide_data, ascm, x, post_length, type, q, ns, stat_func)
  })
  c(min(grid[ps >= alpha]), max(grid[ps >= alpha]), ps[grid == 0][1])
}

compute_permute_ci_linear <- function(wide_data, ascm, grid_int, grid_slope,
                                      post_length, alpha, type,
                                      q, ns, stat_func) {
  grid_comb <- expand.grid(grid_int, grid_slope)
  grid_comb$p_val <- apply(grid_comb, 1, function(x) {
    compute_permute_pval(
      wide_data,
      ascm,
      x[1] + x[2] * seq_len(post_length),
      post_length,
      type,
      q,
      ns,
      stat_func
    )
  })
  ci_int <- c(min(grid_comb[grid_comb$p_val >= alpha, 1]), max(grid_comb[grid_comb$p_val >= alpha, 1]))
  ci_slope <- c(min(grid_comb[grid_comb$p_val >= alpha, 2]), max(grid_comb[grid_comb$p_val >= alpha, 2]))
  int_slope_est <- as.numeric(grid_comb[which.max(grid_comb$p_val), 1:2])
  list(est_int = int_slope_est[1], ci_int = ci_int, est_slope = int_slope_est[2], ci_slope = ci_slope)
}

conformal_inf <- function(ascm, alpha = 0.05,
                          stat_func = NULL, type = "iid",
                          q = 1, ns = 1000, grid_size = 50) {
  wide_data <- ascm$data
  synth_data <- ascm$data$synth_data

  t0 <- nrow(synth_data$Z0)
  tpost <- ncol(wide_data$y)
  t_final <- nrow(synth_data$Y0plot)

  att <- predict(ascm, att = TRUE)
  post_att <- att[(t0 + 1):t_final]
  post_sd <- sqrt(mean(post_att^2))

  cis <- vapply(seq_len(tpost), function(j) {
    new_wide_data <- wide_data
    new_wide_data$X <- cbind(wide_data$X, wide_data$y[, j, drop = TRUE])
    if (tpost > 1) {
      new_wide_data$y <- wide_data$y[, -j, drop = FALSE]
    } else {
      new_wide_data$y <- matrix(1, nrow = nrow(wide_data$X), ncol = 1)
    }
    grid <- seq(att[t0 + j] - 2 * post_sd, att[t0 + j] + 2 * post_sd, length.out = grid_size)
    compute_permute_ci(new_wide_data, ascm, grid, 1, alpha, type, q, ns, stat_func)
  }, numeric(3))

  new_wide_data <- wide_data
  new_wide_data$X <- cbind(wide_data$X, wide_data$y)
  new_wide_data$y <- matrix(1, nrow = nrow(wide_data$X), ncol = 1)
  null_p <- compute_permute_pval(new_wide_data, ascm, 0, ncol(wide_data$y), type, q, ns, stat_func)

  out <- list()
  out$att <- c(att, mean(att[(t0 + 1):t_final]))
  out$lb <- c(rep(NA_real_, t0), cis[1, ], NA_real_)
  out$ub <- c(rep(NA_real_, t0), cis[2, ], NA_real_)
  out$p_val <- c(rep(NA_real_, t0), cis[3, ], null_p)
  out$alpha <- alpha
  out
}

conformal_inf_linear <- function(ascm, alpha = 0.05,
                                 stat_func = NULL, type = "iid",
                                 q = 1, ns = 1000, grid_size = 50) {
  wide_data <- ascm$data
  synth_data <- ascm$data$synth_data

  t0 <- nrow(synth_data$Z0)
  tpost <- ncol(wide_data$y)
  t_final <- nrow(synth_data$Y0plot)

  att <- predict(ascm, att = TRUE)
  post_att <- att[(t0 + 1):t_final]
  post_second <- sqrt(mean(post_att^2))

  ts <- seq_len(tpost)
  lm_out <- summary(stats::lm(post_att ~ ts))$coefficients
  grid_int <- seq(lm_out[1, 1] - 2 * post_second, lm_out[1, 1] + 2 * post_second, length.out = grid_size)

  if (tpost == 2) {
    warning("There are 2 post-treatment time periods, so a linear model has a perfect fit. A confidence interval for the slope may not be reasonable here.", call. = FALSE)
    grid_slope <- seq(lm_out[2, 1] - abs(lm_out[2, 1]), lm_out[2, 1] + abs(lm_out[2, 1]), length.out = grid_size)
  } else if (tpost <= 1) {
    stop("There is only one post-treatment time period, so an intercept and a slope cannot be computed.", call. = FALSE)
  } else {
    grid_slope <- seq(
      lm_out[2, 1] - 4 * lm_out[2, 2] * sqrt(tpost),
      lm_out[2, 1] + 4 * lm_out[2, 2] * sqrt(tpost),
      length.out = grid_size
    )
  }

  new_wide_data <- wide_data
  new_wide_data$X <- cbind(wide_data$X, wide_data$y)
  new_wide_data$y <- matrix(1, nrow = nrow(wide_data$X), ncol = 1)

  compute_permute_ci_linear(new_wide_data, ascm, grid_int, grid_slope, ncol(wide_data$y), alpha, type, q, ns, stat_func)
}

drop_unit_i <- function(wide_data, Z, i) {
  new_wide_data <- list(
    trt = wide_data$trt[-i],
    X = wide_data$X[-i, , drop = FALSE],
    y = wide_data$y[-i, , drop = FALSE],
    Z = if (!is.null(Z)) Z[-i, , drop = FALSE] else NULL
  )

  X0 <- new_wide_data$X[new_wide_data$trt == 0, , drop = FALSE]
  x1 <- matrix(colMeans(new_wide_data$X[new_wide_data$trt == 1, , drop = FALSE]), ncol = 1)

  new_synth_data <- list(
    Z0 = t(X0),
    X0 = t(X0),
    Z1 = x1,
    X1 = x1
  )

  list(wide = new_wide_data, synth_data = new_synth_data, Z = new_wide_data$Z)
}

jackknife_se_single <- function(ascm) {
  wide_data <- ascm$data
  synth_data <- ascm$data$synth_data
  n <- nrow(wide_data$X)
  Z <- wide_data$Z

  t0 <- nrow(synth_data$Z0)
  tpost <- ncol(wide_data$y)
  t_final <- nrow(synth_data$Y0plot)

  nnz_weights <- numeric(n)
  nnz_weights[wide_data$trt == 0] <- round(ascm$weights, 3) != 0
  if (sum(wide_data$trt) > 1) {
    nnz_weights[wide_data$trt == 1] <- 1
  }
  trt_idxs <- seq_len(n)[as.logical(nnz_weights)]

  ests <- vapply(trt_idxs, function(i) {
    new_data <- drop_unit_i(wide_data, Z, i)
    new_ascm <- do.call(
      fit_augsynth_internal,
      c(
        list(
          wide = new_data$wide,
          synth_data = new_data$synth_data,
          Z = new_data$Z,
          progfunc = ascm$progfunc,
          scm = ascm$scm,
          fixedeff = ascm$fixedeff
        ),
        ascm$extra_args
      )
    )
    est <- predict(new_ascm, att = TRUE)[(t0 + 1):t_final]
    c(est, mean(est))
  }, numeric(tpost + 1))

  ests <- matrix(ests, nrow = tpost + 1, ncol = length(trt_idxs))
  se2 <- apply(ests, 1, function(x) (n - 1) / n * sum((x - mean(x, na.rm = TRUE))^2))
  se <- sqrt(se2)

  out <- list()
  att <- predict(ascm, att = TRUE)
  out$att <- c(att, mean(att[(t0 + 1):t_final]))
  out$se <- c(rep(NA_real_, t0), se)
  out
}
