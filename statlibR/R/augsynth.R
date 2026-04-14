# Adapted to provide a single-period augsynth-compatible API over the
# compiled Julia backend in this package.

resolve_column_name <- function(expr, env) {
  if (is.symbol(expr)) {
    name <- as.character(expr)
    value <- tryCatch(get(name, envir = env, inherits = TRUE), error = function(e) NULL)
    if (is.character(value) && length(value) == 1L && nzchar(value)) {
      return(value)
    }
    if (is.name(value) && length(value) == 1L && nzchar(as.character(value))) {
      return(as.character(value))
    }
    return(name)
  }
  value <- eval(expr, env)
  if (is.character(value) && length(value) == 1L && nzchar(value)) {
    return(value)
  }
  stop("Expected a bare column name or scalar character column name", call. = FALSE)
}

formula_rhs1 <- function(form) {
  stats::formula(form, rhs = 1)
}

formula_outcome_expr <- function(form) {
  attr(stats::terms(formula_rhs1(form)), "variables")[[2]]
}

formula_treatment_expr <- function(form) {
  attr(stats::terms(formula_rhs1(form)), "variables")[[3]]
}

is_multi_outcome_formula <- function(form) {
  expr <- formula_outcome_expr(form)
  is.call(expr) && identical(expr[[1]], quote(`+`))
}

eval_formula_expr <- function(expr, data) {
  eval(expr, data, parent.frame())
}

sorted_unique <- function(x) {
  sort(unique(x))
}

matrix_sqrt_psd <- function(V) {
  eig <- eigen((V + t(V)) / 2, symmetric = TRUE)
  vals <- pmax(eig$values, 0)
  eig$vectors %*% (diag(sqrt(vals), nrow = length(vals)) %*% t(eig$vectors))
}

solve_square <- function(A, b = NULL) {
  if (is.null(b)) {
    return(tryCatch(solve(A), error = function(e) qr.solve(A)))
  }
  tryCatch(solve(A, b), error = function(e) qr.solve(A, b))
}

center_by_controls <- function(mat, trt) {
  control_means <- colMeans(mat[trt == 0, , drop = FALSE])
  sweep(mat, 2, control_means, FUN = "-")
}

build_panel_matrix <- function(values, unit_values, time_values, units, times, label) {
  mat <- matrix(
    NA_real_,
    nrow = length(units),
    ncol = length(times),
    dimnames = list(as.character(units), as.character(times))
  )

  keep <- !is.na(values) & time_values %in% times
  unit_index <- match(unit_values[keep], units)
  time_index <- match(time_values[keep], times)
  if (anyNA(unit_index) || anyNA(time_index)) {
    stop(sprintf("Failed to align %s values into a panel matrix", label), call. = FALSE)
  }
  mat[cbind(unit_index, time_index)] <- values[keep]

  if (anyNA(mat)) {
    stop(sprintf("Missing %s values for some unit-time cells", label), call. = FALSE)
  }

  mat
}

infer_treatment_schedule <- function(data, unit_name, time_name, trt_expr) {
  trt_values <- eval_formula_expr(trt_expr, data)
  treated <- !is.na(trt_values) & (trt_values == 1)
  if (!any(treated)) {
    stop("No treated observations found", call. = FALSE)
  }

  first_times <- vapply(
    split(data[[time_name]][treated], data[[unit_name]][treated]),
    min,
    numeric(1)
  )
  sort(unique(first_times))
}

format_data <- function(outcome_expr, trt_expr, unit_name, time_name, t_int, data) {
  unit_values <- data[[unit_name]]
  time_values <- data[[time_name]]
  units <- sorted_unique(unit_values)
  times <- sorted_unique(time_values)

  pre_times <- times[times < t_int]
  post_times <- times[times >= t_int]
  if (length(pre_times) == 0L) {
    stop("No pre-treatment periods found before t_int", call. = FALSE)
  }
  if (length(post_times) == 0L) {
    stop("No post-treatment periods found at or after t_int", call. = FALSE)
  }

  outcome_values <- as.numeric(eval_formula_expr(outcome_expr, data))
  trt_values <- as.numeric(eval_formula_expr(trt_expr, data))

  X <- build_panel_matrix(outcome_values, unit_values, time_values, units, pre_times, "pre-treatment outcome")
  y <- build_panel_matrix(outcome_values, unit_values, time_values, units, post_times, "post-treatment outcome")

  trt_by_unit <- tapply(trt_values, unit_values, function(x) max(x, na.rm = TRUE))
  trt <- as.numeric(trt_by_unit[as.character(units)])

  list(X = X, trt = trt, y = y)
}

format_synth <- function(X, trt, y) {
  synth_data <- list()
  synth_data$Z0 <- t(X[trt == 0, , drop = FALSE])
  synth_data$Z1 <- matrix(colMeans(X[trt == 1, , drop = FALSE]), ncol = 1)
  synth_data$Y0plot <- t(cbind(X[trt == 0, , drop = FALSE], y[trt == 0, , drop = FALSE]))
  synth_data$Y1plot <- matrix(
    colMeans(cbind(X[trt == 1, , drop = FALSE], y[trt == 1, , drop = FALSE])),
    ncol = 1
  )
  synth_data$X0 <- synth_data$Z0
  synth_data$X1 <- synth_data$Z1
  synth_data
}

demean_data <- function(wide_data, synth_data) {
  means <- rowMeans(wide_data$X)
  trt <- wide_data$trt
  new_X <- wide_data$X - means

  new_wide_data <- list(
    X = new_X,
    y = wide_data$y - means,
    trt = trt
  )

  new_synth_data <- list(
    X0 = t(new_X[trt == 0, , drop = FALSE]),
    Z0 = t(new_X[trt == 0, , drop = FALSE]),
    X1 = matrix(colMeans(new_X[trt == 1, , drop = FALSE]), ncol = 1),
    Z1 = matrix(colMeans(new_X[trt == 1, , drop = FALSE]), ncol = 1)
  )

  mhat <- replicate(ncol(wide_data$X) + ncol(wide_data$y), means)

  list(wide = new_wide_data, synth_data = new_synth_data, mhat = mhat)
}

extract_covariates <- function(form, unit_name, time_name, t_int, data, cov_agg) {
  if (is.null(cov_agg)) {
    cov_agg <- list(function(x) mean(x, na.rm = TRUE))
  } else if (is.function(cov_agg)) {
    cov_agg <- list(cov_agg)
  }

  pre_data <- data[data[[time_name]] < t_int, , drop = FALSE]
  cov_terms <- stats::terms(form, rhs = 2, data = data)
  cov_form <- stats::update(stats::formula(stats::delete.response(cov_terms)), ~ . - 1)
  mf <- stats::model.frame(cov_form, pre_data, na.action = NULL)
  mm <- stats::model.matrix(cov_form, mf)

  units <- sorted_unique(data[[unit_name]])
  rows_by_unit <- split(seq_len(nrow(mm)), as.character(pre_data[[unit_name]]))

  cols <- vector("list", 0L)
  names_out <- character()

  for (j in seq_len(ncol(mm))) {
    for (k in seq_along(cov_agg)) {
      f <- cov_agg[[k]]
      vals <- vapply(units, function(unit_value) {
        idx <- rows_by_unit[[as.character(unit_value)]]
        if (is.null(idx)) {
          return(NA_real_)
        }
        value <- f(mm[idx, j])
        if (length(value) != 1L) {
          stop("Each cov_agg function must return a scalar", call. = FALSE)
        }
        as.numeric(value)
      }, numeric(1))
      cols[[length(cols) + 1L]] <- vals
      fname <- names(cov_agg)[k]
      if (is.null(fname) || !nzchar(fname)) {
        fname <- if (length(cov_agg) == 1L) "" else paste0("_agg", k)
      }
      names_out <- c(names_out, paste0(colnames(mm)[j], fname))
    }
  }

  Z <- do.call(cbind, cols)
  if (is.null(dim(Z))) {
    Z <- matrix(Z, ncol = 1L)
  }
  colnames(Z) <- names_out
  rownames(Z) <- as.character(units)

  if (nrow(Z) != length(units) || any(apply(Z, 1, function(x) all(is.na(x))))) {
    stop("Some units are missing all covariate data", call. = FALSE)
  }

  Zsds <- apply(Z, 2, stats::sd)
  if (any(Zsds == 0)) {
    zero_covs <- paste(colnames(Z)[Zsds == 0], collapse = ", ")
    stop(
      paste("The following covariates have no variation across units:", zero_covs),
      call. = FALSE
    )
  }

  Z
}

make_V_matrix <- function(t0, V) {
  if (is.null(V)) {
    diag(rep(1, t0))
  } else if (is.vector(V)) {
    if (length(V) != t0) {
      stop(sprintf("`V` must be a vector with %d elements or a %d x %d matrix", t0, t0, t0), call. = FALSE)
    }
    diag(V)
  } else if (is.matrix(V) && ncol(V) == 1L && nrow(V) == t0) {
    diag(c(V))
  } else if (is.matrix(V) && nrow(V) == 1L && ncol(V) == t0) {
    diag(c(V))
  } else if (is.matrix(V) && nrow(V) == t0 && ncol(V) == t0) {
    V
  } else {
    stop(sprintf("`V` must be a vector with %d elements or a %d x %d matrix", t0, t0, t0), call. = FALSE)
  }
}

jsynth_weights <- function(target, donors) {
  if (!is.matrix(donors) || !is.numeric(donors)) stop("donors must be a numeric matrix", call. = FALSE)
  if (!is.numeric(target)) stop("target must be numeric", call. = FALSE)
  if (ncol(donors) != length(target)) stop("ncol(donors) must equal length(target)", call. = FALSE)
  .Call(C_jsynth_weights, donors, as.double(target), ensure_backend_available())
}

jridge_augsynth_inner <- function(X_c, X_1,
                                  lambda = NULL,
                                  ridge = TRUE,
                                  scm = TRUE,
                                  lambda_min_ratio = 1e-8,
                                  n_lambda = 20,
                                  lambda_max = NULL,
                                  holdout_length = 1,
                                  min_1se = TRUE) {
  if (!is.matrix(X_c) || !is.numeric(X_c)) stop("X_c must be a numeric matrix", call. = FALSE)
  X_1 <- as.double(c(X_1))
  if (ncol(X_c) != length(X_1)) stop("ncol(X_c) must equal length(X_1)", call. = FALSE)

  select_lambda <- isTRUE(ridge) && is.null(lambda)
  lambdas <- numeric()
  if (isTRUE(ridge)) {
    if (select_lambda) {
      if (is.null(lambda_max)) {
        lambda_max <- get_lambda_max(X_c)
      }
      lambdas <- create_lambda_list(lambda_max, lambda_min_ratio, n_lambda)
    } else {
      lambdas <- as.double(lambda)
    }
  }

  out <- .Call(
    C_jridge_augsynth_inner,
    X_c,
    X_1,
    as.logical(ridge),
    as.logical(scm),
    as.logical(select_lambda),
    as.double(lambdas),
    as.integer(holdout_length),
    as.logical(min_1se),
    ensure_backend_available()
  )

  out$weights <- matrix(out$weights, ncol = 1L)
  out$synw <- matrix(out$synw, ncol = 1L)
  if (select_lambda) {
    out$lambdas <- lambdas
  } else {
    out$lambdas <- NULL
    out$lambda_errors <- NULL
    out$lambda_errors_se <- NULL
  }
  out
}

fit_synth_formatted <- function(synth_data, V = NULL) {
  t0 <- nrow(synth_data$Z0)
  V <- make_V_matrix(t0, V)

  donors <- t(synth_data$X0)
  target <- c(synth_data$X1)
  if (!isTRUE(all.equal(V, diag(rep(1, t0))))) {
    sqrtV <- matrix_sqrt_psd(V)
    donors <- donors %*% sqrtV
    target <- c(matrix(target, nrow = 1L) %*% sqrtV)
  }

  weights <- matrix(jsynth_weights(target, donors), ncol = 1L)
  l2_imbalance <- sqrt(sum((synth_data$Z0 %*% weights - synth_data$Z1) ^ 2))
  uni_w <- matrix(1 / ncol(synth_data$Z0), nrow = ncol(synth_data$Z0), ncol = 1L)
  unif_l2_imbalance <- sqrt(sum((synth_data$Z0 %*% uni_w - synth_data$Z1) ^ 2))

  list(
    weights = weights,
    l2_imbalance = l2_imbalance,
    scaled_l2_imbalance = l2_imbalance / unif_l2_imbalance
  )
}

fit_ridgeaug_formatted <- function(wide_data, synth_data,
                                   Z = NULL, lambda = NULL, ridge = TRUE, scm = TRUE,
                                   lambda_min_ratio = 1e-8, n_lambda = 20,
                                   lambda_max = NULL,
                                   holdout_length = 1, min_1se = TRUE,
                                   V = NULL,
                                   residualize = FALSE, ...) {
  extra_params <- list(...)
  if (length(extra_params) > 0L) {
    warning(
      "Unused parameters in using ridge augmented weights: ",
      paste(names(extra_params), collapse = ", "),
      call. = FALSE
    )
  }

  X <- wide_data$X
  y <- wide_data$y
  trt <- wide_data$trt

  X_cent <- center_by_controls(X, trt)
  X_c <- X_cent[trt == 0, , drop = FALSE]
  X_1 <- matrix(colMeans(X_cent[trt == 1, , drop = FALSE]), nrow = 1L)
  y_cent <- center_by_controls(y, trt)
  y_c <- y_cent[trt == 0, , drop = FALSE]

  t0 <- ncol(X_c)
  V <- make_V_matrix(t0, V)
  X_c <- X_c %*% V
  X_1 <- X_1 %*% V

  new_synth_data <- synth_data

  if (!is.null(Z)) {
    Z_cent <- center_by_controls(Z, trt)
    Z_c <- Z_cent[trt == 0, , drop = FALSE]
    Z_1 <- matrix(colMeans(Z_cent[trt == 1, , drop = FALSE]), nrow = 1L)

    if (residualize) {
      z_proj <- solve_square(crossprod(Z_c), t(Z_c))
      Xc_hat <- Z_c %*% z_proj %*% X_c
      X1_hat <- Z_1 %*% z_proj %*% X_c
      res_c <- X_c - Xc_hat
      res_t <- X_1 - X1_hat

      X_c <- res_c
      X_1 <- res_t
      X_cent[trt == 0, ] <- res_c
      X_cent[trt == 1, ] <- res_t

      new_synth_data$Z1 <- t(res_t)
      new_synth_data$X1 <- t(res_t)
      new_synth_data$Z0 <- t(res_c)
      new_synth_data$X0 <- t(res_c)
    } else {
      sdz <- apply(Z_c, 2, stats::sd)
      sdx <- stats::sd(as.numeric(X_c))
      Z_c <- sweep(Z_c, 2, sdz, "/") * sdx
      Z_1 <- sweep(Z_1, 2, sdz, "/") * sdx

      X_c <- cbind(X_c, Z_c)
      X_1 <- cbind(X_1, Z_1)
      new_synth_data$Z1 <- t(X_1)
      new_synth_data$X1 <- t(X_1)
      new_synth_data$Z0 <- t(X_c)
      new_synth_data$X0 <- t(X_c)
    }
  } else {
    new_synth_data$Z1 <- t(X_1)
    new_synth_data$X1 <- t(X_1)
    new_synth_data$Z0 <- t(X_c)
    new_synth_data$X0 <- t(X_c)
  }

  out <- jridge_augsynth_inner(
    X_c = X_c,
    X_1 = X_1,
    lambda = lambda,
    ridge = ridge,
    scm = scm,
    lambda_min_ratio = lambda_min_ratio,
    n_lambda = n_lambda,
    lambda_max = lambda_max,
    holdout_length = holdout_length,
    min_1se = min_1se
  )

  weights <- out$weights
  synw <- out$synw
  lambda <- out$lambda
  lambdas <- out$lambdas
  lambda_errors <- out$lambda_errors
  lambda_errors_se <- out$lambda_errors_se

  if (!is.null(Z)) {
    if (residualize) {
      no_cov_weights <- weights
      ridge_w <- t(t(Z_1) - t(Z_c) %*% weights) %*% solve_square(crossprod(Z_c)) %*% t(Z_c)
      weights <- weights + t(ridge_w)
    } else {
      no_cov_weights <- NULL
    }
  }

  l2_imbalance <- sqrt(sum((synth_data$X0 %*% weights - synth_data$X1) ^ 2))
  uni_w <- matrix(1 / ncol(synth_data$X0), nrow = ncol(synth_data$X0), ncol = 1L)
  unif_l2_imbalance <- sqrt(sum((synth_data$X0 %*% uni_w - synth_data$X1) ^ 2))

  mhat <- matrix(0, nrow = nrow(y), ncol = ncol(y))
  ridge_mhat <- mhat

  if (!is.null(Z)) {
    if (residualize) {
      z_proj <- solve_square(crossprod(Z_c), t(Z_c))
      ridge_mhat <- ridge_mhat + Z_cent %*% z_proj %*% y_c
      yc_hat <- ridge_mhat[trt == 0, , drop = FALSE]
      y_c <- y_c - yc_hat
    } else {
      X_cent <- cbind(X_cent, Z_cent)
    }
  }

  if (ridge) {
    ridge_mhat <- ridge_mhat + X_cent %*% solve_square(
      crossprod(X_c) + lambda * diag(ncol(X_c)),
      t(X_c) %*% y_c
    )
  }

  output <- list(
    weights = weights,
    l2_imbalance = l2_imbalance,
    scaled_l2_imbalance = l2_imbalance / unif_l2_imbalance,
    mhat = mhat,
    lambda = lambda,
    ridge_mhat = ridge_mhat,
    synw = synw,
    lambdas = lambdas,
    lambda_errors = lambda_errors,
    lambda_errors_se = lambda_errors_se
  )

  if (!is.null(Z)) {
    output$no_cov_weights <- no_cov_weights
    z_l2_imbalance <- sqrt(sum((t(Z_c) %*% weights - t(Z_1)) ^ 2))
    z_unif_l2_imbalance <- sqrt(sum((t(Z_c) %*% uni_w - t(Z_1)) ^ 2))
    output$covariate_l2_imbalance <- z_l2_imbalance
    output$scaled_covariate_l2_imbalance <- z_l2_imbalance / z_unif_l2_imbalance
  }

  output
}

get_lambda_max <- function(X_c) {
  base::svd(X_c)$d[1] ^ 2
}

create_lambda_list <- function(lambda_max, lambda_min_ratio, n_lambda) {
  scaler <- (lambda_min_ratio) ^ (1 / n_lambda)
  lambda_max * (scaler ^ (seq(0:n_lambda) - 1))
}

choose_lambda <- function(lambdas, lambda_errors, lambda_errors_se, min_1se) {
  min_idx <- which.min(lambda_errors)
  min_error <- lambda_errors[min_idx]
  min_se <- lambda_errors_se[min_idx]
  lambda_min <- lambdas[min_idx]
  lambda_1se <- max(lambdas[lambda_errors <= min_error + min_se])
  if (isTRUE(min_1se)) lambda_1se else lambda_min
}

fit_augsynth_internal <- function(wide, synth_data, Z, progfunc,
                                  scm, fixedeff, V = NULL, ...) {
  n <- nrow(wide$X)
  t0 <- ncol(wide$X)
  ttot <- t0 + ncol(wide$y)

  if (fixedeff) {
    demeaned <- demean_data(wide, synth_data)
    fit_wide <- demeaned$wide
    fit_synth_data <- demeaned$synth_data
    mhat <- demeaned$mhat
  } else {
    fit_wide <- wide
    fit_synth_data <- synth_data
    mhat <- matrix(0, n, ttot)
  }

  if (is.null(progfunc)) {
    progfunc <- "none"
  }
  progfunc <- tolower(progfunc)

  if (progfunc == "ridge") {
    augsynth_fit <- do.call(
      fit_ridgeaug_formatted,
      c(list(wide_data = fit_wide, synth_data = fit_synth_data, Z = Z, V = V, scm = scm), list(...))
    )
  } else if (progfunc == "none") {
    augsynth_fit <- do.call(
      fit_ridgeaug_formatted,
      c(list(wide_data = fit_wide, synth_data = fit_synth_data, Z = Z, ridge = FALSE, scm = TRUE, V = V), list(...))
    )
  } else {
    stop("Only progfunc = 'ridge' or 'none' is implemented in metricsjl v1", call. = FALSE)
  }

  augsynth_fit$mhat <- mhat + cbind(matrix(0, nrow = n, ncol = t0), augsynth_fit$mhat)
  augsynth_fit$data <- wide
  augsynth_fit$data$Z <- Z
  augsynth_fit$data$synth_data <- synth_data
  augsynth_fit$progfunc <- progfunc
  augsynth_fit$scm <- scm
  augsynth_fit$fixedeff <- fixedeff
  augsynth_fit$extra_args <- list(...)

  if (progfunc == "ridge") {
    augsynth_fit$extra_args$lambda <- augsynth_fit$lambda
  }

  class(augsynth_fit) <- "augsynth"
  augsynth_fit
}

#' Fit Augmented Synthetic Control for single-period treatment.
#' @export
single_augsynth <- function(form, unit, time, t_int, data,
                            progfunc = "ridge",
                            scm = TRUE,
                            fixedeff = FALSE,
                            cov_agg = NULL,
                            .unit_name = NULL,
                            .time_name = NULL,
                            ...) {
  call_name <- match.call()
  form <- Formula::Formula(form)
  unit_name <- if (!is.null(.unit_name)) .unit_name else resolve_column_name(substitute(unit), parent.frame())
  time_name <- if (!is.null(.time_name)) .time_name else resolve_column_name(substitute(time), parent.frame())

  if (is_multi_outcome_formula(form)) {
    stop("Multiple outcomes are not implemented in metricsjl v1", call. = FALSE)
  }

  outcome_expr <- formula_outcome_expr(form)
  trt_expr <- formula_treatment_expr(form)
  wide <- format_data(outcome_expr, trt_expr, unit_name, time_name, t_int, data)
  synth_data <- do.call(format_synth, wide)

  units <- sorted_unique(data[[unit_name]])
  trt_values <- as.numeric(eval_formula_expr(trt_expr, data))
  trt_by_unit <- tapply(trt_values, data[[unit_name]], function(x) max(x, na.rm = TRUE))
  treated_units <- units[trt_by_unit[as.character(units)] == 1]
  control_units <- units[trt_by_unit[as.character(units)] == 0]

  Z <- NULL
  if (length(form)[2] >= 2L) {
    Z <- extract_covariates(form, unit_name, time_name, t_int, data, cov_agg)
  }

  augsynth_fit <- fit_augsynth_internal(wide, synth_data, Z, progfunc, scm, fixedeff, ...)
  augsynth_fit$data$time <- sorted_unique(data[[time_name]])
  augsynth_fit$call <- call_name
  augsynth_fit$t_int <- t_int
  augsynth_fit$weights <- matrix(augsynth_fit$weights, ncol = 1L)
  rownames(augsynth_fit$weights) <- as.character(control_units)
  augsynth_fit$treated_units <- as.character(treated_units)
  augsynth_fit
}

#' Top-level single-period augsynth API.
#' @export
augsynth <- function(form, unit, time, data, t_int = NULL, ...) {
  form <- Formula::Formula(form)
  if (is_multi_outcome_formula(form)) {
    stop("Multiple outcomes are not implemented in metricsjl v1", call. = FALSE)
  }

  unit_name <- resolve_column_name(substitute(unit), parent.frame())
  time_name <- resolve_column_name(substitute(time), parent.frame())
  trt_times <- infer_treatment_schedule(data, unit_name, time_name, formula_treatment_expr(form))

  if (length(trt_times) > 1L) {
    stop("Staggered adoption / multisynth is not implemented in metricsjl v1", call. = FALSE)
  }
  if (is.null(t_int)) {
    t_int <- trt_times[[1]]
  }

  single_augsynth(
    form,
    unit,
    time,
    t_int = t_int,
    data = data,
    .unit_name = unit_name,
    .time_name = time_name,
    ...
  )
}

#' @export
predict.augsynth <- function(object, att = FALSE, ...) {
  augsynth_fit <- object
  X <- augsynth_fit$data$X
  y <- augsynth_fit$data$y
  comb <- cbind(X, y)
  trt <- augsynth_fit$data$trt
  mhat <- augsynth_fit$mhat

  m1 <- colMeans(mhat[trt == 1, , drop = FALSE])
  resid <- comb[trt == 0, , drop = FALSE] - mhat[trt == 0, , drop = FALSE]
  y0 <- m1 + t(resid) %*% augsynth_fit$weights

  if (isTRUE(att)) {
    return(colMeans(comb[trt == 1, , drop = FALSE]) - c(y0))
  }

  y0_vec <- c(y0)
  names(y0_vec) <- rownames(y0)
  y0_vec
}

#' @export
summary.augsynth <- function(object, inf = TRUE, inf_type = "conformal",
                             linear_effect = FALSE, ...) {
  augsynth_fit <- object
  summ <- list()

  t0 <- ncol(augsynth_fit$data$X)
  t_final <- t0 + ncol(augsynth_fit$data$y)

  if (inf) {
    if (inf_type == "jackknife") {
      att_se <- jackknife_se_single(augsynth_fit)
    } else if (inf_type == "jackknife+") {
      att_se <- time_jackknife_plus(augsynth_fit, ...)
    } else if (inf_type == "conformal") {
      att_se <- conformal_inf(augsynth_fit, ...)
      if (linear_effect) {
        att_linear <- conformal_inf_linear(augsynth_fit, ...)
      }
    } else {
      stop(sprintf("%s is not a valid choice of 'inf_type'", inf_type), call. = FALSE)
    }

    att <- data.frame(Time = augsynth_fit$data$time, Estimate = att_se$att[1:t_final])
    if (inf_type == "jackknife") {
      att$Std.Error <- att_se$se[1:t_final]
      att_avg_se <- att_se$se[t_final + 1]
    } else {
      att_avg_se <- NA_real_
    }
    att_avg <- att_se$att[t_final + 1]
    if (inf_type %in% c("jackknife+", "conformal")) {
      att$lower_bound <- att_se$lb[1:t_final]
      att$upper_bound <- att_se$ub[1:t_final]
    }
    if (inf_type == "conformal") {
      att$p_val <- att_se$p_val[1:t_final]
    }
  } else {
    att_est <- predict(augsynth_fit, att = TRUE)
    att <- data.frame(Time = augsynth_fit$data$time, Estimate = att_est)
    att$Std.Error <- NA_real_
    att_avg <- mean(att_est[(t0 + 1):t_final])
    att_avg_se <- NA_real_
  }

  summ$att <- att

  if (inf) {
    if (inf_type == "jackknife+") {
      summ$average_att <- data.frame(
        Value = "Average Post-Treatment Effect",
        Estimate = att_avg,
        Std.Error = att_avg_se,
        lower_bound = att_se$lb[t_final + 1],
        upper_bound = att_se$ub[t_final + 1]
      )
      summ$alpha <- att_se$alpha
    } else if (inf_type == "conformal") {
      if (linear_effect) {
        summ$average_att <- data.frame(
          Value = c("Average Post-Treatment Effect", "Treatment Effect Intercept", "Treatment Effect Slope"),
          Estimate = c(att_avg, att_linear$est_int, att_linear$est_slope),
          Std.Error = c(att_avg_se, NA_real_, NA_real_),
          p_val = c(att_se$p_val[t_final + 1], NA_real_, NA_real_),
          lower_bound = c(att_se$lb[t_final + 1], att_linear$ci_int[1], att_linear$ci_slope[1]),
          upper_bound = c(att_se$ub[t_final + 1], att_linear$ci_int[2], att_linear$ci_slope[2])
        )
      } else {
        summ$average_att <- data.frame(
          Value = "Average Post-Treatment Effect",
          Estimate = att_avg,
          Std.Error = att_avg_se,
          p_val = att_se$p_val[t_final + 1],
          lower_bound = att_se$lb[t_final + 1],
          upper_bound = att_se$ub[t_final + 1]
        )
      }
      summ$alpha <- att_se$alpha
    } else {
      summ$average_att <- data.frame(
        Value = "Average Post-Treatment Effect",
        Estimate = att_avg,
        Std.Error = att_avg_se
      )
    }
  } else {
    summ$average_att <- data.frame(
      Value = "Average Post-Treatment Effect",
      Estimate = att_avg,
      Std.Error = att_avg_se
    )
  }

  summ$t_int <- augsynth_fit$t_int
  summ$call <- augsynth_fit$call
  summ$l2_imbalance <- augsynth_fit$l2_imbalance
  summ$scaled_l2_imbalance <- augsynth_fit$scaled_l2_imbalance

  if (!is.null(augsynth_fit$covariate_l2_imbalance)) {
    summ$covariate_l2_imbalance <- augsynth_fit$covariate_l2_imbalance
    summ$scaled_covariate_l2_imbalance <- augsynth_fit$scaled_covariate_l2_imbalance
  }

  if (tolower(augsynth_fit$progfunc) == "ridge") {
    mhat <- augsynth_fit$ridge_mhat
    w <- augsynth_fit$synw
  } else {
    mhat <- augsynth_fit$mhat
    w <- augsynth_fit$weights
  }

  trt <- augsynth_fit$data$trt
  m1 <- colMeans(mhat[trt == 1, , drop = FALSE])
  if (tolower(augsynth_fit$progfunc) == "none" || (!augsynth_fit$scm)) {
    summ$bias_est <- NA
  } else {
    summ$bias_est <- m1 - t(mhat[trt == 0, , drop = FALSE]) %*% w
  }

  summ$inf_type <- if (inf) inf_type else "None"
  class(summ) <- "summary.augsynth"
  summ
}

#' @export
print.augsynth <- function(x, ...) {
  cat("\nCall:\n", paste(deparse(x$call), collapse = "\n"), "\n\n", sep = "")
  t0 <- ncol(x$data$X)
  ttotal <- t0 + ncol(x$data$y)
  att_post <- predict(x, att = TRUE)[(t0 + 1):ttotal]
  cat("Average ATT Estimate: ", format(round(mean(att_post), 3), nsmall = 3), "\n\n", sep = "")
}

#' @export
print.summary.augsynth <- function(x, ...) {
  summ <- x
  cat("\nCall:\n", paste(deparse(summ$call), collapse = "\n"), "\n\n", sep = "")

  t_final <- nrow(summ$att)
  t_int_idx <- sum(summ$att$Time <= summ$t_int)
  att_post <- summ$average_att$Estimate[1]

  if (summ$inf_type == "jackknife") {
    se_avg <- summ$average_att$Std.Error[1]
    out_msg <- paste0(
      "Average ATT Estimate (Jackknife Std. Error): ",
      format(round(att_post, 3), nsmall = 3),
      "  (",
      format(round(se_avg, 3), nsmall = 3),
      ")\n"
    )
    inf_label <- "Jackknife over units"
  } else if (summ$inf_type == "conformal") {
    p_val <- summ$average_att$p_val[1]
    out_msg <- paste0(
      "Average ATT Estimate (p Value for Joint Null): ",
      format(att_post, digits = 3),
      "  (",
      format(p_val, digits = 2),
      ")\n"
    )
    inf_label <- "Conformal inference"
    if ("Treatment Effect Slope" %in% summ$average_att$Value) {
      lowers <- summ$average_att$lower_bound[2:3]
      uppers <- summ$average_att$upper_bound[2:3]
      out_msg <- paste0(
        out_msg,
        "Confidence intervals for linear-in-time treatment effects (Intercept + Slope * Time)\n",
        "\tIntercept: [",
        format(lowers[1], digits = 3),
        ",",
        format(uppers[1], digits = 3),
        "]\n",
        "\tSlope: [",
        format(lowers[2], digits = 3),
        ",",
        format(uppers[2], digits = 3),
        "]\n"
      )
    }
  } else if (summ$inf_type == "jackknife+") {
    out_msg <- paste0(
      "Average ATT Estimate: ",
      format(round(att_post, 3), nsmall = 3),
      "\n"
    )
    inf_label <- "Jackknife+ over time periods"
  } else {
    out_msg <- paste0(
      "Average ATT Estimate: ",
      format(round(att_post, 3), nsmall = 3),
      "\n"
    )
    inf_label <- "None"
  }

  out_msg <- paste0(
    out_msg,
    "L2 Imbalance: ",
    format(round(summ$l2_imbalance, 3), nsmall = 3),
    "\n",
    "Percent improvement from uniform weights: ",
    format(round(1 - summ$scaled_l2_imbalance, 3) * 100),
    "%\n\n"
  )

  if (!is.null(summ$covariate_l2_imbalance)) {
    out_msg <- paste0(
      out_msg,
      "Covariate L2 Imbalance: ",
      format(round(summ$covariate_l2_imbalance, 3), nsmall = 3),
      "\n",
      "Percent improvement from uniform weights: ",
      format(round(1 - summ$scaled_covariate_l2_imbalance, 3) * 100),
      "%\n\n"
    )
  }

  out_msg <- paste0(
    out_msg,
    "Avg Estimated Bias: ",
    format(round(mean(summ$bias_est), 3), nsmall = 3),
    "\n\n",
    "Inference type: ",
    inf_label,
    "\n\n"
  )
  cat(out_msg)

  if (summ$inf_type == "jackknife") {
    out_att <- summ$att[t_int_idx:t_final, c("Time", "Estimate", "Std.Error")]
  } else if (summ$inf_type == "conformal") {
    out_att <- summ$att[t_int_idx:t_final, c("Time", "Estimate", "lower_bound", "upper_bound", "p_val")]
    names(out_att) <- c(
      "Time",
      "Estimate",
      paste0((1 - summ$alpha) * 100, "% CI Lower Bound"),
      paste0((1 - summ$alpha) * 100, "% CI Upper Bound"),
      "p Value"
    )
  } else if (summ$inf_type == "jackknife+") {
    out_att <- summ$att[t_int_idx:t_final, c("Time", "Estimate", "lower_bound", "upper_bound")]
    names(out_att) <- c(
      "Time",
      "Estimate",
      paste0((1 - summ$alpha) * 100, "% CI Lower Bound"),
      paste0((1 - summ$alpha) * 100, "% CI Upper Bound")
    )
  } else {
    out_att <- summ$att[t_int_idx:t_final, c("Time", "Estimate")]
  }

  value_cols <- setdiff(names(out_att), "Time")
  out_att[value_cols] <- lapply(out_att[value_cols], function(col) round(col, 3))
  print(out_att, row.names = FALSE)
  invisible(summ)
}
