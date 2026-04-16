#' Fit ridge regression with LOOCV-selected lambda via compiled backend
#' @param X numeric matrix
#' @param y numeric vector
#' @param lambdas non-negative numeric tuning grid
#' @export
jridge_fit_xy <- function(X, y, lambdas = 10 ^ seq(-6, 2, length.out = 64)) {
  if (!is.matrix(X) || !is.numeric(X)) stop("X must be a numeric matrix", call. = FALSE)
  if (!is.numeric(y)) stop("y must be numeric", call. = FALSE)
  if (nrow(X) != length(y)) stop("nrow(X) must match length(y)", call. = FALSE)
  if (!is.numeric(lambdas) || length(lambdas) == 0) stop("lambdas must be a non-empty numeric vector", call. = FALSE)
  if (any(!is.finite(lambdas)) || any(lambdas < 0)) stop("lambdas must be finite and non-negative", call. = FALSE)

  out <- .Call(C_jridge_fit_xy, X, as.double(y), as.double(lambdas), ensure_backend_available())
  class(out) <- c("jridge_fit", "list")
  out
}
