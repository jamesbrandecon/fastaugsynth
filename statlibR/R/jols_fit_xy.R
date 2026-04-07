#' Fit OLS from numeric X and y via compiled backend
#' @export
jols_fit_xy <- function(X, y) {
  if (!is.matrix(X) || !is.numeric(X)) stop("X must be a numeric matrix", call. = FALSE)
  if (!is.numeric(y)) stop("y must be numeric", call. = FALSE)
  if (nrow(X) != length(y)) stop("nrow(X) must match length(y)", call. = FALSE)

  fit <- .Call(C_jols_fit_xy, X, as.double(y), backend_path())
  class(fit) <- c("jols_fit", "list")
  fit
}
