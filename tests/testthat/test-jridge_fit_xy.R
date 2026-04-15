test_that("jridge_fit_xy returns valid lambda and improves over extreme lambda", {
  skip_if_not(file.exists(backend_path()), "backend library is not installed")

  set.seed(11)
  n <- 100
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  beta <- c(rep(1.5, 5), rep(0, p - 5))
  y <- as.vector(X %*% beta + rnorm(n, sd = 0.25))
  lambdas <- 10 ^ seq(-6, 2, length.out = 40)

  fit <- jridge_fit_xy(X, y, lambdas)

  expect_true(is.numeric(fit$coefficients))
  expect_equal(length(fit$coefficients), p)
  expect_true(fit$best_lambda %in% lambdas)
  expect_true(is.finite(fit$loocv_mse))

  # Compare training MSE against an intentionally over-shrunk fit.
  b_bad <- solve(crossprod(X) + 1e6 * diag(p), crossprod(X, y))
  mse_bad <- mean((y - X %*% b_bad) ^ 2)
  mse_fit <- mean((y - X %*% fit$coefficients) ^ 2)
  expect_lt(mse_fit, mse_bad)
})
