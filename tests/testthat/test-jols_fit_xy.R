test_that("jols_fit_xy matches lm.fit coefficients", {
  skip_if_not(file.exists(backend_path()), "backend library is not installed")

  set.seed(1)
  n <- 200
  p <- 4
  X <- cbind(1, matrix(rnorm(n * (p - 1)), ncol = p - 1))
  beta <- c(0.5, -1.2, 2.0, 0.3)
  y <- as.vector(X %*% beta + rnorm(n, sd = 0.05))

  got <- jols_fit_xy(X, y)
  ref <- lm.fit(X, y)

  expect_equal(got$coefficients, unname(ref$coefficients), tolerance = 1e-8)
})
