basque_panel <- function() {
  skip_if_not_installed("Synth")
  data("basque", package = "Synth")
  basque$trt <- ifelse(
    basque$year < 1975,
    0,
    ifelse(basque$regionno == 17, 1, 0)
  )
  subset(basque, regionno != 1)
}

test_that("single-period SCM matches upstream basque answers", {
  basque <- basque_panel()

  syn <- augsynth(gdpcap ~ trt, regionno, year, basque, progfunc = "None", scm = TRUE, t_int = 1975)
  summ <- summary(syn, inf = FALSE)

  expect_equal(mean(summ$att$Estimate), -0.3686, tolerance = 1e-4)
  expect_equal(syn$l2_imbalance, 0.377, tolerance = 1e-3)

  syn_inferred <- augsynth(gdpcap ~ trt, regionno, year, basque, progfunc = "None", scm = TRUE)
  expect_equal(c(syn$weights), c(syn_inferred$weights), tolerance = 1e-6)

  basque_rev <- basque[order(-basque$year), ]
  syn_rev <- augsynth(gdpcap ~ trt, regionno, year, basque_rev, progfunc = "None", scm = TRUE)
  expect_equal(c(syn$weights), c(syn_rev$weights), tolerance = 1e-6)
  expect_equal(predict(syn), predict(syn_rev), tolerance = 1e-6)
})

test_that("single-period ridge ASCM matches upstream basque answers", {
  basque <- basque_panel()

  asyn <- augsynth(gdpcap ~ trt, regionno, year, basque, progfunc = "Ridge", scm = TRUE, lambda = 8)
  summ <- summary(asyn, inf = FALSE)

  expect_equal(mean(summ$att$Estimate), -0.3696, tolerance = 1e-3)
  expect_equal(asyn$l2_imbalance, 0.373, tolerance = 1e-3)
})

test_that("ridge lambda tuning metadata follows upstream rules", {
  basque <- basque_panel()

  tuned <- augsynth(gdpcap ~ trt, regionno, year, basque, progfunc = "Ridge", scm = TRUE)
  expect_equal(tail(tuned$lambdas, 1) / tuned$lambdas[1], 1e-8, tolerance = 1e-10)

  min_idx <- which.min(tuned$lambda_errors)
  min_1se <- max(
    tuned$lambdas[
      tuned$lambda_errors <= tuned$lambda_errors[min_idx] + tuned$lambda_errors_se[min_idx]
    ]
  )
  expect_equal(tuned$lambda, min_1se, tolerance = 1e-12)
})

test_that("covariate residualization reproduces upstream basque benchmark", {
  basque <- basque_panel()

  covsyn <- augsynth(
    gdpcap ~ trt | invest + popdens,
    regionno,
    year,
    basque,
    progfunc = "None",
    scm = TRUE,
    residualize = TRUE
  )

  summ <- summary(covsyn, inf = FALSE)
  expect_equal(mean(summ$att$Estimate), -0.1443, tolerance = 1e-3)
  expect_equal(covsyn$l2_imbalance, 0.3720, tolerance = 1e-3)
  expect_equal(covsyn$covariate_l2_imbalance, 0, tolerance = 1e-3)
})
