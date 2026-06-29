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

test_that("single-augsynth accepts named symbols via variables for unit and time", {
  basque <- basque_panel()

  unit_var <- as.name("regionno")
  time_var <- as.name("year")

  syn <- augsynth(gdpcap ~ trt, unit_var, time_var, basque, progfunc = "None", scm = TRUE, t_int = 1975)
  baseline <- augsynth(gdpcap ~ trt, regionno, year, basque, progfunc = "None", scm = TRUE, t_int = 1975)

  expect_equal(c(syn$weights), c(baseline$weights), tolerance = 1e-6)
  expect_equal(mean(summary(syn, inf = FALSE)$att$Estimate), mean(summary(baseline, inf = FALSE)$att$Estimate), tolerance = 1e-6)
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

test_that("jackknife summary works for single-period SCM and ridge fits", {
  basque <- basque_panel()

  syn <- augsynth(gdpcap ~ trt, regionno, year, basque, progfunc = "None", scm = TRUE, t_int = 1975)
  asyn <- augsynth(gdpcap ~ trt, regionno, year, basque, progfunc = "Ridge", scm = TRUE, lambda = 8, t_int = 1975)

  jack_syn <- summary(syn, inf = TRUE, inf_type = "jackknife")
  jack_asyn <- summary(asyn, inf = TRUE, inf_type = "jackknife")

  expect_s3_class(jack_syn, "summary.augsynth")
  expect_equal(jack_syn$average_att$Estimate, -0.6915277, tolerance = 1e-4)
  expect_equal(jack_syn$average_att$Std.Error, 0.05496423, tolerance = 1e-4)
  expect_equal(jack_syn$att$Estimate[21:25], c(0.1443060, 0.0093784, -0.1216668, -0.2876975, -0.4166083), tolerance = 1e-4)
  expect_equal(jack_syn$att$Std.Error[21:25], c(0.07854754, 0.06127062, 0.04507349, 0.06290192, 0.04505673), tolerance = 1e-4)
  expect_match(paste(capture.output(print(jack_syn)), collapse = "\n"), "Average ATT Estimate \\(Jackknife Std\\. Error\\)")

  expect_equal(jack_asyn$average_att$Estimate, -0.6923196, tolerance = 1e-4)
  expect_equal(jack_asyn$average_att$Std.Error, 0.1429878, tolerance = 1e-4)
  expect_equal(jack_asyn$att$Estimate[21:25], c(0.1424433, 0.0078239, -0.1230318, -0.2903846, -0.4198612), tolerance = 1e-4)
  expect_equal(jack_asyn$att$Std.Error[21:25], c(0.03368404, 0.04762395, 0.06166484, 0.08863483, 0.08525669), tolerance = 1e-4)
})

test_that("None-program jackknife follows fitted SCM weights even when scm flag is false", {
  basque <- basque_panel()

  fit <- augsynth(gdpcap ~ trt, regionno, year, basque, progfunc = "None", scm = FALSE, t_int = 1975)
  backend <- jackknife_se_single(fit)
  fallback <- .jackknife_se_single(fit)

  expect_equal(unname(backend$att), unname(fallback$att), tolerance = 1e-8)
  expect_equal(unname(backend$se), unname(fallback$se), tolerance = 1e-8)
})

test_that("conformal summary matches deterministic basque benchmarks", {
  basque <- basque_panel()

  syn <- augsynth(gdpcap ~ trt, regionno, year, basque, progfunc = "None", scm = TRUE, t_int = 1975)
  asyn <- augsynth(gdpcap ~ trt, regionno, year, basque, progfunc = "Ridge", scm = TRUE, lambda = 8, t_int = 1975)

  conf_syn <- summary(
    syn,
    inf = TRUE,
    inf_type = "conformal",
    type = "block",
    grid_size = 11,
    conformal_mode = "reference"
  )
  conf_asyn <- summary(
    asyn,
    inf = TRUE,
    inf_type = "conformal",
    type = "block",
    grid_size = 11,
    conformal_mode = "reference"
  )

  expect_equal(conf_syn$average_att$Estimate, -0.6915277, tolerance = 1e-4)
  expect_equal(conf_syn$average_att$p_val, 0.1860465, tolerance = 1e-6)
  expect_equal(conf_syn$att$Estimate[21:25], c(0.1443060, 0.0093784, -0.1216668, -0.2876975, -0.4166083), tolerance = 1e-4)
  expect_equal(conf_syn$att$lower_bound[21:25], c(0, 0, -0.1216668, -0.2876975, -0.4166083), tolerance = 1e-4)
  expect_equal(conf_syn$att$p_val[21:25], c(0.19047619, 1, 0.23809524, 0.04761905, 0.04761905), tolerance = 1e-6)
  expect_match(paste(capture.output(print(conf_syn)), collapse = "\n"), "Average ATT Estimate \\(p Value for Joint Null\\)")

  expect_equal(conf_asyn$average_att$Estimate, -0.6923196, tolerance = 1e-4)
  expect_equal(conf_asyn$average_att$p_val, 0.3023256, tolerance = 1e-6)
  expect_equal(conf_asyn$att$Estimate[21:25], c(0.1424433, 0.0078239, -0.1230318, -0.2903846, -0.4198612), tolerance = 1e-4)
  expect_equal(conf_asyn$att$lower_bound[21:25], c(0, 0, -0.1230318, -0.2903846, -0.4198612), tolerance = 1e-4)
  expect_equal(conf_asyn$att$p_val[21:25], c(0.19047619, 1, 0.28571429, 0.04761905, 0.04761905), tolerance = 1e-6)
})
