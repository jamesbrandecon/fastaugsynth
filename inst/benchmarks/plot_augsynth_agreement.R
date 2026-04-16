#!/usr/bin/env Rscript

# Visual comparison for fastaugsynth::augsynth() vs upstream augsynth::augsynth().
# Produces a 3 x 3 panel of classic synthetic-control plots:
# - treated outcome
# - synthetic control path from both implementations
# - post-treatment conformal confidence intervals from both implementations
# - treatment-date vertical line

backend_env_var <- function() {
  Sys.getenv("FASTAUGSYNTH_BACKEND_LIB", "")
}

script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_flag <- "--file="
  file_arg <- grep(paste0("^", file_flag), args, value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(sub(file_flag, "", file_arg[[1]]), mustWork = FALSE)))
  }
  normalizePath(getwd(), mustWork = FALSE)
}

default_specs <- function() {
  data.frame(
    donors = c(10L, 10L, 10L, 40L, 40L, 40L, 80L, 80L, 80L),
    pre_periods = c(20L, 60L, 120L, 20L, 60L, 120L, 20L, 60L, 120L),
    post_periods = c(10L, 20L, 10L, 10L, 20L, 10L, 10L, 20L, 10L),
    noise_sd = c(0.5, 0.75, 1.0, 0.75, 1.0, 1.25, 1.0, 1.25, 1.5),
    stringsAsFactors = FALSE
  )
}

default_config <- function() {
  list(
    output_dir = script_dir(),
    backend_lib = backend_env_var(),
    figure_name = "augsynth_agreement_grid.png",
    seed = 20260413L,
    conformal_type = "block",
    conformal_mode = "reference",
    conformal_ns = 1000L,
    conformal_grid_size = 50L,
    specs = default_specs()
  )
}

parse_int_scalar <- function(value, arg_name, min_value = 1L) {
  out <- suppressWarnings(as.integer(value))
  if (length(out) != 1L || is.na(out) || out < min_value) {
    stop(sprintf("Invalid --%s value: %s", arg_name, value), call. = FALSE)
  }
  out
}

parse_cli_args <- function(args) {
  cfg <- default_config()
  print_specs <- FALSE

  idx <- 1L
  while (idx <= length(args)) {
    arg <- args[[idx]]
    if (arg == "--output-dir") {
      idx <- idx + 1L
      cfg$output_dir <- args[[idx]]
    } else if (arg == "--backend-lib") {
      idx <- idx + 1L
      cfg$backend_lib <- args[[idx]]
    } else if (arg == "--figure-name") {
      idx <- idx + 1L
      cfg$figure_name <- args[[idx]]
    } else if (arg == "--seed") {
      idx <- idx + 1L
      cfg$seed <- parse_int_scalar(args[[idx]], "seed", min_value = 0L)
    } else if (arg == "--conformal-type") {
      idx <- idx + 1L
      cfg$conformal_type <- args[[idx]]
    } else if (arg == "--conformal-mode") {
      idx <- idx + 1L
      cfg$conformal_mode <- args[[idx]]
    } else if (arg == "--conformal-ns") {
      idx <- idx + 1L
      cfg$conformal_ns <- parse_int_scalar(args[[idx]], "conformal-ns")
    } else if (arg == "--conformal-grid-size") {
      idx <- idx + 1L
      cfg$conformal_grid_size <- parse_int_scalar(args[[idx]], "conformal-grid-size")
    } else if (arg == "--print-specs") {
      print_specs <- TRUE
    } else {
      stop(sprintf("Unknown argument: %s", arg), call. = FALSE)
    }
    idx <- idx + 1L
  }

  if (!cfg$conformal_type %in% c("iid", "block")) {
    stop("--conformal-type must be either 'iid' or 'block'", call. = FALSE)
  }
  if (!cfg$conformal_mode %in% c("fast", "reference", "reference_conformal")) {
    stop("--conformal-mode must be either 'fast' or 'reference'", call. = FALSE)
  }

  list(config = cfg, print_specs = print_specs)
}

ensure_runtime_deps <- function(cfg) {
  if (!nzchar(cfg$backend_lib) || !file.exists(cfg$backend_lib)) {
    stop(
      "Set FASTAUGSYNTH_BACKEND_LIB or pass --backend-lib so fastaugsynth can find the compiled backend library.",
      call. = FALSE
    )
  }
  if (!requireNamespace("fastaugsynth", quietly = TRUE)) {
    stop("Package 'fastaugsynth' must be installed before running this script.", call. = FALSE)
  }
  if (!requireNamespace("augsynth", quietly = TRUE)) {
    stop("Package 'augsynth' must be installed before running this script.", call. = FALSE)
  }
}

simulate_panel <- function(spec, seed,
                           outcome_name = "gdpcap",
                           unit_name = "regionno",
                           time_name = "year") {
  n_units <- spec$donors + 1L
  unit_values <- paste0("u", seq_len(n_units))
  time_values <- seq_len(spec$pre_periods + spec$post_periods)
  treated_unit <- unit_values[[n_units]]
  t_int <- spec$pre_periods + 1L

  panel <- expand.grid(
    unit = unit_values,
    time = time_values,
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  panel <- setNames(panel, c(unit_name, time_name))
  panel[[unit_name]] <- as.character(panel[[unit_name]])
  panel[[time_name]] <- as.integer(panel[[time_name]])

  set.seed(seed)
  unit_effect <- rnorm(n_units, sd = 1.25)
  unit_slope <- rnorm(n_units, sd = 0.03)
  time_idx <- panel[[time_name]]
  unit_idx <- match(panel[[unit_name]], unit_values)
  trt <- as.integer(panel[[unit_name]] == treated_unit & time_idx >= t_int)

  common_wave <- sin(2 * pi * time_idx / max(time_values)) + 0.02 * time_idx
  baseline <- 4.0 +
    1.8 * unit_effect[unit_idx] +
    unit_slope[unit_idx] * time_idx +
    0.8 * common_wave
  treatment <- as.numeric(trt) * (4.5 + 0.08 * (time_idx - spec$pre_periods))
  panel[[outcome_name]] <- baseline + treatment + rnorm(nrow(panel), sd = spec$noise_sd)
  panel$trt <- trt
  panel <- panel[order(panel[[unit_name]], panel[[time_name]]), ]

  list(
    data = panel,
    outcome = outcome_name,
    unit = unit_name,
    time = time_name,
    t_int = t_int,
    treated_unit = treated_unit
  )
}

ns_fun <- function(pkg, name) {
  get(name, envir = asNamespace(pkg))
}

fit_with_pkg <- function(pkg, formula, unit_name, time_name, data, t_int) {
  fit_fun <- ns_fun(pkg, "augsynth")
  eval(substitute(
    fit_fun(FML, unit = UNIT, time = TIME, data = DATA, progfunc = "None", scm = TRUE, t_int = TINT),
    list(
      fit_fun = fit_fun,
      FML = formula,
      UNIT = as.name(unit_name),
      TIME = as.name(time_name),
      DATA = data,
      TINT = t_int
    )
  ))
}

summary_with_pkg <- function(pkg, fit, cfg) {
  summary_fun <- ns_fun(pkg, "summary.augsynth")
  args <- list(
    fit,
    inf = TRUE,
    inf_type = "conformal",
    type = cfg$conformal_type,
    ns = cfg$conformal_ns,
    grid_size = cfg$conformal_grid_size,
    q = 1
  )
  if (identical(pkg, "fastaugsynth")) {
    args$conformal_mode <- cfg$conformal_mode
  }
  do.call(summary_fun, args)
}

predict_cf_with_pkg <- function(pkg, fit) {
  predict_fun <- ns_fun(pkg, "predict.augsynth")
  predict_fun(fit, att = FALSE)
}

build_panel_data <- function(spec, cfg, spec_id) {
  dataset <- simulate_panel(spec, seed = cfg$seed + spec_id)
  formula <- as.formula(sprintf("%s ~ trt", dataset$outcome))

  fit_metrics <- fit_with_pkg("fastaugsynth", formula, dataset$unit, dataset$time, dataset$data, dataset$t_int)
  fit_upstream <- fit_with_pkg("augsynth", formula, dataset$unit, dataset$time, dataset$data, dataset$t_int)

  sum_metrics <- summary_with_pkg("fastaugsynth", fit_metrics, cfg)
  sum_upstream <- summary_with_pkg("augsynth", fit_upstream, cfg)

  treated_rows <- dataset$data[[dataset$unit]] == dataset$treated_unit
  treated_actual <- dataset$data[treated_rows, dataset$outcome]
  time_values <- dataset$data[treated_rows, dataset$time]
  cf_metrics <- predict_cf_with_pkg("fastaugsynth", fit_metrics)
  cf_upstream <- predict_cf_with_pkg("augsynth", fit_upstream)

  t0 <- spec$pre_periods
  ttotal <- length(time_values)
  post_idx <- seq.int(t0 + 1L, ttotal)

  att_metrics <- sum_metrics$att
  att_upstream <- sum_upstream$att

  metrics_cf_lb <- rep(NA_real_, ttotal)
  metrics_cf_ub <- rep(NA_real_, ttotal)
  upstream_cf_lb <- rep(NA_real_, ttotal)
  upstream_cf_ub <- rep(NA_real_, ttotal)

  metrics_cf_lb[post_idx] <- treated_actual[post_idx] - att_metrics$upper_bound[post_idx]
  metrics_cf_ub[post_idx] <- treated_actual[post_idx] - att_metrics$lower_bound[post_idx]
  upstream_cf_lb[post_idx] <- treated_actual[post_idx] - att_upstream$upper_bound[post_idx]
  upstream_cf_ub[post_idx] <- treated_actual[post_idx] - att_upstream$lower_bound[post_idx]

  spec_label <- sprintf(
    "d=%d, pre=%d, post=%d, noise=%.2f",
    spec$donors, spec$pre_periods, spec$post_periods, spec$noise_sd
  )

  diagnostics <- data.frame(
    spec_id = spec_id,
    spec_label = spec_label,
    donors = spec$donors,
    pre_periods = spec$pre_periods,
    post_periods = spec$post_periods,
    noise_sd = spec$noise_sd,
    max_abs_synth_diff = max(abs(cf_metrics - cf_upstream)),
    max_abs_att_diff = max(abs(att_metrics$Estimate - att_upstream$Estimate)),
    max_abs_cf_lb_diff = max(abs(metrics_cf_lb[post_idx] - upstream_cf_lb[post_idx])),
    max_abs_cf_ub_diff = max(abs(metrics_cf_ub[post_idx] - upstream_cf_ub[post_idx])),
    stringsAsFactors = FALSE
  )

  list(
    time = time_values,
    treated = treated_actual,
    metrics_cf = cf_metrics,
    upstream_cf = cf_upstream,
    metrics_cf_lb = metrics_cf_lb,
    metrics_cf_ub = metrics_cf_ub,
    upstream_cf_lb = upstream_cf_lb,
    upstream_cf_ub = upstream_cf_ub,
    t_int = dataset$t_int,
    label = spec_label,
    diagnostics = diagnostics
  )
}

draw_panel <- function(panel_data, show_legend = FALSE) {
  y_values <- c(
    panel_data$treated,
    panel_data$metrics_cf,
    panel_data$upstream_cf,
    panel_data$metrics_cf_lb,
    panel_data$metrics_cf_ub,
    panel_data$upstream_cf_lb,
    panel_data$upstream_cf_ub
  )
  y_values <- y_values[is.finite(y_values)]
  y_range <- range(y_values)
  pad <- 0.05 * diff(y_range)
  if (!is.finite(pad) || pad <= 0) {
    pad <- 1.0
  }
  y_range <- c(y_range[1] - pad, y_range[2] + pad)

  plot(
    panel_data$time,
    panel_data$treated,
    type = "n",
    ylim = y_range,
    xlab = "Time",
    ylab = "Outcome",
    main = panel_data$label
  )

  post_idx <- panel_data$time >= panel_data$t_int
  post_time <- panel_data$time[post_idx]
  polygon(
    x = c(post_time, rev(post_time)),
    y = c(panel_data$metrics_cf_lb[post_idx], rev(panel_data$metrics_cf_ub[post_idx])),
    col = grDevices::adjustcolor("#1b9e77", alpha.f = 0.18),
    border = NA
  )

  abline(v = panel_data$t_int, col = "grey45", lty = 3, lwd = 1.5)
  lines(panel_data$time, panel_data$treated, col = "black", lwd = 2.2)
  lines(panel_data$time, panel_data$metrics_cf, col = "#1b9e77", lwd = 2.2)
  lines(panel_data$time, panel_data$upstream_cf, col = "#d95f02", lwd = 2, lty = 2)
  lines(post_time, panel_data$upstream_cf_lb[post_idx], col = "#d95f02", lwd = 1.5, lty = 3)
  lines(post_time, panel_data$upstream_cf_ub[post_idx], col = "#d95f02", lwd = 1.5, lty = 3)

  legend_text <- sprintf(
    "max|synth diff|=%.2e\nmax|CI diff|=%.2e",
    panel_data$diagnostics$max_abs_synth_diff,
    max(panel_data$diagnostics$max_abs_cf_lb_diff, panel_data$diagnostics$max_abs_cf_ub_diff)
  )
  usr <- par("usr")
  text(
    x = usr[1] + 0.02 * diff(usr[1:2]),
    y = usr[4] - 0.05 * diff(usr[3:4]),
    labels = legend_text,
    adj = c(0, 1),
    cex = 0.75
  )

  if (isTRUE(show_legend)) {
    legend(
      "bottomleft",
      legend = c(
        "Treated outcome",
        "fastaugsynth synth",
        "augsynth synth",
        "fastaugsynth CI",
        "augsynth CI bounds",
        "Treatment date"
      ),
      col = c("black", "#1b9e77", "#d95f02", "#1b9e77", "#d95f02", "grey45"),
      lty = c(1, 1, 2, 1, 3, 3),
      lwd = c(2.2, 2.2, 2, 8, 1.5, 1.5),
      pch = c(NA, NA, NA, 15, NA, NA),
      pt.cex = c(NA, NA, NA, 2, NA, NA),
      bty = "n",
      cex = 0.8
    )
  }
}

run_agreement_plot <- function(cfg) {
  ensure_runtime_deps(cfg)
  Sys.setenv(FASTAUGSYNTH_BACKEND_LIB = cfg$backend_lib)

  specs <- cfg$specs
  panels <- vector("list", nrow(specs))
  diagnostics <- vector("list", nrow(specs))

  for (idx in seq_len(nrow(specs))) {
    panel <- build_panel_data(specs[idx, ], cfg, spec_id = idx)
    panels[[idx]] <- panel
    diagnostics[[idx]] <- panel$diagnostics
  }

  diagnostics_df <- do.call(rbind, diagnostics)
  results_dir <- file.path(cfg$output_dir, "results_agreement")
  figures_dir <- file.path(cfg$output_dir, "figures_agreement")
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

  figure_path <- file.path(figures_dir, cfg$figure_name)
  png(figure_path, width = 2400, height = 2400, res = 180)
  old_par <- par(no.readonly = TRUE)
  on.exit({
    par(old_par)
    dev.off()
  }, add = TRUE)

  par(mfrow = c(3, 3), mar = c(4, 4, 3, 1) + 0.1, oma = c(0, 0, 3, 0))
  for (idx in seq_along(panels)) {
    draw_panel(panels[[idx]], show_legend = idx == 1L)
  }
  mtext(
    sprintf(
      "Augsynth agreement check (%s conformal CI): fastaugsynth vs upstream augsynth",
      paste(cfg$conformal_type, cfg$conformal_mode)
    ),
    side = 3,
    outer = TRUE,
    cex = 1.2,
    line = 1
  )

  write.csv(specs, file.path(results_dir, "augsynth_agreement_specs.csv"), row.names = FALSE)
  write.csv(diagnostics_df, file.path(results_dir, "augsynth_agreement_diagnostics.csv"), row.names = FALSE)

  list(
    figure = figure_path,
    diagnostics = file.path(results_dir, "augsynth_agreement_diagnostics.csv"),
    specs = file.path(results_dir, "augsynth_agreement_specs.csv")
  )
}

main <- function() {
  parsed <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  cfg <- parsed$config

  if (isTRUE(parsed$print_specs)) {
    cat("Agreement plot specs:\n")
    print(cfg$specs)
    return(invisible(cfg$specs))
  }

  out <- run_agreement_plot(cfg)
  cat("Agreement plot complete.\n")
  cat("Figure:\n")
  cat("  ", out$figure, "\n", sep = "")
  cat("Diagnostics:\n")
  cat("  ", out$diagnostics, "\n", sep = "")
}

if (sys.nframe() == 0L) {
  main()
}
