#!/usr/bin/env Rscript

# This benchmark is intentionally explicit and easy to review:
# - one 2 x 2 x 2 x 2 factorial grid
# - estimation, jackknife, and conformal are timed separately
# - one warmup call happens before measured reps
# - outputs are plain CSV files plus simple base-R bar charts

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

default_config <- function() {
  list(
    output_dir = script_dir(),
    figures_dir = NULL,
    backend_lib = backend_env_var(),
    fast_r_package = "",
    phase = "all",
    reps = 3L,
    seed = 20260412L,
    print_specs = FALSE,
    conformal_type = "iid",
    conformal_ns = 1000L,
    conformal_grid_size = 50L,
    donors = c(10L, 80L),
    pre_periods = c(20L, 120L),
    post_periods = c(20L, 50L),
    noise_sd = c(0.5, 1.5)
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

  idx <- 1L
  while (idx <= length(args)) {
    arg <- args[[idx]]
    if (arg == "--output-dir") {
      idx <- idx + 1L
      cfg$output_dir <- args[[idx]]
    } else if (arg == "--figures-dir") {
      idx <- idx + 1L
      cfg$figures_dir <- args[[idx]]
    } else if (arg == "--backend-lib") {
      idx <- idx + 1L
      cfg$backend_lib <- args[[idx]]
    } else if (arg == "--fast-r-package") {
      idx <- idx + 1L
      cfg$fast_r_package <- args[[idx]]
    } else if (arg == "--phase") {
      idx <- idx + 1L
      cfg$phase <- args[[idx]]
    } else if (arg == "--reps") {
      idx <- idx + 1L
      cfg$reps <- parse_int_scalar(args[[idx]], "reps")
    } else if (arg == "--seed") {
      idx <- idx + 1L
      cfg$seed <- parse_int_scalar(args[[idx]], "seed", min_value = 0L)
    } else if (arg == "--conformal-type") {
      idx <- idx + 1L
      cfg$conformal_type <- args[[idx]]
    } else if (arg == "--conformal-ns") {
      idx <- idx + 1L
      cfg$conformal_ns <- parse_int_scalar(args[[idx]], "conformal-ns")
    } else if (arg == "--conformal-grid-size") {
      idx <- idx + 1L
      cfg$conformal_grid_size <- parse_int_scalar(args[[idx]], "conformal-grid-size")
    } else if (arg == "--print-specs") {
      cfg$print_specs <- TRUE
    } else {
      stop(sprintf("Unknown argument: %s", arg), call. = FALSE)
    }

    idx <- idx + 1L
  }

  if (!cfg$conformal_type %in% c("iid", "block")) {
    stop("--conformal-type must be either 'iid' or 'block'", call. = FALSE)
  }
  if (!cfg$phase %in% c("all", "estimate", "jackknife", "conformal")) {
    stop("--phase must be one of: all, estimate, jackknife, conformal", call. = FALSE)
  }

  cfg
}

progress_log <- function(...) {
  cat(sprintf(...), "\n")
  flush.console()
}

build_spec_grid <- function(cfg) {
  grid <- expand.grid(
    donors = as.integer(cfg$donors),
    pre_periods = as.integer(cfg$pre_periods),
    post_periods = as.integer(cfg$post_periods),
    noise_sd = as.numeric(cfg$noise_sd),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )

  grid <- grid[order(grid$donors, grid$pre_periods, grid$post_periods, grid$noise_sd), ]
  rownames(grid) <- NULL
  grid$spec_id <- seq_len(nrow(grid))
  grid$spec_label <- sprintf(
    "d=%d | pre=%d | post=%d | noise=%.1f",
    grid$donors,
    grid$pre_periods,
    grid$post_periods,
    grid$noise_sd
  )
  grid
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
    t_int = t_int
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

summary_with_pkg <- function(pkg, fit, inf_type, cfg) {
  summary_fun <- ns_fun(pkg, "summary.augsynth")
  summary_fun(
    fit,
    inf = TRUE,
    inf_type = inf_type,
    type = cfg$conformal_type,
    ns = cfg$conformal_ns,
    grid_size = cfg$conformal_grid_size,
    q = 1
  )
}

predict_att_with_pkg <- function(pkg, fit) {
  predict_fun <- ns_fun(pkg, "predict.augsynth")
  predict_fun(fit, att = TRUE)
}

time_repeated <- function(fun, reps) {
  warm_result <- fun()
  if (reps < 1L) {
    return(list(times_ms = numeric(0), result = warm_result))
  }

  times_ms <- numeric(reps)
  last_result <- warm_result
  for (idx in seq_len(reps)) {
    invisible(gc(FALSE))
    elapsed <- system.time({
      last_result <- fun()
    })[["elapsed"]]
    times_ms[[idx]] <- as.numeric(elapsed) * 1000
  }

  list(times_ms = times_ms, result = last_result)
}

benchmark_phase <- function(spec, dataset, cfg, phase) {
  formula <- as.formula(sprintf("%s ~ trt", dataset$outcome))
  panel <- dataset$data
  unit_name <- dataset$unit
  time_name <- dataset$time
  t_int <- dataset$t_int
  fast_r_package <- trimws(cfg$fast_r_package %||% "")
  progress_prefix <- sprintf("[spec %d] %s", spec$spec_id, spec$spec_label)

  time_or_fit <- function(pkg, mode_label, timed) {
    progress_log("%s %s: %s", progress_prefix, mode_label, pkg)
    if (timed) {
      out <- time_repeated(
        function() fit_with_pkg(pkg, formula, unit_name, time_name, panel, t_int),
        cfg$reps
      )
      progress_log(
        "%s %s done: %s median %.2f ms",
        progress_prefix,
        mode_label,
        pkg,
        median(out$times_ms)
      )
      return(out)
    }

    fit <- fit_with_pkg(pkg, formula, unit_name, time_name, panel, t_int)
    progress_log("%s %s done: %s", progress_prefix, mode_label, pkg)
    list(times_ms = numeric(0), result = fit)
  }

  timed_estimate <- phase %in% c("estimate", "all")
  estimate_metrics <- time_or_fit("fastaugsynth", "fit", timed_estimate)
  estimate_upstream <- time_or_fit("augsynth", "fit", timed_estimate)

  metrics_fit <- estimate_metrics$result
  upstream_fit <- estimate_upstream$result

  t0 <- ncol(metrics_fit$data$X)
  post_idx <- seq.int(t0 + 1L, length(predict_att_with_pkg("fastaugsynth", metrics_fit)))
  estimate_gap <- max(abs(
    predict_att_with_pkg("fastaugsynth", metrics_fit)[post_idx] -
      predict_att_with_pkg("augsynth", upstream_fit)[post_idx]
  ))

  phase_times <- list(
    estimate = list(
      fastaugsynth = estimate_metrics$times_ms,
      augsynth = estimate_upstream$times_ms
    )
  )

  diagnostics <- data.frame(
    spec_id = spec$spec_id,
    spec_label = spec$spec_label,
    donors = spec$donors,
    pre_periods = spec$pre_periods,
    post_periods = spec$post_periods,
    noise_sd = spec$noise_sd,
    max_abs_post_att_diff = estimate_gap,
    max_abs_fast_r_conformal_lb_diff = NA_real_,
    max_abs_fast_r_conformal_ub_diff = NA_real_,
    max_abs_fast_r_conformal_p_diff = NA_real_,
    stringsAsFactors = FALSE
  )

  if (phase %in% c("jackknife", "all")) {
    progress_log("%s jackknife: fastaugsynth", progress_prefix)
    jack_metrics <- time_repeated(
      function() summary_with_pkg("fastaugsynth", metrics_fit, "jackknife", cfg),
      cfg$reps
    )
    progress_log("%s jackknife done: fastaugsynth median %.2f ms", progress_prefix, median(jack_metrics$times_ms))
    progress_log("%s jackknife: augsynth", progress_prefix)
    jack_upstream <- time_repeated(
      function() summary_with_pkg("augsynth", upstream_fit, "jackknife", cfg),
      cfg$reps
    )
    progress_log("%s jackknife done: augsynth median %.2f ms", progress_prefix, median(jack_upstream$times_ms))
    phase_times$jackknife <- list(
      fastaugsynth = jack_metrics$times_ms,
      augsynth = jack_upstream$times_ms
    )
  }

  if (phase %in% c("conformal", "all")) {
    fast_r_fit <- NULL
    if (nzchar(fast_r_package)) {
      fast_r_fit <- time_or_fit(fast_r_package, "fit", FALSE)$result
    }
    progress_log("%s conformal: fastaugsynth", progress_prefix)
    conformal_metrics <- time_repeated(
      function() summary_with_pkg("fastaugsynth", metrics_fit, "conformal", cfg),
      cfg$reps
    )
    progress_log("%s conformal done: fastaugsynth median %.2f ms", progress_prefix, median(conformal_metrics$times_ms))
    progress_log("%s conformal: augsynth", progress_prefix)
    conformal_upstream <- time_repeated(
      function() summary_with_pkg("augsynth", upstream_fit, "conformal", cfg),
      cfg$reps
    )
    progress_log("%s conformal done: augsynth median %.2f ms", progress_prefix, median(conformal_upstream$times_ms))
    phase_times$conformal <- list(
      fastaugsynth = conformal_metrics$times_ms,
      augsynth = conformal_upstream$times_ms
    )
    if (nzchar(fast_r_package)) {
      progress_log("%s conformal: %s", progress_prefix, fast_r_package)
      conformal_fast_r <- time_repeated(
        function() summary_with_pkg(fast_r_package, fast_r_fit, "conformal", cfg),
        cfg$reps
      )
      progress_log(
        "%s conformal done: %s median %.2f ms",
        progress_prefix,
        fast_r_package,
        median(conformal_fast_r$times_ms)
      )
      phase_times$conformal[[fast_r_package]] <- conformal_fast_r$times_ms

      upstream_att <- conformal_upstream$result$att
      fast_r_att <- conformal_fast_r$result$att
      post_rows <- seq.int(t0 + 1L, nrow(upstream_att))
      diagnostics$max_abs_fast_r_conformal_lb_diff <- max(abs(
        upstream_att$lower_bound[post_rows] - fast_r_att$lower_bound[post_rows]
      ))
      diagnostics$max_abs_fast_r_conformal_ub_diff <- max(abs(
        upstream_att$upper_bound[post_rows] - fast_r_att$upper_bound[post_rows]
      ))
      diagnostics$max_abs_fast_r_conformal_p_diff <- max(abs(
        upstream_att$p_val[post_rows] - fast_r_att$p_val[post_rows]
      ))
    }
  }

  list(times = phase_times, diagnostics = diagnostics)
}

flatten_timings <- function(spec, phase_name, method_name, values) {
  if (!length(values)) {
    return(NULL)
  }

  data.frame(
    spec_id = spec$spec_id,
    spec_label = spec$spec_label,
    donors = spec$donors,
    pre_periods = spec$pre_periods,
    post_periods = spec$post_periods,
    noise_sd = spec$noise_sd,
    phase = phase_name,
    method = method_name,
    rep = seq_along(values),
    elapsed_ms = as.numeric(values),
    stringsAsFactors = FALSE
  )
}

summarize_timings <- function(timings) {
  grouped <- split(timings, list(timings$spec_id, timings$phase, timings$method), drop = TRUE)
  rows <- lapply(grouped, function(df) {
    template <- df[1L, c("spec_id", "spec_label", "donors", "pre_periods", "post_periods", "noise_sd", "phase", "method")]
    transform(
      template,
      median_ms = median(df$elapsed_ms),
      min_ms = min(df$elapsed_ms),
      max_ms = max(df$elapsed_ms)
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out <- out[order(out$phase, out$spec_id, out$method), ]

  out$speedup_vs_fastaugsynth <- NA_real_
  for (phase_name in unique(out$phase)) {
    phase_rows <- out$phase == phase_name
    phase_df <- out[phase_rows, ]
    for (spec_id in unique(phase_df$spec_id)) {
      spec_rows <- phase_rows & out$spec_id == spec_id
      base_rows <- spec_rows & out$method == "fastaugsynth"
      if (!any(base_rows)) {
        next
      }
      base_time <- out$median_ms[base_rows][[1]]
      out$speedup_vs_fastaugsynth[spec_rows] <- out$median_ms[spec_rows] / base_time
    }
  }

  out
}

plot_phase_bars <- function(summary, phase_name, output_path) {
  phase_df <- summary[summary$phase == phase_name, ]
  if (!nrow(phase_df)) {
    return(invisible(NULL))
  }

  method_priority <- c("fastaugsynth", "augsynth", "augsynthfast")
  methods <- unique(phase_df$method)
  methods <- c(
    method_priority[method_priority %in% methods],
    sort(setdiff(methods, method_priority))
  )
  spec_labels <- unique(phase_df$spec_label)
  bar_matrix <- vapply(
    methods,
    function(method_name) {
      vapply(
        spec_labels,
        function(label) {
          idx <- phase_df$spec_label == label & phase_df$method == method_name
          if (!any(idx)) {
            return(NA_real_)
          }
          phase_df$median_ms[idx][[1]]
        },
        numeric(1)
      )
    },
    numeric(length(spec_labels))
  )

  fill_colors <- c(
    fastaugsynth = "#1b9e77",
    augsynth = "#d95f02",
    augsynthfast = "#7570b3"
  )
  method_colors <- fill_colors[methods]
  missing_colors <- is.na(method_colors)
  if (any(missing_colors)) {
    method_colors[missing_colors] <- grDevices::rainbow(sum(missing_colors))
  }

  png(output_path, width = 2200, height = 1100, res = 160)
  old_par <- par(no.readonly = TRUE)
  on.exit({
    par(old_par)
    dev.off()
  }, add = TRUE)

  par(mar = c(12, 5, 4, 1) + 0.1)
  bar_pos <- barplot(
    t(bar_matrix),
    beside = TRUE,
    col = unname(method_colors),
    xaxt = "n",
    ylab = "Median elapsed per call (ms)",
    main = sprintf("Augsynth factorial benchmark: %s", phase_name)
  )
  tick_pos <- if (is.null(dim(bar_pos))) {
    mean(bar_pos)
  } else if (ncol(bar_pos) == length(spec_labels)) {
    colMeans(bar_pos)
  } else if (nrow(bar_pos) == length(spec_labels)) {
    rowMeans(bar_pos)
  } else {
    seq_along(spec_labels)
  }
  axis(
    side = 1,
    at = tick_pos,
    labels = spec_labels,
    las = 2,
    cex.axis = 0.75
  )
  legend(
    "topleft",
    legend = methods,
    fill = unname(method_colors),
    bty = "n"
  )
}

ensure_runtime_deps <- function(cfg) {
  if (!nzchar(cfg$backend_lib) || !file.exists(cfg$backend_lib)) {
    stop(
      "Set FASTAUGSYNTH_BACKEND_LIB or pass --backend-lib so fastaugsynth can find the compiled backend library.",
      call. = FALSE
    )
  }
  if (!requireNamespace("fastaugsynth", quietly = TRUE)) {
    stop("Package 'fastaugsynth' must be installed before running this benchmark.", call. = FALSE)
  }
  if (!requireNamespace("augsynth", quietly = TRUE)) {
    stop("Package 'augsynth' must be installed before running this benchmark.", call. = FALSE)
  }
  if (nzchar(trimws(cfg$fast_r_package %||% "")) &&
      !requireNamespace(trimws(cfg$fast_r_package), quietly = TRUE)) {
    stop(
      sprintf(
        "Package '%s' must be installed before running this benchmark.",
        trimws(cfg$fast_r_package)
      ),
      call. = FALSE
    )
  }
}

run_factorial_benchmark <- function(cfg) {
  ensure_runtime_deps(cfg)
  Sys.setenv(FASTAUGSYNTH_BACKEND_LIB = cfg$backend_lib)

  specs <- build_spec_grid(cfg)
  timings <- list()
  diagnostics <- list()

  for (idx in seq_len(nrow(specs))) {
    spec <- specs[idx, ]
    progress_log("== spec %d/%d: %s ==", idx, nrow(specs), spec$spec_label)
    dataset <- simulate_panel(spec, seed = cfg$seed + spec$spec_id)
    out <- benchmark_phase(spec, dataset, cfg, phase = cfg$phase)
    diagnostics[[idx]] <- out$diagnostics

    for (phase_name in names(out$times)) {
      for (method_name in names(out$times[[phase_name]])) {
        timings[[length(timings) + 1L]] <- flatten_timings(
          spec,
          phase_name,
          method_name,
          out$times[[phase_name]][[method_name]]
        )
      }
    }
  }

  timings_df <- do.call(rbind, timings)
  diagnostics_df <- do.call(rbind, diagnostics)
  summary_df <- summarize_timings(timings_df)

  results_dir <- file.path(cfg$output_dir, "results_factorial")
  figures_dir <- cfg$figures_dir %||% file.path(cfg$output_dir, "figures_factorial")
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

  write.csv(specs, file.path(results_dir, "augsynth_factorial_specs.csv"), row.names = FALSE)
  write.csv(timings_df, file.path(results_dir, "augsynth_factorial_timings.csv"), row.names = FALSE)
  write.csv(summary_df, file.path(results_dir, "augsynth_factorial_summary.csv"), row.names = FALSE)
  write.csv(diagnostics_df, file.path(results_dir, "augsynth_factorial_diagnostics.csv"), row.names = FALSE)

  if ("estimate" %in% unique(summary_df$phase)) {
    plot_phase_bars(summary_df, "estimate", file.path(figures_dir, "augsynth_factorial_estimate_bar.png"))
  }
  if ("jackknife" %in% unique(summary_df$phase)) {
    plot_phase_bars(summary_df, "jackknife", file.path(figures_dir, "augsynth_factorial_jackknife_bar.png"))
  }
  if ("conformal" %in% unique(summary_df$phase)) {
    plot_phase_bars(summary_df, "conformal", file.path(figures_dir, "augsynth_factorial_conformal_bar.png"))
  }

  list(
    specs = specs,
    timings = timings_df,
    summary = summary_df,
    diagnostics = diagnostics_df,
    results_dir = results_dir,
    figures_dir = figures_dir
  )
}

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs)) rhs else lhs
}

main <- function() {
  cfg <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  specs <- build_spec_grid(cfg)

  if (isTRUE(cfg$print_specs)) {
    cat("Factorial benchmark specs:\n")
    print(specs[, c("spec_id", "donors", "pre_periods", "post_periods", "noise_sd", "spec_label")])
    return(invisible(specs))
  }

  outputs <- run_factorial_benchmark(cfg)
  cat("Factorial benchmark complete.\n")
  cat("Results:\n")
  cat("  ", outputs$results_dir, "\n", sep = "")
  cat("Figures:\n")
  cat("  ", outputs$figures_dir, "\n", sep = "")
}

if (sys.nframe() == 0L) {
  main()
}
