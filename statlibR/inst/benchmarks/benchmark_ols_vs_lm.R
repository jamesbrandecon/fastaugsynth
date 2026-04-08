#!/usr/bin/env Rscript

default_scenarios <- function() {
  data.frame(
    scenario = c("n=500, p=8", "n=1000, p=8", "n=5000, p=8", "n=10000, p=8"),
    n = c(500L, 1000L, 5000L, 10000L),
    p = c(8L, 8L, 8L, 8L),
    stringsAsFactors = FALSE
  )
}

backend_env_var <- function() {
  Sys.getenv("METRICSJL_BACKEND_LIB", Sys.getenv("STATLIB_BACKEND_LIB", ""))
}

benchmark_method_catalog <- function() {
  list(
    jols_fit_xy = list(label = "metricsjl::jols_fit_xy()", color = "#1b9e77", pch = 16),
    lm = list(label = "stats::lm()", color = "#d95f02", pch = 17),
    feols = list(label = "fixest::feols()", color = "#7570b3", pch = 15)
  )
}

script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_flag <- "--file="
  file_arg <- grep(paste0("^", file_flag), args, value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(sub(file_flag, "", file_arg[[1]]), mustWork = FALSE)))
  }

  frame_files <- vapply(sys.frames(), function(frame) {
    if (is.null(frame$ofile)) NA_character_ else frame$ofile
  }, character(1))
  frame_files <- frame_files[!is.na(frame_files)]
  if (length(frame_files)) {
    return(dirname(normalizePath(frame_files[[length(frame_files)]], mustWork = FALSE)))
  }

  normalizePath(getwd(), mustWork = FALSE)
}

timed_eval <- function(expr_fn, iterations = 1L) {
  stopifnot(iterations >= 1L)

  gc(FALSE)
  value <- NULL
  elapsed <- system.time({
    for (idx in seq_len(iterations)) {
      value <- expr_fn()
    }
  })[["elapsed"]]

  list(
    elapsed_sec = unname(elapsed) / iterations,
    value = value
  )
}

build_case <- function(n, p, seed) {
  set.seed(seed)

  z <- matrix(rnorm(n * p), nrow = n, ncol = p)
  colnames(z) <- paste0("x", seq_len(p))
  beta <- seq(0.2, by = 0.15, length.out = p)
  y <- as.vector(1.0 + z %*% beta + rnorm(n, sd = 0.25))

  data <- data.frame(y = y, z, check.names = FALSE)
  formula <- as.formula(paste("y ~", paste(colnames(z), collapse = " + ")))
  x <- cbind("(Intercept)" = 1.0, z)

  list(
    x = x,
    y = y,
    data = data,
    formula = formula
  )
}

case_methods <- function(case) {
  list(
    jols_fit_xy = list(
      fit = function() metricsjl::jols_fit_xy(case$x, case$y),
      coef = function(value) unname(value$coefficients)
    ),
    lm = list(
      fit = function() stats::lm(case$formula, data = case$data),
      coef = function(value) unname(stats::coef(value))
    ),
    feols = list(
      fit = function() fixest::feols(case$formula, data = case$data),
      coef = function(value) unname(stats::coef(value))
    )
  )
}

validate_against_lm <- function(method_id, coef, ref_coef, scenario_name, tolerance) {
  if (length(coef) != length(ref_coef)) {
    stop(
      sprintf(
        "Coefficient length mismatch for %s in %s: got %d vs expected %d",
        method_id,
        scenario_name,
        length(coef),
        length(ref_coef)
      ),
      call. = FALSE
    )
  }

  max_abs_coef_diff <- max(abs(coef - ref_coef))
  if (!is.finite(max_abs_coef_diff) || max_abs_coef_diff > tolerance) {
    stop(
      sprintf(
        "Coefficient mismatch for %s in %s: max abs diff %.3e exceeds tolerance %.3e",
        method_id,
        scenario_name,
        max_abs_coef_diff,
        tolerance
      ),
      call. = FALSE
    )
  }

  max_abs_coef_diff
}

run_case_benchmark <- function(scenario_row, reps, batch_size, seed, tolerance) {
  scenario_name <- scenario_row[["scenario"]]
  n <- as.integer(scenario_row[["n"]])
  p <- as.integer(scenario_row[["p"]])
  case <- build_case(n, p, seed)
  methods <- case_methods(case)
  method_order <- names(benchmark_method_catalog())

  cold <- lapply(method_order, function(method_id) timed_eval(methods[[method_id]]$fit))
  names(cold) <- method_order

  lm_coef <- methods[["lm"]]$coef(cold[["lm"]]$value)
  max_diffs <- vapply(method_order, function(method_id) {
    validate_against_lm(
      method_id = method_id,
      coef = methods[[method_id]]$coef(cold[[method_id]]$value),
      ref_coef = lm_coef,
      scenario_name = scenario_name,
      tolerance = tolerance
    )
  }, numeric(1))

  warm <- lapply(method_order, function(method_id) {
    vapply(seq_len(reps), function(idx) {
      timed_eval(methods[[method_id]]$fit, iterations = batch_size)[["elapsed_sec"]]
    }, numeric(1))
  })
  names(warm) <- method_order

  results <- do.call(rbind, lapply(method_order, function(method_id) {
    rbind(
      data.frame(
        scenario = scenario_name,
        n = n,
        p = p,
        phase = "cold",
        method = method_id,
        iteration = 1L,
        elapsed_sec = cold[[method_id]][["elapsed_sec"]],
        max_abs_coef_diff_vs_lm = max_diffs[[method_id]],
        stringsAsFactors = FALSE
      ),
      data.frame(
        scenario = scenario_name,
        n = n,
        p = p,
        phase = "warm",
        method = method_id,
        iteration = seq_len(reps),
        elapsed_sec = warm[[method_id]],
        max_abs_coef_diff_vs_lm = max_diffs[[method_id]],
        stringsAsFactors = FALSE
      )
    )
  }))

  results
}

summarize_results <- function(results) {
  split_results <- split(
    results,
    list(results$scenario, results$phase, results$method),
    drop = TRUE
  )

  summary_rows <- lapply(split_results, function(df) {
    data.frame(
      scenario = df$scenario[[1]],
      n = df$n[[1]],
      p = df$p[[1]],
      phase = df$phase[[1]],
      method = df$method[[1]],
      runs = nrow(df),
      median_sec = stats::median(df$elapsed_sec),
      mean_sec = mean(df$elapsed_sec),
      min_sec = min(df$elapsed_sec),
      max_sec = max(df$elapsed_sec),
      sd_sec = if (nrow(df) > 1L) stats::sd(df$elapsed_sec) else NA_real_,
      max_abs_coef_diff_vs_lm = df$max_abs_coef_diff_vs_lm[[1]],
      stringsAsFactors = FALSE
    )
  })

  summary <- do.call(rbind, summary_rows)
  rownames(summary) <- NULL
  summary <- summary[order(summary$phase, summary$n, summary$method), ]

  warm_summary <- summary[summary$phase == "warm", c("scenario", "n", "p", "method", "median_sec")]
  speedup <- reshape(warm_summary, idvar = c("scenario", "n", "p"), timevar = "method", direction = "wide")
  names(speedup) <- sub("^median_sec\\.", "", names(speedup))
  speedup <- speedup[order(speedup$n), ]
  speedup$lm_over_jols_fit_xy <- speedup$lm / speedup$jols_fit_xy
  speedup$feols_over_jols_fit_xy <- speedup$feols / speedup$jols_fit_xy
  speedup <- speedup[, c(
    "scenario", "n", "p",
    "jols_fit_xy", "lm", "feols",
    "lm_over_jols_fit_xy", "feols_over_jols_fit_xy"
  )]
  names(speedup)[4:8] <- c(
    "jols_fit_xy_median_sec",
    "lm_median_sec",
    "feols_median_sec",
    "lm_over_jols_fit_xy",
    "feols_over_jols_fit_xy"
  )

  list(summary = summary, speedup = speedup)
}

plot_results <- function(summary, output_dir) {
  figures_dir <- file.path(output_dir, "figures")
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

  catalog <- benchmark_method_catalog()
  method_order <- names(catalog)
  warm <- summary[summary$phase == "warm", ]
  warm <- warm[order(warm$n, warm$method), ]
  scenarios <- unique(warm$scenario)

  bar_matrix <- do.call(rbind, lapply(method_order, function(method_id) {
    vapply(scenarios, function(scenario_name) {
      1000 * warm$median_sec[warm$method == method_id & warm$scenario == scenario_name]
    }, numeric(1))
  }))
  colnames(bar_matrix) <- scenarios
  rownames(bar_matrix) <- vapply(method_order, function(method_id) catalog[[method_id]]$label, character(1))

  method_colors <- vapply(method_order, function(method_id) catalog[[method_id]]$color, character(1))
  method_pch <- vapply(method_order, function(method_id) catalog[[method_id]]$pch, numeric(1))

  bar_path <- file.path(figures_dir, "ols_vs_lm_bar.png")
  png(bar_path, width = 1400, height = 800, res = 160)
  old_par <- par(no.readonly = TRUE)
  on.exit({
    par(old_par)
    dev.off()
  }, add = TRUE)

  par(mar = c(7, 5, 4, 1) + 0.1)
  barplot(
    bar_matrix,
    beside = TRUE,
    col = method_colors,
    ylab = "Median elapsed per call (ms)",
    main = "Warm-call runtime by scenario",
    las = 2
  )
  legend(
    "topleft",
    legend = rownames(bar_matrix),
    fill = method_colors,
    bty = "n"
  )
  dev.off()
  on.exit(NULL, add = FALSE)

  line_path <- file.path(figures_dir, "ols_vs_lm_line.png")
  png(line_path, width = 1400, height = 800, res = 160)
  old_par <- par(no.readonly = TRUE)
  on.exit({
    par(old_par)
    dev.off()
  }, add = TRUE)

  par(mar = c(5, 5, 4, 1) + 0.1)
  y_limits <- range(warm$median_sec, finite = TRUE) * 1000
  plot(
    numeric(0),
    numeric(0),
    xlim = range(warm$n),
    ylim = y_limits,
    xlab = "Observations (n)",
    ylab = "Median elapsed per call (ms)",
    main = "Warm-call runtime scaling"
  )
  for (idx in seq_along(method_order)) {
    method_id <- method_order[[idx]]
    method_rows <- warm[warm$method == method_id, ]
    lines(
      method_rows$n,
      method_rows$median_sec * 1000,
      type = "b",
      pch = method_pch[[idx]],
      lwd = 2,
      col = method_colors[[idx]]
    )
  }
  legend(
    "topleft",
    legend = vapply(method_order, function(method_id) catalog[[method_id]]$label, character(1)),
    col = method_colors,
    pch = method_pch,
    lwd = 2,
    bty = "n"
  )
  dev.off()
  on.exit(NULL, add = FALSE)

  c(bar = bar_path, line = line_path)
}

write_metadata <- function(output_dir, backend_lib, reps, batch_size) {
  meta_path <- file.path(output_dir, "results", "benchmark_metadata.txt")
  dir.create(dirname(meta_path), recursive = TRUE, showWarnings = FALSE)

  lines <- c(
    sprintf("generated_at: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    sprintf("backend_lib: %s", backend_lib),
    sprintf("reps: %d", reps),
    sprintf("batch_size: %d", batch_size),
    sprintf("R.version: %s", R.version.string),
    sprintf("platform: %s", R.version$platform),
    sprintf("metricsjl.version: %s", as.character(utils::packageVersion("metricsjl"))),
    sprintf("fixest.version: %s", as.character(utils::packageVersion("fixest")))
  )
  writeLines(lines, meta_path)
  meta_path
}

run_ols_vs_lm_benchmarks <- function(output_dir = script_dir(),
                                     backend_lib = backend_env_var(),
                                     reps = 12L,
                                     batch_size = 5L,
                                     seed = 20260407L,
                                     tolerance = 1e-8) {
  if (!nzchar(backend_lib)) {
    stop(
      "Set METRICSJL_BACKEND_LIB or pass backend_lib so jols_fit_xy() can find the compiled backend library.",
      call. = FALSE
    )
  }

  if (!requireNamespace("metricsjl", quietly = TRUE)) {
    stop("Package 'metricsjl' must be installed before running the benchmark.", call. = FALSE)
  }
  if (!requireNamespace("fixest", quietly = TRUE)) {
    stop("Package 'fixest' must be installed before running the benchmark.", call. = FALSE)
  }

  Sys.setenv(METRICSJL_BACKEND_LIB = backend_lib)

  scenarios <- default_scenarios()
  results <- do.call(rbind, lapply(seq_len(nrow(scenarios)), function(idx) {
    run_case_benchmark(
      scenario_row = scenarios[idx, , drop = FALSE],
      reps = reps,
      batch_size = batch_size,
      seed = seed + idx,
      tolerance = tolerance
    )
  }))

  output_dir <- normalizePath(output_dir, mustWork = FALSE)
  results_dir <- file.path(output_dir, "results")
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  summary_parts <- summarize_results(results)
  plots <- plot_results(summary_parts$summary, output_dir)
  meta_path <- write_metadata(output_dir, backend_lib, reps, batch_size)

  write.csv(results, file.path(results_dir, "ols_vs_lm_timings.csv"), row.names = FALSE)
  write.csv(summary_parts$summary, file.path(results_dir, "ols_vs_lm_summary.csv"), row.names = FALSE)
  write.csv(summary_parts$speedup, file.path(results_dir, "ols_vs_lm_speedup.csv"), row.names = FALSE)

  list(
    results = results,
    summary = summary_parts$summary,
    speedup = summary_parts$speedup,
    figures = plots,
    metadata = meta_path,
    output_dir = output_dir
  )
}

parse_cli_args <- function(args) {
  parsed <- list(
    output_dir = script_dir(),
    backend_lib = backend_env_var(),
    reps = 12L,
    batch_size = 5L
  )

  idx <- 1L
  while (idx <= length(args)) {
    arg <- args[[idx]]

    if (arg == "--output-dir") {
      idx <- idx + 1L
      parsed$output_dir <- args[[idx]]
    } else if (arg == "--backend-lib") {
      idx <- idx + 1L
      parsed$backend_lib <- args[[idx]]
    } else if (arg == "--reps") {
      idx <- idx + 1L
      parsed$reps <- as.integer(args[[idx]])
    } else if (arg == "--batch-size") {
      idx <- idx + 1L
      parsed$batch_size <- as.integer(args[[idx]])
    } else {
      stop(sprintf("Unknown argument: %s", arg), call. = FALSE)
    }

    idx <- idx + 1L
  }

  parsed
}

main <- function() {
  cli <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  outputs <- run_ols_vs_lm_benchmarks(
    output_dir = cli$output_dir,
    backend_lib = cli$backend_lib,
    reps = cli$reps,
    batch_size = cli$batch_size
  )

  print(outputs$speedup)
  cat("\nWrote benchmark outputs to:\n")
  cat("  ", outputs$output_dir, "\n", sep = "")
}

if (sys.nframe() == 0L) {
  main()
}
