#!/usr/bin/env Rscript

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

source(file.path(script_dir(), "benchmark_augsynth_vs_gsynth.R"))

parse_cli_args <- function(args) {
  parsed <- list(
    output_dir = file.path(script_dir(), "readme_conformal"),
    figures_dir = NULL,
    backend_lib = backend_env_var(),
    reps = 1L,
    seed = 20260415L
  )

  idx <- 1L
  while (idx <= length(args)) {
    arg <- args[[idx]]
    if (arg == "--output-dir") {
      idx <- idx + 1L
      parsed$output_dir <- args[[idx]]
    } else if (arg == "--figures-dir") {
      idx <- idx + 1L
      parsed$figures_dir <- args[[idx]]
    } else if (arg == "--backend-lib") {
      idx <- idx + 1L
      parsed$backend_lib <- args[[idx]]
    } else if (arg == "--reps") {
      idx <- idx + 1L
      parsed$reps <- parse_int_scalar(args[[idx]], "reps")
    } else if (arg == "--seed") {
      idx <- idx + 1L
      parsed$seed <- parse_int_scalar(args[[idx]], "seed")
    } else {
      stop(sprintf("Unknown argument: %s", arg), call. = FALSE)
    }
    idx <- idx + 1L
  }

  parsed
}

time_repeated <- function(fun, reps) {
  times_sec <- numeric(reps)
  result <- NULL
  for (idx in seq_len(reps)) {
    gc(FALSE)
    elapsed <- system.time({
      result <- fun()
    })[["elapsed"]]
    times_sec[[idx]] <- as.numeric(elapsed)
  }
  list(times_sec = times_sec, result = result)
}

format_elapsed_label <- function(x) {
  if (x < 1) {
    sprintf("%.0f ms", x * 1000)
  } else if (x < 10) {
    sprintf("%.2f s", x)
  } else {
    sprintf("%.1f s", x)
  }
}

build_case_grid <- function() {
  data.frame(
    case_id = c("small", "medium", "large"),
    case_label = c(
      "Small\n20 donors, 40 pre, 20 post",
      "Medium\n40 donors, 80 pre, 40 post",
      "Large\n140 donors, 180 pre, 90 post"
    ),
    donors = c(20L, 40L, 140L),
    pre_periods = c(40L, 80L, 180L),
    post_periods = c(20L, 40L, 90L),
    stringsAsFactors = FALSE
  )
}

run_case <- function(case_row, reps, seed) {
  dataset <- simulate_augsynth_dataset(
    n_donors = case_row$donors,
    pre_periods = case_row$pre_periods,
    post_periods = case_row$post_periods,
    seed = seed,
    outcome_name = "gdpcap",
    unit_name = "regionno",
    time_name = "year"
  )

  formula <- as.formula(sprintf("%s ~ trt", dataset$outcome))
  unit_symbol <- as.name(as.character(dataset$unit))
  time_symbol <- as.name(as.character(dataset$time))
  panel <- dataset$data
  t_int <- dataset$t_int

  cat(sprintf(
    "[%s] fitting fastaugsynth and augsynth (%d donors, %d pre, %d post)\n",
    case_row$case_id,
    case_row$donors,
    case_row$pre_periods,
    case_row$post_periods
  ))
  flush.console()

  fast_fit_expr <- as.call(list(
    quote(fastaugsynth::augsynth),
    as.name("formula"),
    unit_symbol,
    time_symbol,
    as.name("panel"),
    progfunc = "None",
    scm = TRUE,
    t_int = as.name("t_int")
  ))
  upstream_fit_expr <- as.call(list(
    quote(augsynth::augsynth),
    as.name("formula"),
    unit_symbol,
    time_symbol,
    as.name("panel"),
    progfunc = "None",
    scm = TRUE,
    t_int = as.name("t_int")
  ))
  fast_fit <- eval(fast_fit_expr)
  upstream_fit <- eval(upstream_fit_expr)

  cat(sprintf("[%s] conformal summary: fastaugsynth\n", case_row$case_id))
  flush.console()
  fast_times <- time_repeated(
    function() summary_with_pkg("fastaugsynth", fast_fit, "conformal"),
    reps = reps
  )

  cat(sprintf("[%s] conformal summary: augsynth\n", case_row$case_id))
  flush.console()
  upstream_times <- time_repeated(
    function() summary_with_pkg("augsynth", upstream_fit, "conformal"),
    reps = reps
  )

  timing_rows <- rbind(
    data.frame(
      case_id = case_row$case_id,
      case_label = case_row$case_label,
      package = "fastaugsynth",
      rep = seq_along(fast_times$times_sec),
      elapsed_sec = fast_times$times_sec,
      donors = case_row$donors,
      pre_periods = case_row$pre_periods,
      post_periods = case_row$post_periods,
      stringsAsFactors = FALSE
    ),
    data.frame(
      case_id = case_row$case_id,
      case_label = case_row$case_label,
      package = "augsynth",
      rep = seq_along(upstream_times$times_sec),
      elapsed_sec = upstream_times$times_sec,
      donors = case_row$donors,
      pre_periods = case_row$pre_periods,
      post_periods = case_row$post_periods,
      stringsAsFactors = FALSE
    )
  )

  list(
    timings = timing_rows,
    fast_result = fast_times$result,
    upstream_result = upstream_times$result
  )
}

summarize_timings <- function(timings) {
  summary <- aggregate(
    elapsed_sec ~ case_id + case_label + package + donors + pre_periods + post_periods,
    data = timings,
    FUN = median
  )
  summary <- summary[order(match(summary$case_id, c("small", "medium", "large")), summary$package), ]
  summary$elapsed_ms <- summary$elapsed_sec * 1000
  summary$label <- vapply(summary$elapsed_sec, format_elapsed_label, character(1))

  speedups <- reshape(
    summary[, c("case_id", "package", "elapsed_sec")],
    idvar = "case_id",
    timevar = "package",
    direction = "wide"
  )
  names(speedups) <- sub("^elapsed_sec\\.", "", names(speedups))
  if (all(c("fastaugsynth", "augsynth") %in% names(speedups))) {
    speedups$speedup_vs_upstream <- speedups$augsynth / speedups$fastaugsynth
  }
  merge(summary, speedups[, c("case_id", "speedup_vs_upstream")], by = "case_id", all.x = TRUE)
}

plot_timings <- function(summary, figures_dir) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required to draw benchmark figures.", call. = FALSE)
  }

  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
  package_colors <- c(
    fastaugsynth = "#1b9e77",
    augsynth = "#d95f02"
  )

  plot_df <- summary
  plot_df$case_label <- factor(plot_df$case_label, levels = unique(build_case_grid()$case_label))
  plot_df$package <- factor(plot_df$package, levels = c("fastaugsynth", "augsynth"))

  fig_path <- file.path(figures_dir, "fastaugsynth_conformal_compare.png")
  plot_obj <- ggplot2::ggplot(
    plot_df,
    ggplot2::aes(x = case_label, y = elapsed_sec, color = package)
  ) +
    ggplot2::geom_line(
      ggplot2::aes(group = package),
      linewidth = 0.8,
      alpha = 0.65,
      show.legend = FALSE
    ) +
    ggplot2::geom_point(size = 4.2) +
    ggplot2::geom_text(
      ggplot2::aes(label = label),
      vjust = -0.35,
      size = 4.2,
      show.legend = FALSE
    ) +
    ggplot2::scale_color_manual(values = package_colors, name = "Package") +
    ggplot2::scale_y_log10(
      name = "Median elapsed per conformal summary (seconds, log scale)",
      breaks = c(0.01, 0.1, 1, 10, 100),
      labels = c("0.01", "0.1", "1", "10", "100")
    ) +
    ggplot2::labs(
      title = "Conformal summary timing: fastaugsynth vs upstream augsynth",
      subtitle = "Each point is a direct summary.augsynth() timing on the same pre-fit simulated panel"
    ) +
    ggplot2::coord_cartesian(clip = "off") +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      legend.position = "right",
      legend.box = "vertical",
      panel.grid.major.x = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(face = "bold"),
      plot.subtitle = ggplot2::element_text(size = 11),
      axis.title.x = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(face = "bold"),
      plot.margin = ggplot2::margin(10, 24, 10, 10)
    )

  ggplot2::ggsave(fig_path, plot_obj, width = 11, height = 6.5, dpi = 180)
  fig_path
}

run_readme_conformal_comparison <- function(output_dir, figures_dir, backend_lib, reps, seed) {
  if (!nzchar(backend_lib)) {
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

  Sys.setenv(FASTAUGSYNTH_BACKEND_LIB = backend_lib)

  output_dir <- normalizePath(output_dir, mustWork = FALSE)
  figures_dir <- if (nzchar(figures_dir %||% "")) {
    normalizePath(figures_dir, mustWork = FALSE)
  } else {
    file.path(output_dir, "figures")
  }
  results_dir <- file.path(output_dir, "results")
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

  cases <- build_case_grid()
  outputs <- lapply(seq_len(nrow(cases)), function(idx) {
    run_case(cases[idx, ], reps = reps, seed = seed + idx)
  })

  timings <- do.call(rbind, lapply(outputs, function(x) x$timings))
  summary <- summarize_timings(timings)

  summary_path <- file.path(results_dir, "fastaugsynth_conformal_compare_summary.csv")
  timings_path <- file.path(results_dir, "fastaugsynth_conformal_compare_timings.csv")
  write.csv(timings, timings_path, row.names = FALSE)
  write.csv(summary, summary_path, row.names = FALSE)

  figure_path <- plot_timings(summary, figures_dir)

  list(
    timings = timings,
    summary = summary,
    timings_path = timings_path,
    summary_path = summary_path,
    figure_path = figure_path,
    output_dir = output_dir
  )
}

main <- function() {
  cli <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  outputs <- run_readme_conformal_comparison(
    output_dir = cli$output_dir,
    figures_dir = cli$figures_dir,
    backend_lib = cli$backend_lib,
    reps = cli$reps,
    seed = cli$seed
  )

  cat("README conformal summary comparison:\n")
  print(outputs$summary)
  cat("\nWrote outputs to:\n")
  cat("  ", outputs$output_dir, "\n", sep = "")
  cat("  ", outputs$figure_path, "\n", sep = "")
}

if (sys.nframe() == 0L) {
  main()
}
