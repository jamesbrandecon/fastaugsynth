#!/usr/bin/env Rscript

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

  frame_files <- vapply(sys.frames(), function(frame) {
    if (is.null(frame$ofile)) NA_character_ else frame$ofile
  }, character(1))
  frame_files <- frame_files[!is.na(frame_files)]
  if (length(frame_files)) {
    return(dirname(normalizePath(frame_files[[length(frame_files)]], mustWork = FALSE)))
  }

  normalizePath(getwd(), mustWork = FALSE)
}

parse_int_scalar <- function(value, arg_name, min_value = 1L) {
  if (is.null(value)) {
    stop(sprintf("Missing value for --%s", arg_name), call. = FALSE)
  }
  out <- suppressWarnings(as.integer(value))
  if (length(out) != 1L || is.na(out) || out < min_value) {
    stop(sprintf("Invalid --%s value: %s", arg_name, value), call. = FALSE)
  }
  out
}

parse_int_vector <- function(value, arg_name) {
  values <- unlist(strsplit(value, "[, ]+"), use.names = FALSE)
  values <- suppressWarnings(as.integer(values))
  values <- values[!is.na(values)]
  values <- unique(values[values > 0L])
  if (!length(values)) {
    stop(sprintf("Invalid --%s values: %s", arg_name, value), call. = FALSE)
  }
  sort(values)
}

parse_inference_modes <- function(value) {
  if (is.null(value)) {
    return(character(0))
  }
  modes <- unique(unlist(strsplit(value, "[, ]+"), use.names = FALSE))
  modes <- modes[nzchar(modes)]
  modes <- sort(unique(modes))
  modes <- modes[modes != "none"]
  if ("both" %in% modes) {
    modes <- union(c("jackknife", "conformal"), setdiff(modes, c("jackknife", "conformal", "both")))
  }
  valid <- c("jackknife", "conformal")
  if (length(modes)) {
    invalid <- setdiff(modes, valid)
    if (length(invalid)) {
      stop(
        sprintf(
          "Invalid --inference value(s): %s (expected comma-separated values from: %s)",
          paste(invalid, collapse = ", "),
          paste(valid, collapse = ", ")
        ),
        call. = FALSE
      )
    }
  }
  modes
}

sweep_labels <- c(
  donors = "donors",
  pre = "pre_periods",
  post = "post_periods"
)

parse_cli_args <- function(args) {
  parsed <- list(
    output_dir = script_dir(),
    figures_dir = NULL,
    backend_lib = backend_env_var(),
    reps = 20L,
    with_gsynth = TRUE,
    seed = 20260409L,
    sweep = "donors",
    sweep_values = NULL,
    donors = 30L,
    pre_periods = 20L,
    post_periods = 10L,
    inference = c("jackknife", "conformal")
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
      parsed$seed <- as.integer(args[[idx]])
    } else if (arg == "--sweep") {
      idx <- idx + 1L
      parsed$sweep <- args[[idx]]
    } else if (arg == "--sweep-values") {
      idx <- idx + 1L
      parsed$sweep_values <- parse_int_vector(args[[idx]], "sweep-values")
    } else if (arg == "--donors") {
      idx <- idx + 1L
      parsed$donors <- parse_int_scalar(args[[idx]], "donors")
    } else if (arg == "--pre-periods") {
      idx <- idx + 1L
      parsed$pre_periods <- parse_int_scalar(args[[idx]], "pre-periods")
    } else if (arg == "--post-periods") {
      idx <- idx + 1L
      parsed$post_periods <- parse_int_scalar(args[[idx]], "post-periods")
    } else if (arg == "--skip-gsynth") {
      parsed$with_gsynth <- FALSE
    } else if (arg == "--with-gsynth") {
      parsed$with_gsynth <- TRUE
    } else if (arg == "--inference") {
      idx <- idx + 1L
      parsed$inference <- parse_inference_modes(args[[idx]])
    } else if (arg == "--skip-inference") {
      parsed$inference <- character(0)
    } else {
      stop(sprintf("Unknown argument: %s", arg), call. = FALSE)
    }

    idx <- idx + 1L
  }

  if (!nzchar(parsed$sweep) || is.na(match(parsed$sweep, names(sweep_labels)))) {
    stop(
      sprintf(
        "Invalid --sweep value: %s (expected one of %s)",
        parsed$sweep,
        paste(names(sweep_labels), collapse = ", ")
      ),
      call. = FALSE
    )
  }
  parsed$reps <- max(1L, parsed$reps)
  if (!length(parsed$inference)) {
    parsed$inference <- character(0)
  }
  parsed
}

build_sweep_grid <- function(sweep, sweep_values, donors, pre_periods, post_periods) {
  if (length(sweep_values) == 0L) {
    sweep_values <- switch(
      sweep,
      donors = donors,
      pre = pre_periods,
      post = post_periods,
      donors
    )
  } else {
    if (sweep == "donors") {
      donors <- sweep_values
    } else if (sweep == "pre") {
      pre_periods <- sweep_values
    } else {
      post_periods <- sweep_values
    }
  }

  expand.grid(
    donors = as.integer(donors),
    pre_periods = as.integer(pre_periods),
    post_periods = as.integer(post_periods),
    stringsAsFactors = FALSE
  )
}

build_scenario_label <- function(sweep, row) {
  if (sweep == "donors") {
    sprintf("donors=%d", row[["donors"]])
  } else if (sweep == "pre") {
    sprintf("pre=%d", row[["pre_periods"]])
  } else {
    sprintf("post=%d", row[["post_periods"]])
  }
}

gsynth_supported <- function(fun, candidate) {
  formals <- names(formals(fun))
  candidate[names(candidate) %in% formals]
}

resolve_gsynth_fit <- function(formula, data, unit_name, time_name) {
  if (!requireNamespace("gsynth", quietly = TRUE)) {
    stop("Package 'gsynth' is not installed.", call. = FALSE)
  }

  gs <- get("gsynth", envir = asNamespace("gsynth"))
  candidate_args <- list(
    list(
      estimator = "mc",
      force = "two-way",
      scm = TRUE,
      CV = FALSE,
      nboots = 0L,
      seed = 1L
    ),
    list(
      estimator = "mc",
      force = "two-way",
      scm = TRUE,
      CV = TRUE,
      nboots = 0L,
      seed = 1L
    ),
    list(
      estimator = "mc",
      force = "two-way",
      CV = FALSE
    ),
    list(
      force = "two-way",
      scm = TRUE,
      CV = FALSE
    ),
    list(
      force = "two-way",
      scm = TRUE
    ),
    list(
      force = "two-way"
    )
  )

  for (candidate in candidate_args) {
    args <- c(
      list(formula = formula, data = data, index = c(unit_name, time_name)),
      gsynth_supported(gs, candidate)
    )
    fit <- tryCatch(
      do.call(gs, args),
      error = function(e) e
    )
    if (!inherits(fit, "error")) {
      return(fit)
    }
  }

  stop("Unable to call gsynth with any supported candidate signature.", call. = FALSE)
}

ns_fun <- function(pkg, name) {
  get(name, envir = asNamespace(pkg))
}

summary_with_pkg <- function(pkg, fit, inf_type) {
  summary_fun <- ns_fun(pkg, "summary.augsynth")
  summary_fun(fit, inf = TRUE, inf_type = inf_type)
}

simulate_augsynth_dataset <- function(n_donors, pre_periods, post_periods, seed,
                                     outcome_name = "gdpcap",
                                     unit_name = "regionno",
                                     time_name = "year") {
  if (n_donors < 1L) {
    stop("n_donors must be >= 1", call. = FALSE)
  }
  if (pre_periods < 1L || post_periods < 1L) {
    stop("pre_periods and post_periods must be >= 1", call. = FALSE)
  }

  n_units <- n_donors + 1L
  unit_values <- paste0("u", seq_len(n_units))
  time_values <- seq_len(pre_periods + post_periods)
  treated_unit <- unit_values[[n_units]]
  t_int <- pre_periods + 1L

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
  time_wave <- sin(2 * pi * panel[[time_name]] / max(1L, max(time_values))) + 0.03 * panel[[time_name]]
  unit_idx <- match(panel[[unit_name]], unit_values)
  trt <- as.integer(panel[[unit_name]] == treated_unit & panel[[time_name]] >= t_int)

  baseline <- 4.0 + 2.0 * unit_effect[unit_idx] + 0.15 * panel[[time_name]] + 0.9 * time_wave
  treatment <- as.numeric(trt) * (5.0 + 0.1 * (panel[[time_name]] - pre_periods))
  panel[[outcome_name]] <- baseline + treatment + rnorm(nrow(panel), sd = 0.75)
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

run_scenario_benchmark <- function(scenario, scenario_id, seed, with_gsynth, reps, inference_modes) {
  dataset <- simulate_augsynth_dataset(
    n_donors = scenario$donors,
    pre_periods = scenario$pre_periods,
    post_periods = scenario$post_periods,
    seed = seed + scenario_id,
    outcome_name = "gdpcap",
    unit_name = "regionno",
    time_name = "year"
  )

  formula <- as.formula(sprintf("%s ~ trt", dataset$outcome))
  unit_name <- as.character(dataset$unit)
  time_name <- as.character(dataset$time)
  t_int <- dataset$t_int
  panel <- dataset$data

  unit_symbol <- as.name(unit_name)
  time_symbol <- as.name(time_name)

  fastaugsynth_fit_expr <- as.call(list(
    quote(fastaugsynth::augsynth),
    as.name("formula"),
    unit_symbol,
    time_symbol,
    as.name("panel"),
    progfunc = "None",
    scm = TRUE,
    t_int = as.name("t_int")
  ))
  augsynth_fit_expr <- as.call(list(
    quote(augsynth::augsynth),
    as.name("formula"),
    unit_symbol,
    time_symbol,
    as.name("panel"),
    progfunc = "None",
    scm = TRUE,
    t_int = as.name("t_int")
  ))
  gsynth_fit_expr <- as.call(list(
    as.name("resolve_gsynth_fit"),
    as.name("formula"),
    as.name("panel"),
    as.name(unit_name),
    as.name(time_name)
  ))

  method_exprs <- list(
    fastaugsynth = fastaugsynth_fit_expr,
    augsynth = augsynth_fit_expr
  )
  if (with_gsynth) {
    method_exprs$gsynth <- gsynth_fit_expr
  }

  if ("jackknife" %in% inference_modes) {
    fastaugsynth_inference_fit <- eval(fastaugsynth_fit_expr)
    augsynth_inference_fit <- eval(augsynth_fit_expr)
    method_exprs$fastaugsynth_jackknife <- quote(summary_with_pkg("fastaugsynth", fastaugsynth_inference_fit, "jackknife"))
    method_exprs$augsynth_jackknife <- quote(summary_with_pkg("augsynth", augsynth_inference_fit, "jackknife"))
  }
  if ("conformal" %in% inference_modes) {
    if (!exists("fastaugsynth_inference_fit", inherits = FALSE)) {
      fastaugsynth_inference_fit <- eval(fastaugsynth_fit_expr)
      augsynth_inference_fit <- eval(augsynth_fit_expr)
    }
    method_exprs$fastaugsynth_conformal <- quote(summary_with_pkg("fastaugsynth", fastaugsynth_inference_fit, "conformal"))
    method_exprs$augsynth_conformal <- quote(summary_with_pkg("augsynth", augsynth_inference_fit, "conformal"))
  }

  mark <- do.call(
    bench::mark,
    c(
      method_exprs,
      list(
        iterations = reps,
        check = FALSE,
        min_time = 0
      )
    )
  )

  method_to_phase <- function(method) {
    if (method == "fastaugsynth" || method == "augsynth" || method == "gsynth") {
      return("estimate")
    }
    if (grepl("_jackknife$", method)) {
      return("jackknife")
    }
    if (grepl("_conformal$", method)) {
      return("conformal")
    }
    "other"
  }

  timings <- as.data.frame(mark)
  timings$method <- names(method_exprs)
  timings <- timings[, c("method", "min", "median", "itr/sec", "mem_alloc")]
  timings$phase <- vapply(timings$method, method_to_phase, character(1))
  timings$min_ms <- as.numeric(timings$min) * 1000
  timings$median_ms <- as.numeric(timings$median) * 1000
  timings$itr_per_sec <- as.numeric(timings$`itr/sec`)
  timings$mem_alloc_mb <- as.numeric(timings$mem_alloc) / 1024^2
  timings <- timings[, c(
    "method",
    "phase",
    "min_ms",
    "median_ms",
    "itr_per_sec",
    "mem_alloc_mb"
  )]
  timings$donors <- scenario$donors
  timings$pre_periods <- scenario$pre_periods
  timings$post_periods <- scenario$post_periods
  timings$scenario <- scenario_id
  timings$sweep_value <- scenario$scenario_value

  warm_summary <- aggregate(
    cbind(min_ms, median_ms, itr_per_sec, mem_alloc_mb) ~
      method + phase + scenario + sweep_value + donors + pre_periods + post_periods,
    data = timings,
    FUN = mean
  )
  warm_summary <- warm_summary[order(warm_summary$scenario, warm_summary$phase, warm_summary$median_ms), ]
  warm_summary$median_ms_over_fastaugsynth <- NA_real_
  warm_summary$itr_per_sec_vs_fastaugsynth <- NA_real_
  phase_base <- list(
    estimate = "fastaugsynth",
    jackknife = "fastaugsynth_jackknife",
    conformal = "fastaugsynth_conformal"
  )

  for (scenario_idx in unique(warm_summary$scenario)) {
    for (phase_idx in unique(warm_summary$phase)) {
      if (is.na(phase_idx)) {
        next
      }
      base_method <- phase_base[[phase_idx]]
      if (is.null(base_method)) {
        next
      }
      base_rows <- warm_summary$scenario == scenario_idx & warm_summary$phase == phase_idx & warm_summary$method == base_method
      if (!any(base_rows)) {
        next
      }
      base <- warm_summary$median_ms[base_rows][[1]]
      base_itr <- warm_summary$itr_per_sec[base_rows][[1]]
      row_idx <- warm_summary$scenario == scenario_idx & warm_summary$phase == phase_idx
      warm_summary$median_ms_over_fastaugsynth[row_idx] <- warm_summary$median_ms[row_idx] / base
      warm_summary$itr_per_sec_vs_fastaugsynth[row_idx] <- warm_summary$itr_per_sec[row_idx] / base_itr
    }
  }

  list(
    timings = timings,
    summary = warm_summary,
    methods = names(method_exprs),
    with_gsynth = with_gsynth
  )
}

run_benchmark <- function(sweep = "donors", sweep_values = NULL, reps = 20L, seed = 20260409L,
                          with_gsynth = TRUE, inference = c("jackknife", "conformal"),
                          donors = 30L, pre_periods = 20L, post_periods = 10L) {
  if (!requireNamespace("bench", quietly = TRUE)) {
    stop("Package 'bench' is required to run benchmark.", call. = FALSE)
  }
  if (!requireNamespace("fastaugsynth", quietly = TRUE)) {
    stop("Package 'fastaugsynth' must be installed before running this benchmark.", call. = FALSE)
  }
  if (!requireNamespace("augsynth", quietly = TRUE)) {
    stop("Package 'augsynth' must be installed before running this benchmark.", call. = FALSE)
  }

  with_gsynth <- isTRUE(with_gsynth) && requireNamespace("gsynth", quietly = TRUE)
  if (!with_gsynth) {
    message("Skipping gsynth because package 'gsynth' is not installed.")
  }

  scenario_grid <- build_sweep_grid(
    sweep = sweep,
    sweep_values = sweep_values,
    donors = donors,
    pre_periods = pre_periods,
    post_periods = post_periods
  )
  scenario_grid$scenario <- vapply(
    seq_len(nrow(scenario_grid)),
    function(idx) build_scenario_label(sweep, scenario_grid[idx, ]),
    character(1)
  )
  scenario_grid$scenario_value <- scenario_grid[[sweep_labels[[sweep]]]]

  outputs <- lapply(
    seq_len(nrow(scenario_grid)),
    function(idx) {
      run_scenario_benchmark(
        scenario = scenario_grid[idx, ],
        scenario_id = idx,
        seed = seed + idx,
        with_gsynth = with_gsynth,
        reps = reps,
        inference_modes = inference
      )
    }
  )

  timings <- do.call(rbind, lapply(outputs, function(x) x$timings))
  summary <- do.call(rbind, lapply(outputs, function(x) x$summary))
  methods <- outputs[[1]]$methods

  list(
    timings = timings,
    summary = summary,
    methods = methods,
    inference = inference,
    scenarios = scenario_grid,
    with_gsynth = with_gsynth
  )
}

plot_bars_by_sweep <- function(summary, methods, sweep, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required to draw benchmark figures.", call. = FALSE)
  }
  package_colors <- c(
    fastaugsynth = "#1b9e77",
    augsynth = "#d95f02",
    gsynth = "#7570b3"
  )
  phase_labels <- c(
    estimate = "Fit",
    jackknife = "Jackknife",
    conformal = "Conformal"
  )
  package_labels <- c(
    fastaugsynth = "fastaugsynth",
    augsynth = "augsynth",
    gsynth = "gsynth"
  )
  phase_order <- intersect(c("estimate", "jackknife", "conformal"), unique(as.character(summary$phase)))
  package_order <- c("fastaugsynth", "augsynth", "gsynth")

  phase_of_method <- function(method) {
    if (method %in% c("fastaugsynth", "augsynth", "gsynth")) {
      return("estimate")
    }
    if (grepl("_jackknife$", method)) {
      return("jackknife")
    }
    if (grepl("_conformal$", method)) {
      return("conformal")
    }
    "other"
  }

  package_of_method <- function(method) {
    if (grepl("^fastaugsynth", method)) {
      return("fastaugsynth")
    }
    if (grepl("^augsynth", method)) {
      return("augsynth")
    }
    if (grepl("^gsynth", method)) {
      return("gsynth")
    }
    "other"
  }

  method_info <- data.frame(
    method = methods,
    package = vapply(methods, package_of_method, character(1)),
    stringsAsFactors = FALSE
  )
  method_info <- method_info[method_info$package %in% package_order, , drop = FALSE]
  method_info <- method_info[order(match(method_info$package, package_order)), , drop = FALSE]
  warm <- summary
  label <- if (sweep == "donors") "donors" else if (sweep == "pre") "pre periods" else "post periods"
  plot_df <- merge(warm, method_info, by = "method", all.x = FALSE, all.y = FALSE)
  plot_df$phase <- as.character(plot_df$phase)
  plot_df$package <- as.character(plot_df$package)
  plot_df$phase_label <- factor(
    unname(phase_labels[plot_df$phase]),
    levels = unname(phase_labels[phase_order])
  )
  plot_df$package_label <- factor(
    unname(package_labels[plot_df$package]),
    levels = unname(package_labels[intersect(package_order, unique(plot_df$package))])
  )
  plot_df$sweep_label <- factor(as.character(plot_df$sweep_value), levels = as.character(sort(unique(plot_df$sweep_value))))

  bar_path <- file.path(output_dir, sprintf("augsynth_vs_gsynth_%s_bar.png", sweep))
  plot_obj <- ggplot2::ggplot(
    plot_df,
    ggplot2::aes(x = sweep_label, y = median_ms, fill = package_label)
  ) +
    ggplot2::geom_col(
      position = ggplot2::position_dodge(width = 0.72),
      width = 0.62
    ) +
    ggplot2::facet_wrap(~phase_label, nrow = 1, scales = "free_y") +
    ggplot2::scale_fill_manual(values = package_colors[intersect(package_order, unique(plot_df$package))], name = "Package") +
    ggplot2::labs(
      title = sprintf("Augsynth timing sweep: %s", label),
      x = label,
      y = "Median elapsed per call (ms)"
    ) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      legend.position = "right",
      legend.box = "vertical",
      panel.grid.major.x = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(face = "bold"),
      strip.background = ggplot2::element_rect(fill = "#eef2f7", colour = NA),
      plot.title = ggplot2::element_text(face = "bold"),
      axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5, hjust = 1)
    )
  ggplot2::ggsave(bar_path, plot_obj, width = 11, height = 6.5, dpi = 180)

  c(bar = bar_path)
}

write_metadata <- function(output_dir, backend_lib, reps, seed, sweep, sweep_values, inference, with_gsynth, scenarios) {
  meta_path <- file.path(output_dir, "results", "benchmark_metadata.txt")
  dir.create(dirname(meta_path), recursive = TRUE, showWarnings = FALSE)

  lines <- c(
    sprintf("generated_at: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    sprintf("backend_lib: %s", backend_lib),
    sprintf("reps: %d", reps),
    sprintf("seed: %d", seed),
    sprintf("sweep: %s", sweep),
    sprintf("sweep_values: %s", paste(as.character(sweep_values), collapse = ",")),
    sprintf("inference: %s", if (length(inference)) paste(inference, collapse = ",") else "none"),
    sprintf("with_gsynth: %s", as.character(with_gsynth)),
    sprintf("R.version: %s", R.version.string),
    sprintf("platform: %s", R.version$platform),
    sprintf("fastaugsynth.version: %s", as.character(utils::packageVersion("fastaugsynth"))),
    sprintf("augsynth.version: %s", as.character(utils::packageVersion("augsynth"))),
    if (isTRUE(with_gsynth)) sprintf("gsynth.version: %s", as.character(utils::packageVersion("gsynth"))) else "gsynth.version: not run"
  )
  scenario_lines <- vapply(
    seq_len(nrow(scenarios)),
    function(idx) {
      row <- scenarios[idx, ]
      sprintf(
        "scenario[%d]: donors=%d, pre_periods=%d, post_periods=%d",
        idx,
        as.integer(row$donors),
        as.integer(row$pre_periods),
        as.integer(row$post_periods)
      )
    },
    character(1)
  )
  lines <- c(lines, scenario_lines)
  writeLines(lines, meta_path)
  meta_path
}

run_augsynth_vs_gsynth <- function(output_dir = script_dir(),
                                   figures_dir = NULL,
                                   backend_lib = backend_env_var(),
                                   reps = 20L,
                                   with_gsynth = TRUE,
                                   inference = c("jackknife", "conformal"),
                                   seed = 20260409L,
                                   sweep = "donors",
                                   sweep_values = NULL,
                                   donors = 30L,
                                   pre_periods = 20L,
                                   post_periods = 10L) {
  if (!nzchar(backend_lib)) {
    stop(
      "Set FASTAUGSYNTH_BACKEND_LIB or pass --backend-lib so augsynth() can find the compiled backend library.",
      call = FALSE
    )
  }
  if (!requireNamespace("stats", quietly = TRUE)) {
    stop("Package 'stats' must be available before running the benchmark.", call. = FALSE)
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

  outputs <- run_benchmark(
    sweep = sweep,
    sweep_values = sweep_values,
    reps = reps,
    seed = seed,
    with_gsynth = with_gsynth,
    inference = inference,
    donors = donors,
    pre_periods = pre_periods,
    post_periods = post_periods
  )
  timings <- outputs$timings
  summary <- outputs$summary

  write.csv(timings, file.path(results_dir, "augsynth_vs_gsynth_timings.csv"), row.names = FALSE)
  write.csv(summary, file.path(results_dir, "augsynth_vs_gsynth_summary.csv"), row.names = FALSE)
  write.csv(
    summary[, c(
      "scenario",
      "sweep_value",
      "phase",
      "method",
      "median_ms",
      "median_ms_over_fastaugsynth",
      "itr_per_sec_vs_fastaugsynth",
      "mem_alloc_mb"
    )],
    file.path(results_dir, "augsynth_vs_gsynth_speedup.csv"),
    row.names = FALSE
  )
  meta_path <- write_metadata(
    output_dir = output_dir,
    backend_lib = backend_lib,
    reps = reps,
    seed = seed,
    sweep = sweep,
    sweep_values = sweep_values %||% outputs$scenarios[[sweep_labels[[sweep]]]],
    inference = inference,
    with_gsynth = outputs$with_gsynth,
    scenarios = outputs$scenarios
  )

  plot_paths <- plot_bars_by_sweep(
    summary = summary,
    methods = outputs$methods,
    sweep = sweep,
    output_dir = figures_dir
  )

  list(
    timings = timings,
    summary = summary,
    figures = plot_paths,
    metadata = meta_path,
    output_dir = output_dir,
    figures_dir = figures_dir,
    scenarios = outputs$scenarios
  )
}

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs)) rhs else lhs
}

main <- function() {
  cli <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  outputs <- run_augsynth_vs_gsynth(
    output_dir = cli$output_dir,
    figures_dir = cli$figures_dir,
    backend_lib = cli$backend_lib,
    reps = cli$reps,
    with_gsynth = cli$with_gsynth,
    seed = cli$seed,
    sweep = cli$sweep,
    sweep_values = cli$sweep_values,
    donors = cli$donors,
    pre_periods = cli$pre_periods,
    post_periods = cli$post_periods
    ,
    inference = cli$inference
  )

  cat("Augsynth vs gsynth summary:\n")
  print(outputs$summary)
  cat("\nWrote benchmark outputs to:\n")
  cat("  ", outputs$output_dir, "\n", sep = "")
  cat("  ", outputs$figures_dir, "\n", sep = "")
}

if (sys.nframe() == 0L) {
  main()
}
