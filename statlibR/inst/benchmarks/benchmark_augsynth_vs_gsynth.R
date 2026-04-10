#!/usr/bin/env Rscript

backend_env_var <- function() {
  Sys.getenv("METRICSJL_BACKEND_LIB", Sys.getenv("STATLIB_BACKEND_LIB", ""))
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

load_dataset <- function() {
  if (!requireNamespace("Synth", quietly = TRUE)) {
    stop(
      "Package 'Synth' is required for the built-in basque benchmark dataset.",
      call. = FALSE
    )
  }
  library(Synth)
  data("basque", package = "Synth")
  basque <- transform(
    basque,
    trt = ifelse(year < 1975, 0, ifelse(regionno == 17, 1, 0))
  )
  subset(basque, regionno != 1)
}

parse_cli_args <- function(args) {
  parsed <- list(
    output_dir = script_dir(),
    backend_lib = backend_env_var(),
    reps = 20L,
    with_gsynth = TRUE,
    seed = 20260409L
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
    } else if (arg == "--seed") {
      idx <- idx + 1L
      parsed$seed <- as.integer(args[[idx]])
    } else if (arg == "--skip-gsynth") {
      parsed$with_gsynth <- FALSE
    } else if (arg == "--with-gsynth") {
      parsed$with_gsynth <- TRUE
    } else {
      stop(sprintf("Unknown argument: %s", arg), call. = FALSE)
    }

    idx <- idx + 1L
  }

  parsed$reps <- max(1L, parsed$reps)
  parsed
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

run_benchmark <- function(with_gsynth = TRUE, reps = 20L, seed = 20260409L) {
  if (!requireNamespace("bench", quietly = TRUE)) {
    stop("Package 'bench' is required to run benchmark.", call. = FALSE)
  }
  if (!requireNamespace("metricsjl", quietly = TRUE)) {
    stop("Package 'metricsjl' must be installed before running this benchmark.", call. = FALSE)
  }
  if (!requireNamespace("augsynth", quietly = TRUE)) {
    stop("Package 'augsynth' must be installed before running this benchmark.", call. = FALSE)
  }

  with_gsynth <- isTRUE(with_gsynth) && requireNamespace("gsynth", quietly = TRUE)
  if (!with_gsynth) {
    message("Skipping gsynth because package 'gsynth' is not installed.")
  }

  set.seed(seed)
  basque_panel <- load_dataset()
  unit_name <- "regionno"
  time_name <- "year"
  formula <- gdpcap ~ trt

  if (with_gsynth) {
    mark <- bench::mark(
      metricsjl = metricsjl::augsynth(
        formula,
        unit_name,
        time_name,
        basque_panel,
        progfunc = "None",
        scm = TRUE,
        t_int = 1975
      ),
      augsynth = augsynth::augsynth(
        formula,
        unit_name,
        time_name,
        basque_panel,
        progfunc = "None",
        scm = TRUE,
        t_int = 1975
      ),
      gsynth = resolve_gsynth_fit(formula, basque_panel, unit_name, time_name),
      iterations = reps,
      check = FALSE,
      min_time = 0
    )
  } else {
    mark <- bench::mark(
      metricsjl = metricsjl::augsynth(
        formula,
        unit_name,
        time_name,
        basque_panel,
        progfunc = "None",
        scm = TRUE,
        t_int = 1975
      ),
      augsynth = augsynth::augsynth(
        formula,
        unit_name,
        time_name,
        basque_panel,
        progfunc = "None",
        scm = TRUE,
        t_int = 1975
      ),
      iterations = reps,
      check = FALSE,
      min_time = 0
    )
  }

  timings <- as.data.frame(mark)
  timings$method <- vapply(timings$expression, as.character, character(1))
  timings <- timings[, c("method", "min", "median", "itr/sec", "mem_alloc")]
  timings$min_ms <- as.numeric(timings$min) * 1000
  timings$median_ms <- as.numeric(timings$median) * 1000
  timings$itr_per_sec <- as.numeric(timings$`itr/sec`)
  timings$mem_alloc_mb <- as.numeric(timings$mem_alloc) / 1024^2
  timings <- timings[, c(
    "method",
    "min_ms",
    "median_ms",
    "itr_per_sec",
    "mem_alloc_mb"
  )]

  warm_summary <- aggregate(
    cbind(min_ms, median_ms, itr_per_sec, mem_alloc_mb) ~ method,
    data = timings,
    FUN = mean
  )
  warm_summary <- warm_summary[order(warm_summary$median_ms), ]

  base_ms <- warm_summary$median_ms[warm_summary$method == "metricsjl"]
  if (length(base_ms) != 1L) {
    stop("Unable to locate metricsjl row for speedup calculation.", call. = FALSE)
  }
  warm_summary$median_ms_over_metricsjl <- warm_summary$median_ms / base_ms
  warm_summary$itr_per_sec_vs_metricsjl <- warm_summary$itr_per_sec / warm_summary$itr_per_sec[warm_summary$method == "metricsjl"]

  list(
    timings = timings,
    summary = warm_summary,
    methods = if (with_gsynth) c("metricsjl", "augsynth", "gsynth") else c("metricsjl", "augsynth"),
    with_gsynth = with_gsynth
  )
}

write_metadata <- function(output_dir, backend_lib, reps, seed, with_gsynth) {
  meta_path <- file.path(output_dir, "results", "benchmark_metadata.txt")
  dir.create(dirname(meta_path), recursive = TRUE, showWarnings = FALSE)

  lines <- c(
    sprintf("generated_at: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
    sprintf("backend_lib: %s", backend_lib),
    sprintf("reps: %d", reps),
    sprintf("seed: %d", seed),
    sprintf("with_gsynth: %s", as.character(with_gsynth)),
    sprintf("R.version: %s", R.version.string),
    sprintf("platform: %s", R.version$platform),
    sprintf("metricsjl.version: %s", as.character(utils::packageVersion("metricsjl"))),
    sprintf("augsynth.version: %s", as.character(utils::packageVersion("augsynth"))),
    if (isTRUE(with_gsynth)) sprintf("gsynth.version: %s", as.character(utils::packageVersion("gsynth"))) else "gsynth.version: not run"
  )
  writeLines(lines, meta_path)
  meta_path
}

run_augsynth_vs_gsynth <- function(output_dir = script_dir(),
                                   backend_lib = backend_env_var(),
                                   reps = 20L,
                                   with_gsynth = TRUE,
                                   seed = 20260409L) {
  if (!nzchar(backend_lib)) {
    stop(
      "Set METRICSJL_BACKEND_LIB or pass --backend-lib so augsynth() can find the compiled backend library.",
      call = FALSE
    )
  }
  if (!requireNamespace("stats", quietly = TRUE)) {
    stop("Package 'stats' must be available before running the benchmark.", call. = FALSE)
  }

  Sys.setenv(METRICSJL_BACKEND_LIB = backend_lib)

  output_dir <- normalizePath(output_dir, mustWork = FALSE)
  results_dir <- file.path(output_dir, "results")
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)

  outputs <- run_benchmark(with_gsynth = with_gsynth, reps = reps, seed = seed)
  timings <- outputs$timings
  summary <- outputs$summary

  write.csv(timings, file.path(results_dir, "augsynth_vs_gsynth_timings.csv"), row.names = FALSE)
  write.csv(summary, file.path(results_dir, "augsynth_vs_gsynth_summary.csv"), row.names = FALSE)
  meta_path <- write_metadata(output_dir, backend_lib, reps, seed, outputs$with_gsynth)

  list(
    timings = timings,
    summary = summary,
    metadata = meta_path,
    output_dir = output_dir
  )
}

main <- function() {
  cli <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  outputs <- run_augsynth_vs_gsynth(
    output_dir = cli$output_dir,
    backend_lib = cli$backend_lib,
    reps = cli$reps,
    with_gsynth = cli$with_gsynth,
    seed = cli$seed
  )

  cat("Augsynth vs gsynth summary:\n")
  print(outputs$summary)
  cat("\nWrote benchmark outputs to:\n")
  cat("  ", outputs$output_dir, "\n", sep = "")
}

if (sys.nframe() == 0L) {
  main()
}
