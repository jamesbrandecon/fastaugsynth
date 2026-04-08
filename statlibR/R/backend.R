backend_path <- function() {
  p <- Sys.getenv("METRICSJL_BACKEND_LIB", Sys.getenv("STATLIB_BACKEND_LIB", ""))
  if (nzchar(p)) return(normalizePath(p, mustWork = FALSE))

  ext <- if (.Platform$OS.type == "windows") "dll" else if (Sys.info()[["sysname"]] == "Darwin") "dylib" else "so"
  cache_roots <- c(
    file.path(path.expand("~"), ".cache", "metricsjl", "backend"),
    file.path(path.expand("~"), ".cache", "statlibR", "backend")
  )
  candidates <- unlist(lapply(cache_roots, function(cache_root) {
    c(
      file.path(cache_root, "lib", paste0("libstatlibbackend.", ext)),
      file.path(cache_root, paste0("libstatlibbackend.", ext))
    )
  }), use.names = FALSE)

  hit <- candidates[file.exists(candidates)]
  normalizePath(if (length(hit)) hit[[1]] else candidates[[1]], mustWork = FALSE)
}

backend_status <- function() {
  p <- backend_path()
  list(path = p, exists = file.exists(p))
}
