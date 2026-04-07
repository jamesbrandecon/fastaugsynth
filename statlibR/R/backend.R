backend_path <- function() {
  p <- Sys.getenv("STATLIB_BACKEND_LIB", "")
  if (nzchar(p)) return(normalizePath(p, mustWork = FALSE))

  ext <- if (.Platform$OS.type == "windows") "dll" else if (Sys.info()[["sysname"]] == "Darwin") "dylib" else "so"
  file.path(path.expand("~"), ".cache", "statlibR", "backend", paste0("libstatlibbackend.", ext))
}

backend_status <- function() {
  p <- backend_path()
  list(path = p, exists = file.exists(p))
}
