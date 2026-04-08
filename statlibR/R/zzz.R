.onLoad <- function(libname, pkgname) {
  auto_install <- tolower(Sys.getenv("METRICSJL_AUTO_INSTALL_BACKEND", "true"))
  if (!(auto_install %in% c("1", "true", "yes"))) {
    return(invisible(NULL))
  }

  try({
    if (!file.exists(backend_path())) {
      backend_install(quiet = TRUE)
    }
  }, silent = TRUE)

  invisible(NULL)
}
