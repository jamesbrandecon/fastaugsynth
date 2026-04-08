backend_env_var <- function() {
  Sys.getenv("METRICSJL_BACKEND_LIB", Sys.getenv("STATLIB_BACKEND_LIB", ""))
}

backend_desc <- function() {
  tryCatch(utils::packageDescription("metricsjl"), error = function(e) NULL)
}

backend_desc_field <- function(desc, field) {
  if (is.null(desc)) {
    return("")
  }

  value <- desc[[field]]
  if (is.null(value) || length(value) != 1 || is.na(value) || !nzchar(value)) {
    return("")
  }

  value
}

backend_cache_roots <- function() {
  c(
    file.path(path.expand("~"), ".cache", "metricsjl", "backend"),
    file.path(path.expand("~"), ".cache", "statlibR", "backend")
  )
}

backend_cache_root <- function() {
  backend_cache_roots()[[1]]
}

backend_library_ext <- function() {
  if (.Platform$OS.type == "windows") {
    "dll"
  } else if (identical(Sys.info()[["sysname"]], "Darwin")) {
    "dylib"
  } else {
    "so"
  }
}

backend_candidates <- function(cache_roots = backend_cache_roots()) {
  ext <- backend_library_ext()
  unlist(lapply(cache_roots, function(cache_root) {
    c(
      file.path(cache_root, "lib", paste0("libstatlibbackend.", ext)),
      file.path(cache_root, paste0("libstatlibbackend.", ext))
    )
  }), use.names = FALSE)
}

backend_path <- function() {
  p <- backend_env_var()
  if (nzchar(p)) return(normalizePath(p, mustWork = FALSE))

  candidates <- backend_candidates()
  hit <- candidates[file.exists(candidates)]
  normalizePath(if (length(hit)) hit[[1]] else candidates[[1]], mustWork = FALSE)
}

backend_status <- function() {
  p <- backend_path()
  list(
    path = p,
    exists = file.exists(p),
    repo = backend_repo(),
    ref = backend_ref(),
    sha = backend_sha()
  )
}

backend_repo <- function() {
  override <- Sys.getenv("METRICSJL_BACKEND_REPO", "")
  if (nzchar(override)) return(override)

  desc <- backend_desc()
  username <- backend_desc_field(desc, "RemoteUsername")
  repo <- backend_desc_field(desc, "RemoteRepo")
  if (nzchar(username) && nzchar(repo)) {
    return(paste(username, repo, sep = "/"))
  }

  "jamesbrandecon/jlrstats"
}

normalize_backend_ref <- function(ref) {
  ref <- sub("^refs/heads/", "", ref)
  sub("^origin/", "", ref)
}

is_commit_ref <- function(ref) {
  grepl("^[0-9a-f]{7,40}$", ref)
}

backend_ref <- function() {
  override <- Sys.getenv("METRICSJL_BACKEND_REF", "")
  if (nzchar(override)) return(normalize_backend_ref(override))

  desc <- backend_desc()
  ref <- backend_desc_field(desc, "RemoteRef")
  if (nzchar(ref)) {
    return(normalize_backend_ref(ref))
  }

  "main"
}

backend_sha <- function() {
  override <- Sys.getenv("METRICSJL_BACKEND_SHA", "")
  if (nzchar(override)) return(tolower(override))

  desc <- backend_desc()
  sha <- backend_desc_field(desc, "RemoteSha")
  if (nzchar(sha)) {
    return(tolower(sha))
  }

  ref <- backend_ref()
  if (is_commit_ref(ref)) {
    return(tolower(ref))
  }

  ""
}

backend_token_from_gh <- function() {
  gh <- Sys.which("gh")
  if (!nzchar(gh)) {
    return("")
  }

  out <- tryCatch(
    suppressWarnings(system2(gh, c("auth", "token"), stdout = TRUE, stderr = FALSE)),
    error = function(e) character()
  )
  token <- trimws(paste(out, collapse = "\n"))
  if (nzchar(token)) token else ""
}

backend_token <- function() {
  token <- Sys.getenv("METRICSJL_GITHUB_PAT", Sys.getenv("GITHUB_PAT", ""))
  if (nzchar(token)) {
    return(token)
  }

  token <- backend_token_from_gh()
  if (nzchar(token)) {
    return(token)
  }

  creds <- tryCatch(gitcreds::gitcreds_get(url = "https://github.com"), error = function(e) NULL)
  if (!is.null(creds) && nzchar(creds$password)) {
    return(creds$password)
  }

  ""
}

backend_platform <- function() {
  sysname <- tolower(Sys.info()[["sysname"]])
  machine <- tolower(Sys.info()[["machine"]])

  os <- switch(
    sysname,
    darwin = "darwin",
    linux = "linux",
    windows = "windows",
    stop(sprintf("Unsupported OS for backend artifact download: %s", sysname), call. = FALSE)
  )

  arch <- if (machine %in% c("x86_64", "amd64")) {
    "x86_64"
  } else if (machine %in% c("arm64", "aarch64")) {
    "arm64"
  } else {
    stop(sprintf("Unsupported CPU architecture for backend artifact download: %s", machine), call. = FALSE)
  }

  list(os = os, arch = arch)
}

backend_artifact_name <- function() {
  platform <- backend_platform()
  sprintf("statlibbackend-%s-%s", platform$os, platform$arch)
}

github_headers <- function(token = "", accept = "application/vnd.github+json") {
  headers <- c(
    Accept = accept,
    "X-GitHub-Api-Version" = "2022-11-28",
    "User-Agent" = "metricsjl/0.1.0"
  )
  if (nzchar(token)) {
    headers <- c(headers, Authorization = paste("Bearer", token))
  }
  headers
}

github_fetch_json <- function(url, token = "") {
  response <- curl::curl_fetch_memory(
    url,
    handle = curl::new_handle(httpheader = github_headers(token))
  )

  if (response$status_code >= 300L) {
    message <- trimws(rawToChar(response$content))
    if (!nzchar(message)) {
      message <- "empty response body"
    }
    if (response$status_code == 404L) {
      message <- paste(
        message,
        "If this repo is private, make sure GITHUB_PAT or METRICSJL_GITHUB_PAT has repo access.",
        sep = " "
      )
    }
    stop(
      sprintf("GitHub API request failed (%d): %s", response$status_code, message),
      call. = FALSE
    )
  }

  jsonlite::fromJSON(rawToChar(response$content), simplifyVector = TRUE)
}

github_download_file <- function(url, destfile, token = "", accept = "application/octet-stream") {
  curl::curl_download(
    url = url,
    destfile = destfile,
    handle = curl::new_handle(
      followlocation = TRUE,
      httpheader = github_headers(token, accept = accept)
    ),
    quiet = TRUE
  )
}

backend_install_message <- function() {
  paste(
    "Backend library is not installed.",
    "Run metricsjl::backend_install() explicitly or set METRICSJL_BACKEND_LIB.",
    "For this private repo, ensure GITHUB_PAT or METRICSJL_GITHUB_PAT has repo access,",
    "or log in with gh auth/login, or store a GitHub token with gitcreds::gitcreds_set().",
    sep = " "
  )
}

backend_find_library <- function(cache_dir) {
  pattern <- paste0("libstatlibbackend\\.", backend_library_ext(), "$")
  matches <- list.files(cache_dir, pattern = pattern, recursive = TRUE, full.names = TRUE)
  if (!length(matches)) {
    stop(
      sprintf("Downloaded backend artifact did not contain %s", pattern),
      call. = FALSE
    )
  }
  normalizePath(matches[[1]], mustWork = FALSE)
}

github_workflow_runs_url <- function(repo, workflow, ref = "", per_page = 50L) {
  url <- sprintf(
    "https://api.github.com/repos/%s/actions/workflows/%s/runs?status=success&per_page=%d",
    repo,
    workflow,
    as.integer(per_page)
  )
  if (nzchar(ref) && !is_commit_ref(ref)) {
    url <- paste0(url, "&branch=", utils::URLencode(ref, reserved = TRUE))
  }
  url
}

backend_install <- function(repo = backend_repo(),
                            ref = backend_ref(),
                            sha = backend_sha(),
                            workflow = "phase1-no-julia-runtime.yml",
                            token = backend_token(),
                            cache_dir = backend_cache_root(),
                            force = FALSE,
                            quiet = FALSE) {
  existing <- backend_path()
  if (!force && file.exists(existing)) {
    return(invisible(existing))
  }

  runs_url <- github_workflow_runs_url(repo, workflow, ref = ref)
  runs <- github_fetch_json(runs_url, token = token)
  run_rows <- runs$workflow_runs
  if (is.null(run_rows) || !NROW(run_rows)) {
    stop(
      sprintf("No successful workflow runs found for %s on ref '%s'.", repo, ref),
      call. = FALSE
    )
  }

  if (nzchar(sha) && "head_sha" %in% names(run_rows)) {
    sha_matches <- run_rows[tolower(run_rows$head_sha) == tolower(sha), , drop = FALSE]
    if (NROW(sha_matches)) {
      run_rows <- sha_matches
    }
  }

  run <- run_rows[1, , drop = FALSE]
  artifacts_url <- sprintf(
    "https://api.github.com/repos/%s/actions/runs/%s/artifacts?per_page=100",
    repo,
    run$id[[1]]
  )
  artifacts <- github_fetch_json(artifacts_url, token = token)
  wanted_name <- backend_artifact_name()
  artifact_rows <- artifacts$artifacts[
    artifacts$artifacts$name == wanted_name & !artifacts$artifacts$expired,
    ,
    drop = FALSE
  ]
  if (!NROW(artifact_rows)) {
    stop(
      sprintf(
        "No artifact named '%s' found in successful run %s for %s on ref '%s'.",
        wanted_name,
        run$id[[1]],
        repo,
        ref
      ),
      call. = FALSE
    )
  }

  artifact <- artifact_rows[1, , drop = FALSE]
  zipfile <- tempfile(fileext = ".zip")
  unpack_dir <- tempfile("metricsjl-artifact-")
  on.exit(unlink(c(zipfile, unpack_dir), recursive = TRUE, force = TRUE), add = TRUE)
  dir.create(unpack_dir, recursive = TRUE, showWarnings = FALSE)

  github_download_file(artifact$archive_download_url[[1]], zipfile, token = token)
  extracted <- utils::unzip(zipfile, exdir = unpack_dir)
  tarball <- extracted[grepl("\\.tar\\.gz$", extracted)]
  if (!length(tarball)) {
    stop("Downloaded artifact did not contain a .tar.gz payload.", call. = FALSE)
  }

  unlink(cache_dir, recursive = TRUE, force = TRUE)
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
  utils::untar(tarball[[1]], exdir = cache_dir)
  installed_lib <- backend_find_library(cache_dir)

  if (!quiet) {
    message(
      sprintf(
        "Installed backend artifact '%s' from %s on ref '%s' into %s",
        wanted_name,
        repo,
        if (nzchar(sha)) sha else ref,
        cache_dir
      )
    )
  }

  invisible(installed_lib)
}

ensure_backend_available <- function() {
  p <- backend_path()
  if (file.exists(p)) {
    return(p)
  }

  auto_install <- tolower(Sys.getenv("METRICSJL_AUTO_INSTALL_BACKEND", "true"))
  if (auto_install %in% c("1", "true", "yes")) {
    try(backend_install(quiet = TRUE), silent = TRUE)
    p <- backend_path()
    if (file.exists(p)) {
      return(p)
    }
  }

  stop(backend_install_message(), call. = FALSE)
}
