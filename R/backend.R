backend_env_var <- function() {
  Sys.getenv("FASTAUGSYNTH_BACKEND_LIB", "")
}

backend_desc <- function() {
  tryCatch(utils::packageDescription("fastaugsynth"), error = function(e) NULL)
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
  file.path(path.expand("~"), ".cache", "fastaugsynth", "backend")
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
  override <- Sys.getenv("FASTAUGSYNTH_BACKEND_REPO", "")
  if (nzchar(override)) return(override)

  desc <- backend_desc()
  username <- backend_desc_field(desc, "RemoteUsername")
  repo <- backend_desc_field(desc, "RemoteRepo")
  if (nzchar(username) && nzchar(repo)) {
    return(paste(username, repo, sep = "/"))
  }

  "jamesbrandecon/fastaugsynth"
}

normalize_backend_ref <- function(ref) {
  ref <- sub("^refs/heads/", "", ref)
  sub("^origin/", "", ref)
}

is_commit_ref <- function(ref) {
  grepl("^[0-9a-f]{7,40}$", ref)
}

backend_ref <- function() {
  override <- Sys.getenv("FASTAUGSYNTH_BACKEND_REF", "")
  if (nzchar(override)) return(normalize_backend_ref(override))

  desc <- backend_desc()
  ref <- backend_desc_field(desc, "RemoteRef")
  if (nzchar(ref)) {
    return(normalize_backend_ref(ref))
  }

  "main"
}

backend_sha <- function() {
  override <- Sys.getenv("FASTAUGSYNTH_BACKEND_SHA", "")
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

backend_gh <- function() {
  Sys.which("gh")
}

github_api_endpoint <- function(url) {
  sub("^https://api\\.github\\.com/", "", url)
}

gh_api_request <- function(url,
                           accept = "application/vnd.github+json",
                           output = NULL) {
  gh <- backend_gh()
  if (!nzchar(gh)) {
    stop("gh CLI is not available", call. = FALSE)
  }

  endpoint <- github_api_endpoint(url)
  args <- c("api", "-X", "GET")
  if (!is.null(accept) && nzchar(accept)) {
    args <- c(args, "-H", paste0("Accept:", accept))
  }
  args <- c(args, "-H", "X-GitHub-Api-Version:2022-11-28", endpoint)

  stderr_file <- tempfile("fastaugsynth-gh-stderr-")
  on.exit(unlink(stderr_file, force = TRUE), add = TRUE)

  status <- suppressWarnings(system2(
    gh,
    args,
    stdout = if (is.null(output)) TRUE else output,
    stderr = stderr_file
  ))
  stderr_text <- paste(readLines(stderr_file, warn = FALSE), collapse = "\n")

  if (!is.null(output)) {
    if (!identical(status, 0L)) {
      stop(
        sprintf("gh api request failed (%s): %s", status, stderr_text),
        call. = FALSE
      )
    }
    return(invisible(output))
  }

  stdout_text <- paste(status, collapse = "\n")
  exit_status <- attr(status, "status")
  if (!is.null(exit_status) && exit_status != 0L) {
    stop(
      sprintf("gh api request failed (%s): %s", exit_status, stdout_text),
      call. = FALSE
    )
  }

  stdout_text
}

backend_token <- function() {
  token <- Sys.getenv("FASTAUGSYNTH_GITHUB_PAT", Sys.getenv("GITHUB_PAT", ""))
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
  machine <- gsub("-", "_", machine, fixed = TRUE)

  os <- switch(
    sysname,
    darwin = "darwin",
    linux = "linux",
    windows = "windows",
    stop(sprintf("Unsupported OS for backend artifact download: %s", sysname), call. = FALSE)
  )

  arch <- if (machine %in% c("x86_64", "amd64", "x64")) {
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
    "User-Agent" = "fastaugsynth/0.1.0"
  )
  if (nzchar(token)) {
    headers <- c(headers, Authorization = paste("Bearer", token))
  }
  headers
}

windows_powershell <- function() {
  if (.Platform$OS.type != "windows") {
    return("")
  }

  ps <- Sys.which("powershell.exe")
  if (nzchar(ps)) {
    return(ps)
  }

  Sys.which("powershell")
}

windows_ps_quote <- function(x) {
  sprintf("'%s'", gsub("'", "''", x, fixed = TRUE))
}

windows_github_fetch_json <- function(url,
                                      token = "",
                                      accept = "application/vnd.github+json") {
  ps <- windows_powershell()
  if (!nzchar(ps)) {
    stop("PowerShell is not available", call. = FALSE)
  }

  script <- tempfile("fastaugsynth-gh-", fileext = ".ps1")
  json_file <- tempfile("fastaugsynth-gh-json-", fileext = ".json")
  stderr_file <- tempfile("fastaugsynth-gh-stderr-")
  on.exit(unlink(c(script, json_file, stderr_file), force = TRUE), add = TRUE)

  lines <- c(
    "$ErrorActionPreference = 'Stop'",
    "$headers = @{}",
    sprintf("$headers['Accept'] = %s", windows_ps_quote(accept)),
    "$headers['X-GitHub-Api-Version'] = '2022-11-28'"
  )
  if (nzchar(token)) {
    lines <- c(lines, sprintf("$headers['Authorization'] = %s", windows_ps_quote(paste("Bearer", token))))
  }
  lines <- c(
    lines,
    sprintf("$resp = Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri %s", windows_ps_quote(url)),
    sprintf(
      "[System.IO.File]::WriteAllText(%s, $resp.Content, [System.Text.UTF8Encoding]::new($false))",
      windows_ps_quote(normalizePath(json_file, winslash = "\\", mustWork = FALSE))
    )
  )
  writeLines(lines, script, useBytes = TRUE)

  output <- tryCatch(
    suppressWarnings(system2(
      ps,
      c("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script),
      stdout = TRUE,
      stderr = stderr_file
    )),
    error = function(e) e
  )
  stderr_text <- paste(readLines(stderr_file, warn = FALSE), collapse = "\n")
  if (inherits(output, "error")) {
    stop(conditionMessage(output), call. = FALSE)
  }

  exit_status <- attr(output, "status")
  if (!is.null(exit_status) && exit_status != 0L) {
    message <- trimws(stderr_text)
    if (!nzchar(message)) {
      message <- "empty response body"
    }
    stop(
      sprintf("PowerShell GitHub request failed (%s): %s", exit_status, message),
      call. = FALSE
    )
  }

  jsonlite::fromJSON(paste(readLines(json_file, warn = FALSE, encoding = "UTF-8"), collapse = "\n"), simplifyVector = TRUE)
}

windows_github_download_file <- function(url,
                                         destfile,
                                         token = "",
                                         accept = "application/octet-stream") {
  ps <- windows_powershell()
  if (!nzchar(ps)) {
    stop("PowerShell is not available", call. = FALSE)
  }

  script <- tempfile("fastaugsynth-gh-download-", fileext = ".ps1")
  stderr_file <- tempfile("fastaugsynth-gh-download-stderr-")
  on.exit(unlink(c(script, stderr_file), force = TRUE), add = TRUE)

  lines <- c(
    "$ErrorActionPreference = 'Stop'",
    "$headers = @{}",
    sprintf("$headers['Accept'] = %s", windows_ps_quote(accept)),
    "$headers['X-GitHub-Api-Version'] = '2022-11-28'"
  )
  if (nzchar(token)) {
    lines <- c(lines, sprintf("$headers['Authorization'] = %s", windows_ps_quote(paste("Bearer", token))))
  }
  lines <- c(
    lines,
    sprintf("Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri %s -OutFile %s",
            windows_ps_quote(url),
            windows_ps_quote(normalizePath(destfile, winslash = "\\", mustWork = FALSE)))
  )
  writeLines(lines, script, useBytes = TRUE)

  output <- tryCatch(
    suppressWarnings(system2(
      ps,
      c("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script),
      stdout = TRUE,
      stderr = stderr_file
    )),
    error = function(e) e
  )
  stderr_text <- paste(readLines(stderr_file, warn = FALSE), collapse = "\n")
  if (inherits(output, "error")) {
    stop(conditionMessage(output), call. = FALSE)
  }

  exit_status <- attr(output, "status")
  if (!is.null(exit_status) && exit_status != 0L) {
    message <- trimws(stderr_text)
    if (!nzchar(message)) {
      message <- "empty response body"
    }
    stop(
      sprintf("PowerShell GitHub download failed (%s): %s", exit_status, message),
      call. = FALSE
    )
  }

  invisible(destfile)
}

github_fetch_json <- function(url, token = "") {
  if (.Platform$OS.type == "windows" && nzchar(windows_powershell()) && nzchar(token)) {
    return(windows_github_fetch_json(url, token = token))
  }

  response <- tryCatch(
    curl::curl_fetch_memory(
      url,
      handle = curl::new_handle(httpheader = github_headers(token))
    ),
    error = function(e) e
  )

  if (inherits(response, "error")) {
    if (nzchar(windows_powershell())) {
      return(windows_github_fetch_json(url, token = token))
    }
    if (nzchar(backend_gh())) {
      return(jsonlite::fromJSON(gh_api_request(url), simplifyVector = TRUE))
    }
    stop(conditionMessage(response), call. = FALSE)
  }

  if (response$status_code >= 300L) {
    if (response$status_code %in% c(401L, 403L, 404L) && nzchar(windows_powershell())) {
      return(windows_github_fetch_json(url, token = token))
    }
    if (response$status_code %in% c(401L, 403L, 404L) && nzchar(backend_gh())) {
      return(jsonlite::fromJSON(gh_api_request(url), simplifyVector = TRUE))
    }
    message <- trimws(rawToChar(response$content))
    if (!nzchar(message)) {
      message <- "empty response body"
    }
    if (response$status_code == 404L) {
      message <- paste(
        message,
        "If this repo is private, make sure GITHUB_PAT or FASTAUGSYNTH_GITHUB_PAT has repo access.",
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
  if (.Platform$OS.type == "windows" && nzchar(windows_powershell()) && nzchar(token)) {
    return(windows_github_download_file(url, destfile, token = token, accept = accept))
  }

  tryCatch(
    curl::curl_download(
      url = url,
      destfile = destfile,
      handle = curl::new_handle(
        followlocation = TRUE,
        httpheader = github_headers(token, accept = accept)
      ),
      quiet = TRUE
    ),
    error = function(e) {
      if (nzchar(windows_powershell())) {
        return(windows_github_download_file(url, destfile, token = token, accept = accept))
      }
      if (nzchar(backend_gh())) {
        return(gh_api_request(url, accept = NULL, output = destfile))
      }
      stop(conditionMessage(e), call. = FALSE)
    }
  )
}

backend_install_message <- function() {
  paste(
    "Backend library is not installed.",
    "Run fastaugsynth::backend_install() explicitly or set FASTAUGSYNTH_BACKEND_LIB.",
    "For this private repo, ensure GITHUB_PAT or FASTAUGSYNTH_GITHUB_PAT has repo access,",
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
  unpack_dir <- tempfile("fastaugsynth-artifact-")
  on.exit(unlink(c(zipfile, unpack_dir), recursive = TRUE, force = TRUE), add = TRUE)
  dir.create(unpack_dir, recursive = TRUE, showWarnings = FALSE)

  github_download_file(
    artifact$archive_download_url[[1]],
    zipfile,
    token = token,
    accept = "application/vnd.github+json"
  )
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

  auto_install <- tolower(Sys.getenv("FASTAUGSYNTH_AUTO_INSTALL_BACKEND", "true"))
  if (auto_install %in% c("1", "true", "yes")) {
    try(backend_install(quiet = TRUE), silent = TRUE)
    p <- backend_path()
    if (file.exists(p)) {
      return(p)
    }
  }

  stop(backend_install_message(), call. = FALSE)
}
