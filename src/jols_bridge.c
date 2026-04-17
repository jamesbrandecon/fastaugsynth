#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <stdlib.h>
#include <string.h>

typedef int (*fit_ols_dense_fn)(int, int,
                                const double*, const double*,
                                double*, double*, int*, double*,
                                char*, int);

typedef int (*fit_ridge_loocv_dense_fn)(int, int,
                                        const double*, const double*,
                                        int, const double*,
                                        double*, double*, double*,
                                        char*, int);

typedef int (*fit_synth_weights_fn)(int, int,
                                    const double*, const double*,
                                    double*, char*, int);

typedef int (*fit_ridge_augsynth_inner_fn)(int, int,
                                           const double*, const double*,
                                           int, int, int,
                                           int, const double*,
                                           int, int,
                                           double*, double*, double*,
                                           double*, double*,
                                           char*, int);

typedef int (*jackknife_plus_fn)(int, int, int,
                                const double*, const double*, const double*,
                                double*, double*, double*, double*,
                                double, int, int, int, const double*,
                                int, int, char*, int);

typedef int (*jackknife_unit_std_fn)(int, int, int,
                                    const double*, const double*, const double*,
                                    double*, double*,
                                    int, int, const double*,
                                    int, int, char*, int);

typedef int (*conformal_inference_fn)(int, int, int,
                                      const double*, const double*, const double*,
                                      double*, double*, double*, double*,
                                      double, int, double, int, int,
                                      int, int, const double*,
                                      int, int, char*, int);

typedef int (*augsynth_inference_fn)(int, int, int,
                                      const double*, const double*, const double*,
                                      double*, double*, double*,
                                      double*, double*, double*,
                                      int, double, int, int, double, int, int, int,
                                      int, int, const double*,
                                      int, int, char*, int);

typedef int (*backend_thread_count_fn)(void);

typedef void (*init_julia_fn)(int, char**);

#ifdef _WIN32
typedef HMODULE backend_lib_handle_t;
#else
typedef void* backend_lib_handle_t;
#endif

static backend_lib_handle_t backend_handle = NULL;
static fit_ols_dense_fn fit_ols_dense_ptr = NULL;
static fit_ridge_loocv_dense_fn fit_ridge_loocv_dense_ptr = NULL;
static fit_synth_weights_fn fit_synth_weights_ptr = NULL;
static fit_ridge_augsynth_inner_fn fit_ridge_augsynth_inner_ptr = NULL;
static jackknife_plus_fn jackknife_plus_ptr = NULL;
static jackknife_unit_std_fn jackknife_unit_std_ptr = NULL;
static conformal_inference_fn conformal_inference_ptr = NULL;
static augsynth_inference_fn augsynth_inference_ptr = NULL;
static backend_thread_count_fn backend_thread_count_ptr = NULL;
static init_julia_fn init_julia_ptr = NULL;
static char backend_libpath[4096] = "";
static char backend_error_buffer[4096] = "";

static const char* backend_last_error(void) {
  return backend_error_buffer[0] != '\0' ? backend_error_buffer : "unknown error";
}

#ifdef _WIN32
static void set_windows_error_message(DWORD error_code) {
  DWORD status;

  backend_error_buffer[0] = '\0';
  if (error_code == 0) {
    snprintf(backend_error_buffer, sizeof(backend_error_buffer), "unknown error");
    return;
  }

  status = FormatMessageA(
    FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
    NULL,
    error_code,
    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
    backend_error_buffer,
    (DWORD)sizeof(backend_error_buffer),
    NULL
  );
  if (status == 0 || backend_error_buffer[0] == '\0') {
    snprintf(backend_error_buffer, sizeof(backend_error_buffer), "Windows error code %lu", (unsigned long)error_code);
    return;
  }

  while (status > 0 &&
         (backend_error_buffer[status - 1] == '\r' || backend_error_buffer[status - 1] == '\n')) {
    backend_error_buffer[status - 1] = '\0';
    status--;
  }
}

static void backend_dirname(const char* path, char* out, size_t out_size) {
  size_t len;

  if (out_size == 0) {
    return;
  }

  out[0] = '\0';
  if (path == NULL || path[0] == '\0') {
    snprintf(out, out_size, ".");
    return;
  }

  len = strlen(path);
  while (len > 0 && (path[len - 1] == '\\' || path[len - 1] == '/')) {
    len--;
  }
  while (len > 0 && path[len - 1] != '\\' && path[len - 1] != '/') {
    len--;
  }
  while (len > 1 && (path[len - 1] == '\\' || path[len - 1] == '/')) {
    len--;
  }

  if (len == 0) {
    snprintf(out, out_size, ".");
    return;
  }

  if (len >= out_size) {
    len = out_size - 1;
  }
  memcpy(out, path, len);
  out[len] = '\0';
}

static void backend_join_path(const char* left, const char* right, char* out, size_t out_size) {
  size_t len;

  if (out_size == 0) {
    return;
  }

  if (left == NULL || left[0] == '\0') {
    snprintf(out, out_size, "%s", right != NULL ? right : "");
    return;
  }
  if (right == NULL || right[0] == '\0') {
    snprintf(out, out_size, "%s", left);
    return;
  }

  len = strlen(left);
  if (left[len - 1] == '\\' || left[len - 1] == '/') {
    snprintf(out, out_size, "%s%s", left, right);
  } else {
    snprintf(out, out_size, "%s\\%s", left, right);
  }
}

static int backend_is_directory(const char* path) {
  DWORD attrs;

  if (path == NULL || path[0] == '\0') {
    return 0;
  }

  attrs = GetFileAttributesA(path);
  return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY);
}

static void backend_prepend_directory_to_path(const char* dir) {
  DWORD needed;
  char* current_path = NULL;
  char* new_path = NULL;

  if (!backend_is_directory(dir)) {
    return;
  }

  needed = GetEnvironmentVariableA("PATH", NULL, 0);
  if (needed > 0) {
    current_path = (char*)malloc((size_t)needed);
    if (current_path == NULL) {
      return;
    }
    if (GetEnvironmentVariableA("PATH", current_path, needed) == 0) {
      free(current_path);
      current_path = NULL;
      needed = 0;
    }
  }

  if (current_path != NULL && strstr(current_path, dir) != NULL) {
    free(current_path);
    return;
  }

  if (current_path != NULL && current_path[0] != '\0') {
    size_t new_len = strlen(dir) + 1 + strlen(current_path) + 1;
    new_path = (char*)malloc(new_len);
    if (new_path != NULL) {
      snprintf(new_path, new_len, "%s;%s", dir, current_path);
      SetEnvironmentVariableA("PATH", new_path);
      free(new_path);
    }
    free(current_path);
    return;
  }

  SetEnvironmentVariableA("PATH", dir);
  if (current_path != NULL) {
    free(current_path);
  }
}

static void backend_prepare_windows_search_path(const char* libpath) {
  char lib_dir[4096];
  char root_dir[4096];
  char candidate[4096];
  char parent_candidate[4096];

  backend_dirname(libpath, lib_dir, sizeof(lib_dir));
  backend_prepend_directory_to_path(lib_dir);

  backend_join_path(lib_dir, "julia", candidate, sizeof(candidate));
  backend_prepend_directory_to_path(candidate);

  backend_dirname(lib_dir, root_dir, sizeof(root_dir));

  backend_join_path(root_dir, "lib", candidate, sizeof(candidate));
  backend_prepend_directory_to_path(candidate);

  backend_join_path(candidate, "julia", parent_candidate, sizeof(parent_candidate));
  backend_prepend_directory_to_path(parent_candidate);

  backend_join_path(root_dir, "bin", candidate, sizeof(candidate));
  backend_prepend_directory_to_path(candidate);

  backend_join_path(root_dir, "julia", candidate, sizeof(candidate));
  backend_prepend_directory_to_path(candidate);
}
#endif

static backend_lib_handle_t backend_dlopen(const char* libpath) {
  backend_error_buffer[0] = '\0';
#ifdef _WIN32
  backend_prepare_windows_search_path(libpath);
  backend_handle = LoadLibraryExA(libpath, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (backend_handle == NULL) {
    set_windows_error_message(GetLastError());
  }
  return backend_handle;
#else
  backend_handle = dlopen(libpath, RTLD_NOW | RTLD_GLOBAL);
  if (backend_handle == NULL) {
    const char* err = dlerror();
    snprintf(backend_error_buffer, sizeof(backend_error_buffer), "%s", err != NULL ? err : "unknown error");
  }
  return backend_handle;
#endif
}

static void* backend_dlsym(backend_lib_handle_t handle, const char* symbol) {
  backend_error_buffer[0] = '\0';
#ifdef _WIN32
  FARPROC proc = GetProcAddress(handle, symbol);
  if (proc == NULL) {
    set_windows_error_message(GetLastError());
  }
  return (void*)proc;
#else
  void* proc;
  dlerror();
  proc = dlsym(handle, symbol);
  if (proc == NULL) {
    const char* err = dlerror();
    if (err != NULL) {
      snprintf(backend_error_buffer, sizeof(backend_error_buffer), "%s", err);
    }
  }
  return proc;
#endif
}

static const char* configured_julia_threads(void) {
  const char* threads = getenv("FASTAUGSYNTH_JULIA_THREADS");
  if (threads != NULL && threads[0] != '\0') {
    return threads;
  }
  threads = getenv("JULIA_NUM_THREADS");
  if (threads != NULL && threads[0] != '\0') {
    return threads;
  }
  return NULL;
}

static backend_lib_handle_t load_backend(const char* libpath) {
  if (backend_handle != NULL) {
    if (strcmp(backend_libpath, libpath) != 0) {
      Rf_error("Backend already loaded from '%s'; cannot switch to '%s' in the same R session", backend_libpath, libpath);
    }
    return backend_handle;
  }

  backend_handle = backend_dlopen(libpath);
  if (backend_handle == NULL) {
    Rf_error("Failed to load backend library at '%s': %s", libpath, backend_last_error());
  }

  init_julia_ptr = (init_julia_fn)backend_dlsym(backend_handle, "init_julia");
  if (init_julia_ptr == NULL) {
    Rf_error("Symbol init_julia not found in backend library: %s", backend_last_error());
  }

  {
    char program_name[] = "fastaugsynth";
    const char* threads = configured_julia_threads();
    if (threads != NULL) {
      char threads_arg[128];
      snprintf(threads_arg, sizeof(threads_arg), "--threads=%s", threads);
      char* julia_argv[] = {program_name, threads_arg, NULL};
      init_julia_ptr(2, julia_argv);
    } else {
      char* julia_argv[] = {program_name, NULL};
      init_julia_ptr(1, julia_argv);
    }
  }

  snprintf(backend_libpath, sizeof(backend_libpath), "%s", libpath);
  return backend_handle;
}

SEXP C_backend_thread_count(SEXP libpath_) {
  if (TYPEOF(libpath_) != STRSXP || Rf_length(libpath_) != 1) {
    Rf_error("libpath must be a scalar string");
  }

  const char* libpath = CHAR(STRING_ELT(libpath_, 0));
  backend_lib_handle_t h = load_backend(libpath);

  if (backend_thread_count_ptr == NULL) {
    backend_thread_count_ptr = (backend_thread_count_fn)backend_dlsym(h, "backend_thread_count");
  }
  if (backend_thread_count_ptr == NULL) {
    Rf_error("Symbol backend_thread_count not found in backend library: %s", backend_last_error());
  }

  return Rf_ScalarInteger(backend_thread_count_ptr());
}

SEXP C_jols_fit_xy(SEXP X_, SEXP y_, SEXP libpath_) {
  if (!Rf_isMatrix(X_) || TYPEOF(X_) != REALSXP) Rf_error("X must be a numeric matrix");
  if (TYPEOF(y_) != REALSXP) Rf_error("y must be numeric");
  if (TYPEOF(libpath_) != STRSXP || Rf_length(libpath_) != 1) Rf_error("libpath must be a scalar string");

  SEXP dim = Rf_getAttrib(X_, R_DimSymbol);
  int n = INTEGER(dim)[0];
  int p = INTEGER(dim)[1];
  if (Rf_length(y_) != n) Rf_error("nrow(X) must equal length(y)");

  const char* libpath = CHAR(STRING_ELT(libpath_, 0));
  backend_lib_handle_t h = load_backend(libpath);

  if (fit_ols_dense_ptr == NULL) {
    fit_ols_dense_ptr = (fit_ols_dense_fn)backend_dlsym(h, "fit_ols_dense");
  }
  if (fit_ols_dense_ptr == NULL) {
    Rf_error("Symbol fit_ols_dense not found in backend library: %s", backend_last_error());
  }

  SEXP coef = PROTECT(Rf_allocVector(REALSXP, p));
  SEXP sigma2 = PROTECT(Rf_allocVector(REALSXP, 1));
  SEXP df_resid = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP rss = PROTECT(Rf_allocVector(REALSXP, 1));

  char errbuf[512];
  memset(errbuf, 0, sizeof(errbuf));
  int status = fit_ols_dense_ptr(n, p, REAL(X_), REAL(y_), REAL(coef), REAL(sigma2), INTEGER(df_resid), REAL(rss), errbuf, 512);
  if (status != 0) Rf_error("Backend fit_ols_dense failed with status %d: %s", status, errbuf);

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
  SET_VECTOR_ELT(out, 0, coef);
  SET_VECTOR_ELT(out, 1, sigma2);
  SET_VECTOR_ELT(out, 2, df_resid);
  SET_VECTOR_ELT(out, 3, rss);

  SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));
  SET_STRING_ELT(names, 0, Rf_mkChar("coefficients"));
  SET_STRING_ELT(names, 1, Rf_mkChar("sigma2"));
  SET_STRING_ELT(names, 2, Rf_mkChar("df_resid"));
  SET_STRING_ELT(names, 3, Rf_mkChar("rss"));
  Rf_setAttrib(out, R_NamesSymbol, names);

  UNPROTECT(6);
  return out;
}

SEXP C_jridge_fit_xy(SEXP X_, SEXP y_, SEXP lambdas_, SEXP libpath_) {
  if (!Rf_isMatrix(X_) || TYPEOF(X_) != REALSXP) Rf_error("X must be a numeric matrix");
  if (TYPEOF(y_) != REALSXP) Rf_error("y must be numeric");
  if (TYPEOF(lambdas_) != REALSXP) Rf_error("lambdas must be numeric");
  if (TYPEOF(libpath_) != STRSXP || Rf_length(libpath_) != 1) Rf_error("libpath must be a scalar string");

  SEXP dim = Rf_getAttrib(X_, R_DimSymbol);
  int n = INTEGER(dim)[0];
  int p = INTEGER(dim)[1];
  int nlambda = Rf_length(lambdas_);
  if (Rf_length(y_) != n) Rf_error("nrow(X) must equal length(y)");
  if (nlambda <= 0) Rf_error("lambdas must be non-empty");

  const char* libpath = CHAR(STRING_ELT(libpath_, 0));
  backend_lib_handle_t h = load_backend(libpath);

  if (fit_ridge_loocv_dense_ptr == NULL) {
    fit_ridge_loocv_dense_ptr = (fit_ridge_loocv_dense_fn)backend_dlsym(h, "fit_ridge_loocv_dense");
  }
  if (fit_ridge_loocv_dense_ptr == NULL) {
    Rf_error("Symbol fit_ridge_loocv_dense not found in backend library: %s", backend_last_error());
  }

  SEXP coef = PROTECT(Rf_allocVector(REALSXP, p));
  SEXP best_lambda = PROTECT(Rf_allocVector(REALSXP, 1));
  SEXP loocv_mse = PROTECT(Rf_allocVector(REALSXP, 1));

  char errbuf[512];
  memset(errbuf, 0, sizeof(errbuf));
  int status = fit_ridge_loocv_dense_ptr(n, p, REAL(X_), REAL(y_), nlambda, REAL(lambdas_), REAL(coef), REAL(best_lambda), REAL(loocv_mse), errbuf, 512);
  if (status != 0) Rf_error("Backend fit_ridge_loocv_dense failed with status %d: %s", status, errbuf);

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 3));
  SET_VECTOR_ELT(out, 0, coef);
  SET_VECTOR_ELT(out, 1, best_lambda);
  SET_VECTOR_ELT(out, 2, loocv_mse);

  SEXP names = PROTECT(Rf_allocVector(STRSXP, 3));
  SET_STRING_ELT(names, 0, Rf_mkChar("coefficients"));
  SET_STRING_ELT(names, 1, Rf_mkChar("best_lambda"));
  SET_STRING_ELT(names, 2, Rf_mkChar("loocv_mse"));
  Rf_setAttrib(out, R_NamesSymbol, names);

  UNPROTECT(5);
  return out;
}

SEXP C_jsynth_weights(SEXP donors_, SEXP target_, SEXP libpath_) {
  if (!Rf_isMatrix(donors_) || TYPEOF(donors_) != REALSXP) Rf_error("donors must be a numeric matrix");
  if (TYPEOF(target_) != REALSXP) Rf_error("target must be numeric");
  if (TYPEOF(libpath_) != STRSXP || Rf_length(libpath_) != 1) Rf_error("libpath must be a scalar string");

  SEXP dim = Rf_getAttrib(donors_, R_DimSymbol);
  int n0 = INTEGER(dim)[0];
  int t0 = INTEGER(dim)[1];
  if (Rf_length(target_) != t0) Rf_error("ncol(donors) must equal length(target)");

  const char* libpath = CHAR(STRING_ELT(libpath_, 0));
  backend_lib_handle_t h = load_backend(libpath);

  if (fit_synth_weights_ptr == NULL) {
    fit_synth_weights_ptr = (fit_synth_weights_fn)backend_dlsym(h, "fit_synth_weights");
  }
  if (fit_synth_weights_ptr == NULL) {
    Rf_error("Symbol fit_synth_weights not found in backend library: %s", backend_last_error());
  }

  SEXP weights = PROTECT(Rf_allocVector(REALSXP, n0));
  char errbuf[512];
  memset(errbuf, 0, sizeof(errbuf));

  int status = fit_synth_weights_ptr(
    n0, t0, REAL(donors_), REAL(target_), REAL(weights), errbuf, 512
  );
  if (status != 0) Rf_error("Backend fit_synth_weights failed with status %d: %s", status, errbuf);

  UNPROTECT(1);
  return weights;
}

SEXP C_jridge_augsynth_inner(SEXP Xc_, SEXP x1_, SEXP ridge_, SEXP scm_,
                             SEXP select_lambda_, SEXP lambdas_,
                             SEXP holdout_length_, SEXP min1se_,
                             SEXP libpath_) {
  if (!Rf_isMatrix(Xc_) || TYPEOF(Xc_) != REALSXP) Rf_error("Xc must be a numeric matrix");
  if (TYPEOF(x1_) != REALSXP) Rf_error("x1 must be numeric");
  if (TYPEOF(lambdas_) != REALSXP) Rf_error("lambdas must be numeric");
  if (TYPEOF(libpath_) != STRSXP || Rf_length(libpath_) != 1) Rf_error("libpath must be a scalar string");

  SEXP dim = Rf_getAttrib(Xc_, R_DimSymbol);
  int n0 = INTEGER(dim)[0];
  int t0 = INTEGER(dim)[1];
  int nlambda = Rf_length(lambdas_);

  if (Rf_length(x1_) != t0) Rf_error("ncol(Xc) must equal length(x1)");

  const char* libpath = CHAR(STRING_ELT(libpath_, 0));
  backend_lib_handle_t h = load_backend(libpath);

  if (fit_ridge_augsynth_inner_ptr == NULL) {
    fit_ridge_augsynth_inner_ptr = (fit_ridge_augsynth_inner_fn)backend_dlsym(h, "fit_ridge_augsynth_inner");
  }
  if (fit_ridge_augsynth_inner_ptr == NULL) {
    Rf_error("Symbol fit_ridge_augsynth_inner not found in backend library: %s", backend_last_error());
  }

  SEXP weights = PROTECT(Rf_allocVector(REALSXP, n0));
  SEXP synw = PROTECT(Rf_allocVector(REALSXP, n0));
  SEXP lambda = PROTECT(Rf_allocVector(REALSXP, 1));
  SEXP lambda_errors = PROTECT(Rf_allocVector(REALSXP, nlambda));
  SEXP lambda_errors_se = PROTECT(Rf_allocVector(REALSXP, nlambda));

  char errbuf[512];
  memset(errbuf, 0, sizeof(errbuf));

  int status = fit_ridge_augsynth_inner_ptr(
    n0, t0,
    REAL(Xc_), REAL(x1_),
    Rf_asLogical(ridge_),
    Rf_asLogical(scm_),
    Rf_asLogical(select_lambda_),
    nlambda, REAL(lambdas_),
    Rf_asInteger(holdout_length_),
    Rf_asLogical(min1se_),
    REAL(weights), REAL(synw), REAL(lambda),
    REAL(lambda_errors), REAL(lambda_errors_se),
    errbuf, 512
  );
  if (status != 0) Rf_error("Backend fit_ridge_augsynth_inner failed with status %d: %s", status, errbuf);

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 5));
  SET_VECTOR_ELT(out, 0, weights);
  SET_VECTOR_ELT(out, 1, synw);
  SET_VECTOR_ELT(out, 2, lambda);
  SET_VECTOR_ELT(out, 3, lambda_errors);
  SET_VECTOR_ELT(out, 4, lambda_errors_se);

  SEXP names = PROTECT(Rf_allocVector(STRSXP, 5));
  SET_STRING_ELT(names, 0, Rf_mkChar("weights"));
  SET_STRING_ELT(names, 1, Rf_mkChar("synw"));
  SET_STRING_ELT(names, 2, Rf_mkChar("lambda"));
  SET_STRING_ELT(names, 3, Rf_mkChar("lambda_errors"));
  SET_STRING_ELT(names, 4, Rf_mkChar("lambda_errors_se"));
  Rf_setAttrib(out, R_NamesSymbol, names);

  UNPROTECT(7);
  return out;
}

SEXP C_jackknife_plus(SEXP X_, SEXP y_, SEXP trt_, SEXP ridge_,
                      SEXP scm_, SEXP lambda_, SEXP alpha_, SEXP conservative_,
                      SEXP holdout_length_, SEXP min1se_, SEXP libpath_) {
  if (!Rf_isMatrix(X_) || TYPEOF(X_) != REALSXP) Rf_error("X must be a numeric matrix");
  if (!Rf_isMatrix(y_) || TYPEOF(y_) != REALSXP) Rf_error("y must be a numeric matrix");
  if (TYPEOF(trt_) != REALSXP) Rf_error("trt must be numeric");
  if (TYPEOF(lambda_) != REALSXP) Rf_error("lambda must be numeric");
  if (Rf_length(lambda_) != 1) Rf_error("lambda must be length 1");
  if (TYPEOF(libpath_) != STRSXP || Rf_length(libpath_) != 1) Rf_error("libpath must be a scalar string");

  SEXP xdim = Rf_getAttrib(X_, R_DimSymbol);
  int n = INTEGER(xdim)[0];
  int t0 = INTEGER(xdim)[1];
  SEXP ydim = Rf_getAttrib(y_, R_DimSymbol);
  int tpost = INTEGER(ydim)[1];
  if (INTEGER(ydim)[0] != n) Rf_error("nrow(y) must equal nrow(X)");
  if (Rf_length(trt_) != n) Rf_error("length(trt) must equal nrow(X)");

  const char* libpath = CHAR(STRING_ELT(libpath_, 0));
  backend_lib_handle_t h = load_backend(libpath);

  if (jackknife_plus_ptr == NULL) {
    jackknife_plus_ptr = (jackknife_plus_fn)backend_dlsym(h, "jackknife_plus");
  }
  if (jackknife_plus_ptr == NULL) {
    Rf_error("Symbol jackknife_plus not found in backend library: %s", backend_last_error());
  }

  int total = t0 + tpost + 1;
  SEXP att = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP lb = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP ub = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP heldout_att = PROTECT(Rf_allocVector(REALSXP, total));
  char errbuf[512];
  memset(errbuf, 0, sizeof(errbuf));

  int status = jackknife_plus_ptr(
    n, t0, tpost,
    REAL(X_), REAL(y_), REAL(trt_),
    REAL(att), REAL(lb), REAL(ub), REAL(heldout_att),
    Rf_asReal(alpha_), Rf_asLogical(conservative_),
    Rf_asLogical(ridge_), Rf_asLogical(scm_),
    REAL(lambda_),
    Rf_asInteger(holdout_length_),
    Rf_asLogical(min1se_),
    errbuf, 512
  );
  if (status != 0) Rf_error("Backend jackknife_plus failed with status %d: %s", status, errbuf);

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 4));
  SET_VECTOR_ELT(out, 0, att);
  SET_VECTOR_ELT(out, 1, lb);
  SET_VECTOR_ELT(out, 2, ub);
  SET_VECTOR_ELT(out, 3, heldout_att);

  SEXP names = PROTECT(Rf_allocVector(STRSXP, 4));
  SET_STRING_ELT(names, 0, Rf_mkChar("att"));
  SET_STRING_ELT(names, 1, Rf_mkChar("lb"));
  SET_STRING_ELT(names, 2, Rf_mkChar("ub"));
  SET_STRING_ELT(names, 3, Rf_mkChar("heldout_att"));
  Rf_setAttrib(out, R_NamesSymbol, names);

  UNPROTECT(6);
  return out;
}

SEXP C_jackknife_unit_std(SEXP X_, SEXP y_, SEXP trt_, SEXP ridge_,
                          SEXP scm_, SEXP lambda_, SEXP holdout_length_, SEXP min1se_, SEXP libpath_) {
  if (!Rf_isMatrix(X_) || TYPEOF(X_) != REALSXP) Rf_error("X must be a numeric matrix");
  if (!Rf_isMatrix(y_) || TYPEOF(y_) != REALSXP) Rf_error("y must be a numeric matrix");
  if (TYPEOF(trt_) != REALSXP) Rf_error("trt must be numeric");
  if (TYPEOF(lambda_) != REALSXP) Rf_error("lambda must be numeric");
  if (Rf_length(lambda_) != 1) Rf_error("lambda must be length 1");
  if (TYPEOF(libpath_) != STRSXP || Rf_length(libpath_) != 1) Rf_error("libpath must be a scalar string");

  SEXP xdim = Rf_getAttrib(X_, R_DimSymbol);
  int n = INTEGER(xdim)[0];
  int t0 = INTEGER(xdim)[1];
  SEXP ydim = Rf_getAttrib(y_, R_DimSymbol);
  int tpost = INTEGER(ydim)[1];
  if (INTEGER(ydim)[0] != n) Rf_error("nrow(y) must equal nrow(X)");
  if (Rf_length(trt_) != n) Rf_error("length(trt) must equal nrow(X)");

  const char* libpath = CHAR(STRING_ELT(libpath_, 0));
  backend_lib_handle_t h = load_backend(libpath);

  if (jackknife_unit_std_ptr == NULL) {
    jackknife_unit_std_ptr = (jackknife_unit_std_fn)backend_dlsym(h, "jackknife_unit_std");
  }
  if (jackknife_unit_std_ptr == NULL) {
    Rf_error("Symbol jackknife_unit_std not found in backend library: %s", backend_last_error());
  }

  int total = t0 + tpost + 1;
  SEXP att = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP se = PROTECT(Rf_allocVector(REALSXP, total));
  char errbuf[512];
  memset(errbuf, 0, sizeof(errbuf));

  int status = jackknife_unit_std_ptr(
    n, t0, tpost,
    REAL(X_), REAL(y_), REAL(trt_),
    REAL(att), REAL(se),
    Rf_asLogical(ridge_), Rf_asLogical(scm_),
    REAL(lambda_),
    Rf_asInteger(holdout_length_),
    Rf_asLogical(min1se_),
    errbuf, 512
  );
  if (status != 0) Rf_error("Backend jackknife_unit_std failed with status %d: %s", status, errbuf);

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(out, 0, att);
  SET_VECTOR_ELT(out, 1, se);

  SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(names, 0, Rf_mkChar("att"));
  SET_STRING_ELT(names, 1, Rf_mkChar("se"));
  Rf_setAttrib(out, R_NamesSymbol, names);

  UNPROTECT(4);
  return out;
}

SEXP C_conformal_inference(SEXP X_, SEXP y_, SEXP trt_, SEXP ridge_,
                           SEXP scm_, SEXP lambda_, SEXP alpha_, SEXP type_,
                           SEXP q_, SEXP ns_, SEXP grid_size_, SEXP holdout_length_,
                           SEXP min1se_, SEXP libpath_) {
  if (!Rf_isMatrix(X_) || TYPEOF(X_) != REALSXP) Rf_error("X must be a numeric matrix");
  if (!Rf_isMatrix(y_) || TYPEOF(y_) != REALSXP) Rf_error("y must be a numeric matrix");
  if (TYPEOF(trt_) != REALSXP) Rf_error("trt must be numeric");
  if (TYPEOF(lambda_) != REALSXP) Rf_error("lambda must be numeric");
  if (Rf_length(lambda_) != 1) Rf_error("lambda must be length 1");
  if (TYPEOF(alpha_) != REALSXP) Rf_error("alpha must be numeric");
  if (TYPEOF(q_) != REALSXP) Rf_error("q must be numeric");
  if (TYPEOF(ns_) != INTSXP && TYPEOF(ns_) != REALSXP) Rf_error("ns must be integer");
  if (TYPEOF(grid_size_) != INTSXP && TYPEOF(grid_size_) != REALSXP) Rf_error("grid_size must be integer");
  if (TYPEOF(libpath_) != STRSXP || Rf_length(libpath_) != 1) Rf_error("libpath must be a scalar string");

  SEXP xdim = Rf_getAttrib(X_, R_DimSymbol);
  int n = INTEGER(xdim)[0];
  int t0 = INTEGER(xdim)[1];
  SEXP ydim = Rf_getAttrib(y_, R_DimSymbol);
  int tpost = INTEGER(ydim)[1];
  if (INTEGER(ydim)[0] != n) Rf_error("nrow(y) must equal nrow(X)");
  if (Rf_length(trt_) != n) Rf_error("length(trt) must equal nrow(X)");

  int type = 1;
  if (TYPEOF(type_) == STRSXP && Rf_length(type_) == 1) {
    const char* type_str = CHAR(STRING_ELT(type_, 0));
    type = (strcmp(type_str, "iid") == 0) ? 0 : 1;
  } else if (TYPEOF(type_) == INTSXP) {
    type = Rf_asInteger(type_);
  } else if (TYPEOF(type_) == REALSXP) {
    type = (int)Rf_asReal(type_);
  } else {
    Rf_error("type must be a string or integer");
  }

  const char* libpath = CHAR(STRING_ELT(libpath_, 0));
  backend_lib_handle_t h = load_backend(libpath);

  if (conformal_inference_ptr == NULL) {
    conformal_inference_ptr = (conformal_inference_fn)backend_dlsym(h, "conformal_inference");
  }
  if (conformal_inference_ptr == NULL) {
    Rf_error("Symbol conformal_inference not found in backend library: %s", backend_last_error());
  }

  int total = t0 + tpost + 1;
  SEXP att = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP lb = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP ub = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP pval = PROTECT(Rf_allocVector(REALSXP, total));
  char errbuf[512];
  memset(errbuf, 0, sizeof(errbuf));

  int status = conformal_inference_ptr(
    n, t0, tpost,
    REAL(X_), REAL(y_), REAL(trt_),
    REAL(att), REAL(lb), REAL(ub), REAL(pval),
    Rf_asReal(alpha_), type, Rf_asReal(q_), Rf_asInteger(ns_), Rf_asInteger(grid_size_),
    Rf_asLogical(ridge_), Rf_asLogical(scm_),
    REAL(lambda_),
    Rf_asInteger(holdout_length_),
    Rf_asLogical(min1se_),
    errbuf, 512
  );
  if (status != 0) Rf_error("Backend conformal_inference failed with status %d: %s", status, errbuf);

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 5));
  SET_VECTOR_ELT(out, 0, att);
  SET_VECTOR_ELT(out, 1, lb);
  SET_VECTOR_ELT(out, 2, ub);
  SET_VECTOR_ELT(out, 3, pval);

  SEXP alpha_out = PROTECT(Rf_allocVector(REALSXP, 1));
  REAL(alpha_out)[0] = Rf_asReal(alpha_);
  SET_VECTOR_ELT(out, 4, alpha_out);

  SEXP names = PROTECT(Rf_allocVector(STRSXP, 5));
  SET_STRING_ELT(names, 0, Rf_mkChar("att"));
  SET_STRING_ELT(names, 1, Rf_mkChar("lb"));
  SET_STRING_ELT(names, 2, Rf_mkChar("ub"));
  SET_STRING_ELT(names, 3, Rf_mkChar("p_val"));
  SET_STRING_ELT(names, 4, Rf_mkChar("alpha"));
  Rf_setAttrib(out, R_NamesSymbol, names);

  UNPROTECT(7);
  return out;
}

SEXP C_augsynth_inference(SEXP X_, SEXP y_, SEXP trt_, SEXP inf_type_,
                          SEXP alpha_, SEXP conservative_,
                          SEXP type_, SEXP q_, SEXP ns_, SEXP grid_size_, SEXP conformal_mode_,
                          SEXP ridge_, SEXP scm_, SEXP lambda_,
                          SEXP holdout_length_, SEXP min1se_,
                          SEXP libpath_) {
  if (!Rf_isMatrix(X_) || TYPEOF(X_) != REALSXP) Rf_error("X must be a numeric matrix");
  if (!Rf_isMatrix(y_) || TYPEOF(y_) != REALSXP) Rf_error("y must be a numeric matrix");
  if (TYPEOF(trt_) != REALSXP) Rf_error("trt must be numeric");
  if (TYPEOF(alpha_) != REALSXP) Rf_error("alpha must be numeric");
  if (TYPEOF(conservative_) != LGLSXP && TYPEOF(conservative_) != INTSXP) Rf_error("conservative must be logical");
  if (TYPEOF(q_) != REALSXP) Rf_error("q must be numeric");
  if (TYPEOF(ns_) != INTSXP && TYPEOF(ns_) != REALSXP) Rf_error("ns must be integer");
  if (TYPEOF(grid_size_) != INTSXP && TYPEOF(grid_size_) != REALSXP) Rf_error("grid_size must be integer");
  if (TYPEOF(conformal_mode_) != STRSXP && TYPEOF(conformal_mode_) != INTSXP &&
      TYPEOF(conformal_mode_) != REALSXP && TYPEOF(conformal_mode_) != NILSXP) {
    Rf_error("conformal_mode must be a string, integer, or NULL");
  }
  if (TYPEOF(ridge_) != LGLSXP && TYPEOF(ridge_) != INTSXP) Rf_error("ridge must be logical");
  if (TYPEOF(scm_) != LGLSXP && TYPEOF(scm_) != INTSXP) Rf_error("scm must be logical");
  if (TYPEOF(lambda_) != REALSXP) Rf_error("lambda must be numeric");
  if (Rf_length(lambda_) != 1) Rf_error("lambda must be length 1");
  if (TYPEOF(holdout_length_) != INTSXP && TYPEOF(holdout_length_) != REALSXP) Rf_error("holdout_length must be integer");
  if (TYPEOF(min1se_) != LGLSXP && TYPEOF(min1se_) != INTSXP) Rf_error("min1se must be logical");
  if (TYPEOF(libpath_) != STRSXP || Rf_length(libpath_) != 1) Rf_error("libpath must be a scalar string");

  int inf_type = 0;
  if (TYPEOF(inf_type_) == STRSXP && Rf_length(inf_type_) == 1) {
    const char* type = CHAR(STRING_ELT(inf_type_, 0));
    if (strcmp(type, "jackknife") == 0) {
      inf_type = 1;
    } else if (strcmp(type, "jackknife+") == 0 || strcmp(type, "jackknife_plus") == 0) {
      inf_type = 2;
    } else if (strcmp(type, "conformal") == 0) {
      inf_type = 3;
    } else {
      Rf_error("inf_type must be one of: jackknife, jackknife+, conformal");
    }
  } else if (TYPEOF(inf_type_) == INTSXP || TYPEOF(inf_type_) == REALSXP) {
    inf_type = (int)Rf_asInteger(inf_type_);
  } else {
    Rf_error("inf_type must be a string or integer");
  }

  SEXP xdim = Rf_getAttrib(X_, R_DimSymbol);
  int n = INTEGER(xdim)[0];
  int t0 = INTEGER(xdim)[1];
  SEXP ydim = Rf_getAttrib(y_, R_DimSymbol);
  int tpost = INTEGER(ydim)[1];
  if (INTEGER(ydim)[0] != n) Rf_error("nrow(y) must equal nrow(X)");
  if (Rf_length(trt_) != n) Rf_error("length(trt) must equal nrow(X)");

  const char* libpath = CHAR(STRING_ELT(libpath_, 0));
  backend_lib_handle_t h = load_backend(libpath);

  if (augsynth_inference_ptr == NULL) {
    augsynth_inference_ptr = (augsynth_inference_fn)backend_dlsym(h, "augsynth_inference");
  }
  if (augsynth_inference_ptr == NULL) {
    Rf_error("Symbol augsynth_inference not found in backend library: %s", backend_last_error());
  }

  int total = t0 + tpost + 1;
  SEXP att = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP lb = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP ub = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP se = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP heldout_att = PROTECT(Rf_allocVector(REALSXP, total));
  SEXP p_val = PROTECT(Rf_allocVector(REALSXP, total));
  int type = 1;
  if (TYPEOF(type_) == STRSXP && Rf_length(type_) == 1) {
    const char* type_str = CHAR(STRING_ELT(type_, 0));
    type = (strcmp(type_str, "iid") == 0) ? 0 : 1;
  } else if (TYPEOF(type_) == INTSXP) {
    type = Rf_asInteger(type_);
  } else if (TYPEOF(type_) == REALSXP) {
    type = (int)Rf_asReal(type_);
  } else if (TYPEOF(type_) == NILSXP) {
    type = 1;
  } else {
    Rf_error("type must be a string, integer, or NULL");
  }

  int conformal_mode = 0;
  if (TYPEOF(conformal_mode_) == STRSXP && Rf_length(conformal_mode_) == 1) {
    const char* mode_str = CHAR(STRING_ELT(conformal_mode_, 0));
    if (strcmp(mode_str, "fast") == 0 || strcmp(mode_str, "adaptive") == 0) {
      conformal_mode = 0;
    } else if (strcmp(mode_str, "reference") == 0 || strcmp(mode_str, "reference_conformal") == 0 ||
               strcmp(mode_str, "grid") == 0 || strcmp(mode_str, "fixed_grid") == 0) {
      conformal_mode = 1;
    } else {
      Rf_error("conformal_mode must be one of: fast, reference");
    }
  } else if (TYPEOF(conformal_mode_) == INTSXP || TYPEOF(conformal_mode_) == REALSXP) {
    conformal_mode = (int)Rf_asInteger(conformal_mode_);
  } else if (TYPEOF(conformal_mode_) == NILSXP) {
    conformal_mode = 0;
  }

  char errbuf[512];
  memset(errbuf, 0, sizeof(errbuf));

  int status = augsynth_inference_ptr(
    n, t0, tpost,
    REAL(X_), REAL(y_), REAL(trt_),
    REAL(att), REAL(lb), REAL(ub),
    REAL(se), REAL(heldout_att), REAL(p_val),
    inf_type, Rf_asReal(alpha_), Rf_asLogical(conservative_), type,
    Rf_asReal(q_), Rf_asInteger(ns_), Rf_asInteger(grid_size_), conformal_mode,
    Rf_asLogical(ridge_), Rf_asLogical(scm_),
    REAL(lambda_),
    Rf_asInteger(holdout_length_),
    Rf_asLogical(min1se_),
    errbuf, 512
  );
  if (status != 0) Rf_error("Backend augsynth_inference failed with status %d: %s", status, errbuf);

  SEXP out = PROTECT(Rf_allocVector(VECSXP, 7));
  SET_VECTOR_ELT(out, 0, att);
  SET_VECTOR_ELT(out, 1, lb);
  SET_VECTOR_ELT(out, 2, ub);
  SET_VECTOR_ELT(out, 3, se);
  SET_VECTOR_ELT(out, 4, heldout_att);
  SET_VECTOR_ELT(out, 5, p_val);

  SEXP alpha_out = PROTECT(Rf_allocVector(REALSXP, 1));
  REAL(alpha_out)[0] = Rf_asReal(alpha_);
  SET_VECTOR_ELT(out, 6, alpha_out);

  SEXP names = PROTECT(Rf_allocVector(STRSXP, 7));
  SET_STRING_ELT(names, 0, Rf_mkChar("att"));
  SET_STRING_ELT(names, 1, Rf_mkChar("lb"));
  SET_STRING_ELT(names, 2, Rf_mkChar("ub"));
  SET_STRING_ELT(names, 3, Rf_mkChar("se"));
  SET_STRING_ELT(names, 4, Rf_mkChar("heldout_att"));
  SET_STRING_ELT(names, 5, Rf_mkChar("p_val"));
  SET_STRING_ELT(names, 6, Rf_mkChar("alpha"));
  Rf_setAttrib(out, R_NamesSymbol, names);

  UNPROTECT(9);
  return out;
}

static const R_CallMethodDef CallEntries[] = {
  {"C_backend_thread_count", (DL_FUNC)&C_backend_thread_count, 1},
  {"C_jols_fit_xy", (DL_FUNC)&C_jols_fit_xy, 3},
  {"C_jridge_fit_xy", (DL_FUNC)&C_jridge_fit_xy, 4},
  {"C_jsynth_weights", (DL_FUNC)&C_jsynth_weights, 3},
  {"C_jridge_augsynth_inner", (DL_FUNC)&C_jridge_augsynth_inner, 9},
  {"C_jackknife_plus", (DL_FUNC)&C_jackknife_plus, 11},
  {"C_jackknife_unit_std", (DL_FUNC)&C_jackknife_unit_std, 9},
  {"C_conformal_inference", (DL_FUNC)&C_conformal_inference, 13},
  {"C_augsynth_inference", (DL_FUNC)&C_augsynth_inference, 17},
  {NULL, NULL, 0}
};

void R_init_fastaugsynth(DllInfo* dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
