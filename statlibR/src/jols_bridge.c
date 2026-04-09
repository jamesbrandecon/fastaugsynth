#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

#include <dlfcn.h>
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

typedef void (*init_julia_fn)(int, char**);

static void* backend_handle = NULL;
static fit_ols_dense_fn fit_ols_dense_ptr = NULL;
static fit_ridge_loocv_dense_fn fit_ridge_loocv_dense_ptr = NULL;
static fit_synth_weights_fn fit_synth_weights_ptr = NULL;
static fit_ridge_augsynth_inner_fn fit_ridge_augsynth_inner_ptr = NULL;
static init_julia_fn init_julia_ptr = NULL;
static char backend_libpath[4096] = "";

static void* load_backend(const char* libpath) {
  if (backend_handle != NULL) {
    if (strcmp(backend_libpath, libpath) != 0) {
      Rf_error("Backend already loaded from '%s'; cannot switch to '%s' in the same R session", backend_libpath, libpath);
    }
    return backend_handle;
  }

  backend_handle = dlopen(libpath, RTLD_NOW | RTLD_GLOBAL);
  if (backend_handle == NULL) {
    Rf_error("Failed to load backend library at '%s': %s", libpath, dlerror());
  }

  init_julia_ptr = (init_julia_fn)dlsym(backend_handle, "init_julia");
  if (init_julia_ptr == NULL) {
    Rf_error("Symbol init_julia not found in backend library");
  }

  {
    char* julia_argv[] = {"metricsjl", NULL};
    init_julia_ptr(1, julia_argv);
  }

  snprintf(backend_libpath, sizeof(backend_libpath), "%s", libpath);
  return backend_handle;
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
  void* h = load_backend(libpath);

  if (fit_ols_dense_ptr == NULL) {
    fit_ols_dense_ptr = (fit_ols_dense_fn)dlsym(h, "fit_ols_dense");
  }
  if (fit_ols_dense_ptr == NULL) {
    Rf_error("Symbol fit_ols_dense not found in backend library");
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
  void* h = load_backend(libpath);

  if (fit_ridge_loocv_dense_ptr == NULL) {
    fit_ridge_loocv_dense_ptr = (fit_ridge_loocv_dense_fn)dlsym(h, "fit_ridge_loocv_dense");
  }
  if (fit_ridge_loocv_dense_ptr == NULL) {
    Rf_error("Symbol fit_ridge_loocv_dense not found in backend library");
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
  void* h = load_backend(libpath);

  if (fit_synth_weights_ptr == NULL) {
    fit_synth_weights_ptr = (fit_synth_weights_fn)dlsym(h, "fit_synth_weights");
  }
  if (fit_synth_weights_ptr == NULL) {
    Rf_error("Symbol fit_synth_weights not found in backend library");
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
  void* h = load_backend(libpath);

  if (fit_ridge_augsynth_inner_ptr == NULL) {
    fit_ridge_augsynth_inner_ptr = (fit_ridge_augsynth_inner_fn)dlsym(h, "fit_ridge_augsynth_inner");
  }
  if (fit_ridge_augsynth_inner_ptr == NULL) {
    Rf_error("Symbol fit_ridge_augsynth_inner not found in backend library");
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

static const R_CallMethodDef CallEntries[] = {
  {"C_jols_fit_xy", (DL_FUNC)&C_jols_fit_xy, 3},
  {"C_jridge_fit_xy", (DL_FUNC)&C_jridge_fit_xy, 4},
  {"C_jsynth_weights", (DL_FUNC)&C_jsynth_weights, 3},
  {"C_jridge_augsynth_inner", (DL_FUNC)&C_jridge_augsynth_inner, 9},
  {NULL, NULL, 0}
};

void R_init_metricsjl(DllInfo* dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
