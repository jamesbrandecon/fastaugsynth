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

static void* load_backend(const char* libpath) {
  void* h = dlopen(libpath, RTLD_NOW | RTLD_LOCAL);
  if (h == NULL) {
    Rf_error("Failed to load backend library at '%s': %s", libpath, dlerror());
  }
  return h;
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

  fit_ols_dense_fn fit = (fit_ols_dense_fn)dlsym(h, "fit_ols_dense");
  if (fit == NULL) {
    dlclose(h);
    Rf_error("Symbol fit_ols_dense not found in backend library");
  }

  SEXP coef = PROTECT(Rf_allocVector(REALSXP, p));
  SEXP sigma2 = PROTECT(Rf_allocVector(REALSXP, 1));
  SEXP df_resid = PROTECT(Rf_allocVector(INTSXP, 1));
  SEXP rss = PROTECT(Rf_allocVector(REALSXP, 1));

  char errbuf[512];
  memset(errbuf, 0, sizeof(errbuf));
  int status = fit(n, p, REAL(X_), REAL(y_), REAL(coef), REAL(sigma2), INTEGER(df_resid), REAL(rss), errbuf, 512);

  dlclose(h);
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

  fit_ridge_loocv_dense_fn fit = (fit_ridge_loocv_dense_fn)dlsym(h, "fit_ridge_loocv_dense");
  if (fit == NULL) {
    dlclose(h);
    Rf_error("Symbol fit_ridge_loocv_dense not found in backend library");
  }

  SEXP coef = PROTECT(Rf_allocVector(REALSXP, p));
  SEXP best_lambda = PROTECT(Rf_allocVector(REALSXP, 1));
  SEXP loocv_mse = PROTECT(Rf_allocVector(REALSXP, 1));

  char errbuf[512];
  memset(errbuf, 0, sizeof(errbuf));
  int status = fit(n, p, REAL(X_), REAL(y_), nlambda, REAL(lambdas_), REAL(coef), REAL(best_lambda), REAL(loocv_mse), errbuf, 512);

  dlclose(h);
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

static const R_CallMethodDef CallEntries[] = {
  {"C_jols_fit_xy", (DL_FUNC)&C_jols_fit_xy, 3},
  {"C_jridge_fit_xy", (DL_FUNC)&C_jridge_fit_xy, 4},
  {NULL, NULL, 0}
};

void R_init_statlibR(DllInfo* dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
