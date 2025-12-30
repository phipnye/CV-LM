// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <Rcpp.h>
#include <RcppEigen.h>

#include "include/engineOLS.h"
#include "include/engineRidge.h"
#include "include/utils.h"

// [[Rcpp::export(name="cv.lm.rcpp")]]
Rcpp::DataFrame cvLMRCpp(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                         const int k0, const double lambda,
                         const bool generalized, const int seed,
                         const int nthreads) {
  // Preparation: Determine a valid number of folds as close to the passed
  // argument as possible
  const int nrow{static_cast<int>(x.rows())};
  const int k{generalized ? k0 : CV::Utils::kCheck(nrow, k0)};
  double cvVal{0.0};

  // OLS
  if (lambda == 0.0) {
    if (generalized) {
      cvVal = CV::OLS::gcv(y, x);
    } else if (k == nrow) {
      cvVal = CV::OLS::loocv(y, x);
    } else {
      cvVal = CV::OLS::parCV(y, x, k, seed, nthreads);
    }
  } else {  // Ridge regression
    if (generalized) {
      cvVal = CV::Ridge::gcv(y, x, lambda);
    } else if (k == nrow) {
      cvVal = CV::Ridge::loocv(y, x, lambda);
    } else {
      cvVal = CV::Ridge::parCV(y, x, k, lambda, seed, nthreads);
    }
  }

  return Rcpp::DataFrame::create(Rcpp::Named("K") = k,
                                 Rcpp::Named("CV") = cvVal,
                                 Rcpp::Named("seed") = seed);
}
