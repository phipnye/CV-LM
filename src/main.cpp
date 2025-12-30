// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <Rcpp.h>
#include <RcppEigen.h>

#include "include/engineOLS.h"
#include "include/engineRidge.h"
#include "include/utils.h"

// [[Rcpp::export(name="cv.lm.rcpp")]]
double cvLMRCpp(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                const int k0, const double lambda, const bool generalized,
                const int seed, const int nThreads) {
  const bool useOLS{lambda == 0.0};

  if (generalized) {
    return useOLS ? CV::OLS::gcv(y, x) : CV::Ridge::gcv(y, x, lambda);
  }

  // Preparation: Determine a valid number of folds as close to the passed
  // argument as possible
  const int nrow{static_cast<int>(x.rows())};
  const int k{CV::Utils::kCheck(nrow, k0)};

  // LOOCV
  if (k == nrow) {
    return useOLS ? CV::OLS::loocv(y, x) : CV::Ridge::loocv(y, x, lambda);
  }

  // K-fold CV
  return useOLS ? CV::OLS::parCV(y, x, k, seed, nThreads)
                : CV::Ridge::parCV(y, x, k, lambda, seed, nThreads);
}
