#include <Rcpp.h>
#include <RcppEigen.h>

#include "CV-OLS-Fit.h"
#include "CV-Ridge-Fit.h"
#include "CV-WorkerModel.h"
#include "CV-engine.h"
#include "CV-utils.h"

// [[Rcpp::export(name="cv.lm.rcpp")]]
double cvLMRCpp(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                const int k0, const double lambda, const bool generalized,
                const int seed, const int nThreads, const bool centered) {
  const bool useOLS{lambda == 0.0};  // TO DO: Implement tolerance

  if (generalized) {
    return useOLS ? CV::gcv<CV::OLS::Fit>(y, x)
                  : CV::gcv<CV::Ridge::Fit>(y, x, lambda, centered);
  }

  // https://cran.r-project.org/doc/manuals/r-release/R-ints.html
  // "Matrices are stored as vectors and so were also limited to 2^31-1
  // elements. Now longer vectors are allowed on 64-bit platforms, matrices with
  // more elements are supported provided that each of the dimensions is no more
  // than 2^31-1."
  const int nrow{static_cast<int>(x.rows())};

  // Preparation: Determine a valid number of folds as close to the passed
  // argument as possible
  const int k{CV::kCheck(nrow, k0)};

  // LOOCV
  if (k == nrow) {
    return useOLS ? CV::loocv<CV::OLS::Fit>(y, x)
                  : CV::loocv<CV::Ridge::Fit>(y, x, lambda, centered);
  }

  // K-fold CV
  return useOLS ? CV::parCV<CV::OLS::WorkerModel>(y, x, k, seed, nThreads)
                : CV::parCV<CV::Ridge::WorkerModel>(y, x, k, seed, nThreads, lambda, x.cols());;
}

// // [[Rcpp::export(name="grid.search.rcpp")]]
// Rcpp::List gridSearch(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
//                       const int k0, const double maxLambda,
//                       const double precision, const bool generalized,
//                       const int seed, const int nThreads, const bool
//                       centered) {
// }
