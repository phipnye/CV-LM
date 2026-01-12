#include <Rcpp.h>
#include <RcppEigen.h>

#include <limits>

#include "CV-OLS-Fit.h"
#include "CV-Ridge-Fit.h"
#include "CV-Utils-utils.h"
#include "CV-WorkerModel.h"
#include "CV-engine.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-engine.h"

// [[Rcpp::export(name="cv.lm.rcpp")]]
double cvLMRCpp(const Eigen::Map<Eigen::VectorXd> y,
                const Eigen::Map<Eigen::MatrixXd> x, const int k0,
                const double lambda, const bool generalized, const int seed,
                const int nThreads, const double threshold,
                const bool centered) {
  // Determine which type of model we're fitting (this has potentially important
  // implications since OLS uses complete orthogonal decomposition whereas ridge
  // regression uses cholesky
  const bool useOLS{lambda <= threshold};

  if (generalized) {
    return useOLS ? CV::gcv<CV::OLS::Fit>(y, x, threshold)
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
  const int k{CV::Utils::kCheck(nrow, k0)};

  // LOOCV
  if (k == nrow) {
    return useOLS ? CV::loocv<CV::OLS::Fit>(y, x, threshold)
                  : CV::loocv<CV::Ridge::Fit>(y, x, lambda, centered);
  }

  // K-fold CV
  if (useOLS) {
    return CV::parCV<CV::OLS::WorkerModel>(y, x, k, seed, nThreads, threshold);
  }

  // Dispatch based on dimensionality (use dual form only if ncol strictly
  // exceeds the largest training size)
  if (x.cols() > (nrow - (nrow / k))) {
    return CV::parCV<CV::Ridge::Wide::WorkerModel>(y, x, k, seed, nThreads,
                                                   lambda);
  }

  // Otherwise, use primal form
  return CV::parCV<CV::Ridge::Narrow::WorkerModel>(y, x, k, seed, nThreads,
                                                   lambda);
}

// [[Rcpp::export(name="grid.search.rcpp")]]
Rcpp::List gridSearch(const Eigen::Map<Eigen::VectorXd> y,
                      const Eigen::Map<Eigen::MatrixXd> x, const int k0,
                      const double maxLambda, const double precision,
                      const bool generalized, const int seed,
                      const int nThreads, const double threshold,
                      const bool centered) {
  // Lightweight generator for creating lambda values
  const Grid::Generator lambdasGrid{maxLambda, precision, threshold};

  // Limit the grid size (2^32 max)
  if (lambdasGrid.size() > (1LL << 32)) {
    Rcpp::stop(
        "Lambda grid is too large. Please limit search size to something less "
        "than 2^32.");
  }

  // Optimal CV results in the form [CV, lambda]
  Grid::LambdaCV optimalPair;

  // Generalized CV
  if (generalized) {
    optimalPair = Grid::gcv(y, x, lambdasGrid, nThreads, threshold, centered);
  } else {
    // Determine a valid number of folds as close to the passed argument as
    // possible
    const int nrow{
        static_cast<int>(x.rows())};  // safe (cannot exceed 2^31 - 1)

    // Leave-one-out CV
    if (const int k{CV::Utils::kCheck(nrow, k0)}; k == nrow) {
      optimalPair =
          Grid::loocv(y, x, lambdasGrid, nThreads, threshold, centered);
    } else {  // K-fold CV
      optimalPair = Grid::kcv(y, x, k, lambdasGrid, seed, nThreads, threshold);
    }
  }

  return Rcpp::List::create(Rcpp::Named("CV") = optimalPair.cv,
                            Rcpp::Named("lambda") = optimalPair.lambda);
}
