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
  const int k{CV::Utils::kCheck(nrow, k0)};

  // LOOCV
  if (k == nrow) {
    return useOLS ? CV::loocv<CV::OLS::Fit>(y, x)
                  : CV::loocv<CV::Ridge::Fit>(y, x, lambda, centered);
  }

  // K-fold CV
  return useOLS ? CV::parCV<CV::OLS::WorkerModel>(y, x, k, seed, nThreads)
                : CV::parCV<CV::Ridge::WorkerModel>(y, x, k, seed, nThreads,
                                                    lambda);
}

// [[Rcpp::export(name="grid.search.rcpp")]]
Rcpp::List gridSearch(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                      const int k0, const double maxLambda,
                      const double precision, const bool generalized,
                      const int seed, const int nThreads, const bool centered) {
  // Light weight generator for creating lambda values
  const Grid::Generator lambdasGrid{maxLambda, precision};

  // Limit the grid size
  if (static_cast<int>(lambdasGrid.size()) > std::numeric_limits<int>::max()) {
    Rcpp::stop("Lambda grid is too large.");
  }

  // Optimal CV results in the form [CV, lambda]
  Grid::LambdaCV results{0.0, std::numeric_limits<double>::infinity()};

  // Generalized CV
  if (generalized) {
    results = Grid::gcv(y, x, lambdasGrid, nThreads, centered);
  } else {
    // Determine a valid number of folds as close to the passed argument as
    // possible
    const int nrow{
        static_cast<int>(x.rows())};  // safe (cannot exceed 2^31 - 1)

    // Leave-one-out CV
    if (const int k{CV::Utils::kCheck(nrow, k0)}; k == nrow) {
      results = Grid::loocv(y, x, lambdasGrid, nThreads, centered);
    } else {  // K-fold CV
      results = Grid::kcv(y, x, k, lambdasGrid, seed, nThreads);
    }
  }

  return Rcpp::List::create(Rcpp::Named("CV") = results.cv,
                            Rcpp::Named("lambda") = results.lambda);
}
