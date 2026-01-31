#include <RcppArmadillo.h>

#include <limits>
#include <string>

#include "CV.h"
#include "Enums.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid.h"
#include "Utils-Folds.h"

// Enforce IEEE 754 / IEC 559 compliance ("All R platforms are required to work
// with values conforming to the IEC 60559 (also known as IEEE 754) standard" -
// https://stat.ethz.ch/R-manual/R-devel/library/base/html/double.html)
static_assert(
    std::numeric_limits<double>::is_iec559,
    "Full IEC 60559 (IEEE 754) compliance is required for this R package.");

// [[Rcpp::export(name="cv.lm.rcpp")]]
double cvLMRCpp(const arma::mat& X, const arma::vec& y, const arma::uword k0,
                const double lambda, const bool generalized, const int seed,
                const int nThreads, const double tolerance, const bool center) {
  // Determine a valid number of folds as close to the passed K argument as
  // possible
  const arma::uword k{Utils::Folds::kCheck(X.n_rows, k0, generalized)};

  // --- Function dispatch
  using namespace Enums;

  // --- GCV or LOOCV (closed-form solutions)
  if (generalized || k == X.n_rows) {
    using CV::Deterministic::computeCV;

    // GCV
    if (generalized) {
      return center
                 ? computeCV<CrossValidationMethod::GCV, CenteringMethod::Mean>(
                       X, y, tolerance, lambda)
                 : computeCV<CrossValidationMethod::GCV, CenteringMethod::None>(
                       X, y, tolerance, lambda);
    }

    // LOOCV
    return center
               ? computeCV<CrossValidationMethod::LOOCV, CenteringMethod::Mean>(
                     X, y, tolerance, lambda)
               : computeCV<CrossValidationMethod::LOOCV, CenteringMethod::None>(
                     X, y, tolerance, lambda);
  }

  // --- K-fold CV
  using CV::Stochastic::computeCV;
  return center ? computeCV<CenteringMethod::Mean>(X, y, k, seed, nThreads,
                                                   tolerance, lambda)
                : computeCV<CenteringMethod::None>(X, y, k, seed, nThreads,
                                                   tolerance, lambda);
}

// [[Rcpp::export(name="grid.search.rcpp")]]
Rcpp::List gridSearch(const arma::mat& X, const arma::vec& y, const int k0,
                      const double maxLambda, const double precision,
                      const bool generalized, const int seed,
                      const int nThreads, const double tolerance,
                      const bool center) {
  // Determine a valid number of folds as close to the passed K argument as
  // possible
  const arma::uword k{Utils::Folds::kCheck(X.n_rows, k0, generalized)};

  // Lightweight generator for creating lambda values
  const Grid::Generator lambdasGrid{maxLambda, precision};

  // Optimal CV results in the form [CV, lambda]
  Grid::LambdaCV optimalPair{0.0, 0.0};

  // --- Function dispatch
  using namespace Enums;

  if (generalized || k == X.n_rows) {
    // Closed-form cv solutions
    using Grid::Deterministic::search;

    // GCV
    if (generalized) {
      optimalPair =
          center ? search<CrossValidationMethod::GCV, CenteringMethod::Mean>(
                       X, y, lambdasGrid, nThreads, tolerance)
                 : search<CrossValidationMethod::GCV, CenteringMethod::None>(
                       X, y, lambdasGrid, nThreads, tolerance);
    } else {  // LOOCV
      optimalPair =
          center ? search<CrossValidationMethod::LOOCV, CenteringMethod::Mean>(
                       X, y, lambdasGrid, nThreads, tolerance)
                 : search<CrossValidationMethod::LOOCV, CenteringMethod::None>(
                       X, y, lambdasGrid, nThreads, tolerance);
    }
  } else {
    // K-fold CV
    using Grid::Stochastic::search;
    optimalPair = center ? search<CenteringMethod::Mean>(
                               X, y, k, lambdasGrid, seed, nThreads, tolerance)
                         : search<CenteringMethod::None>(
                               X, y, k, lambdasGrid, seed, nThreads, tolerance);
  }

  return Rcpp::List::create(Rcpp::Named("CV") = optimalPair.cv_,
                            Rcpp::Named("lambda") = optimalPair.lambda_);
}
