#include <Rcpp.h>
#include <RcppEigen.h>

#include "CV-engine.h"
#include "Enums-enums.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-engine.h"
#include "Utils-Folds-utils.h"

// Enforce IEEE 754 / IEC 559 compliance ("All R platforms are required to work
// with values conforming to the IEC 60559 (also known as IEEE 754) standard" -
// https://stat.ethz.ch/R-manual/R-devel/library/base/html/double.html)
static_assert(
    std::numeric_limits<double>::is_iec559,
    "Full IEC 60559 (IEEE 754) compliance is required for this R package.");

// [[Rcpp::export(name="cv.lm.rcpp")]]
double cvLMRCpp(const Eigen::Map<Eigen::VectorXd> y,
                const Eigen::Map<Eigen::MatrixXd> x, const int k0,
                const double lambda, const bool generalized, const int seed,
                const int nThreads, const double threshold,
                const bool centered) {
  // Determine which type of model we're fitting (this has potentially important
  // implications since OLS uses complete orthogonal decomposition whereas ridge
  // regression uses singular value decomposition (we also safe-guard against
  // negative values which should be handled in R but is added here as a
  // precaution)
  const bool useOLS{lambda <= 0.0};

  // Skip checking number of folds if we're using generalized CV
  if (generalized) {
    using CV::Deterministic::computeCV;
    return useOLS
               ? computeCV<Enums::FitMethod::OLS, Enums::AnalyticMethod::GCV>(
                     y, x, threshold)
               : computeCV<Enums::FitMethod::Ridge, Enums::AnalyticMethod::GCV>(
                     y, x, threshold, lambda, centered);
  }

  // https://cran.r-project.org/doc/manuals/r-release/R-ints.html
  // "Matrices are stored as vectors and so were also limited to 2^31-1
  // elements. Now longer vectors are allowed on 64-bit platforms, matrices with
  // more elements are supported provided that each of the dimensions is no more
  // than 2^31-1."
  const int nrow{static_cast<int>(x.rows())};

  // Determine a valid number of folds as close to the passed K argument as
  // possible
  const int k{Utils::Folds::kCheck(nrow, k0)};

  // LOOCV
  if (k == nrow) {
    using CV::Deterministic::computeCV;
    return useOLS
               ? computeCV<Enums::FitMethod::OLS, Enums::AnalyticMethod::LOOCV>(
                     y, x, threshold)
               : computeCV<Enums::FitMethod::Ridge,
                           Enums::AnalyticMethod::LOOCV>(y, x, threshold,
                                                         lambda, centered);
  }

  // K-fold CV
  using CV::Stochastic::computeCV;
  return useOLS ? computeCV<Enums::FitMethod::OLS>(y, x, k, seed, nThreads,
                                                   threshold)
                : computeCV<Enums::FitMethod::Ridge>(y, x, k, seed, nThreads,
                                                     threshold, lambda);
}

// [[Rcpp::export(name="grid.search.rcpp")]]
Rcpp::List gridSearch(const Eigen::Map<Eigen::VectorXd> y,
                      const Eigen::Map<Eigen::MatrixXd> x, const int k0,
                      const double maxLambda, const double precision,
                      const bool generalized, const int seed,
                      const int nThreads, const double threshold,
                      const bool centered) {
  // Lightweight generator for creating lambda values
  const Grid::Generator lambdasGrid{maxLambda, precision};

  // Optimal CV results in the form [CV, lambda]
  Grid::LambdaCV optimalPair{};

  // Generalized CV
  if (generalized) {
    optimalPair = Grid::Deterministic::search<Enums::AnalyticMethod::GCV>(
        y, x, lambdasGrid, nThreads, threshold, centered);
  } else {
    // Determine a valid number of folds as close to the passed K as possible
    const int nrow{static_cast<int>(x.rows())};

    // Leave-one-out CV
    if (const int k{Utils::Folds::kCheck(nrow, k0)}; k == nrow) {
      optimalPair = Grid::Deterministic::search<Enums::AnalyticMethod::LOOCV>(
          y, x, lambdasGrid, nThreads, threshold, centered);
    } else {
      // K-fold CV
      optimalPair = Grid::Stochastic::search(y, x, k, lambdasGrid, seed,
                                             nThreads, threshold);
    }
  }

  return Rcpp::List::create(Rcpp::Named("CV") = optimalPair.cv,
                            Rcpp::Named("lambda") = optimalPair.lambda);
}
