#include <Rcpp.h>
#include <RcppEigen.h>

#include "CV-engine.h"
#include "Enums.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-engine.h"
#include "Utils-Folds.h"

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
                const int nThreads, const double threshold, const bool center) {
  // Determine which type of model we're fitting (this has potentially important
  // implications since OLS uses complete orthogonal decomposition whereas ridge
  // regression uses singular value decomposition (we also safe-guard against
  // negative values which should be handled in R but is added here as a
  // precaution)
  const bool useOLS{lambda <= 0.0};

  // https://cran.r-project.org/doc/manuals/r-release/R-ints.html
  // "Matrices are stored as vectors and so were also limited to 2^31-1
  // elements. Now longer vectors are allowed on 64-bit platforms, matrices with
  // more elements are supported provided that each of the dimensions is no more
  // than 2^31-1."
  const int nrow{static_cast<int>(x.rows())};

  // Determine a valid number of folds as close to the passed K argument as
  // possible
  const int k{Utils::Folds::kCheck(nrow, k0, generalized)};

  // --- Function dispatch
  using namespace Enums;

  // GCV or LOOCV (closed-form solutions)
  if (generalized || k == nrow) {
    using CV::Deterministic::computeCV;

    // OLS
    if (useOLS) {
      if (generalized) {
        return center ? computeCV<FitMethod::OLS, AnalyticMethod::GCV,
                                  CenteringMethod::Mean>(y, x, threshold)
                      : computeCV<FitMethod::OLS, AnalyticMethod::GCV,
                                  CenteringMethod::None>(y, x, threshold);
      }

      return center ? computeCV<FitMethod::OLS, AnalyticMethod::LOOCV,
                                CenteringMethod::Mean>(y, x, threshold)
                    : computeCV<FitMethod::OLS, AnalyticMethod::LOOCV,
                                CenteringMethod::None>(y, x, threshold);
    }

    // Ridge regression GCV
    if (generalized) {
      return center ? computeCV<FitMethod::Ridge, AnalyticMethod::GCV,
                                CenteringMethod::Mean>(y, x, threshold, lambda)
                    : computeCV<FitMethod::Ridge, AnalyticMethod::GCV,
                                CenteringMethod::None>(y, x, threshold, lambda);
    }

    // Ridge regression LOOCV
    return center ? computeCV<FitMethod::Ridge, AnalyticMethod::LOOCV,
                              CenteringMethod::Mean>(y, x, threshold, lambda)
                  : computeCV<FitMethod::Ridge, AnalyticMethod::LOOCV,
                              CenteringMethod::None>(y, x, threshold, lambda);
  }

  // K-fold CV
  using CV::Stochastic::computeCV;

  // OLS
  if (useOLS) {
    return center ? computeCV<FitMethod::OLS, CenteringMethod::Mean>(
                        y, x, k, seed, nThreads, threshold)
                  : computeCV<FitMethod::OLS, CenteringMethod::None>(
                        y, x, k, seed, nThreads, threshold);
  }

  // Ridge regression
  return center ? computeCV<FitMethod::Ridge, CenteringMethod::Mean>(
                      y, x, k, seed, nThreads, threshold, lambda)
                : computeCV<FitMethod::Ridge, CenteringMethod::None>(
                      y, x, k, seed, nThreads, threshold, lambda);
}

// [[Rcpp::export(name="grid.search.rcpp")]]
Rcpp::List gridSearch(const Eigen::Map<Eigen::VectorXd> y,
                      const Eigen::Map<Eigen::MatrixXd> x, const int k0,
                      const double maxLambda, const double precision,
                      const bool generalized, const int seed,
                      const int nThreads, const double threshold,
                      const bool center) {
  // Determine a valid number of folds as close to the passed K argument as
  // possible
  const int nrow{static_cast<int>(x.rows())};
  const int k{Utils::Folds::kCheck(nrow, k0, generalized)};

  // Lightweight generator for creating lambda values
  const Grid::Generator lambdasGrid{maxLambda, precision};

  // Optimal CV results in the form [CV, lambda]
  Grid::LambdaCV optimalPair{0.0, 0.0};

  // --- Function dispatch
  using namespace Enums;

  if (generalized || k == nrow) {
    // Closed-form cv solutions
    using Grid::Deterministic::search;

    // GCV
    if (generalized) {
      optimalPair = center ? search<AnalyticMethod::GCV, CenteringMethod::Mean>(
                                 y, x, lambdasGrid, nThreads, threshold)
                           : search<AnalyticMethod::GCV, CenteringMethod::None>(
                                 y, x, lambdasGrid, nThreads, threshold);
    } else {  // LOOCV
      optimalPair = center
                        ? search<AnalyticMethod::LOOCV, CenteringMethod::Mean>(
                              y, x, lambdasGrid, nThreads, threshold)
                        : search<AnalyticMethod::LOOCV, CenteringMethod::None>(
                              y, x, lambdasGrid, nThreads, threshold);
    }
  } else {
    // K-fold CV
    using Grid::Stochastic::search;
    optimalPair = center ? search<CenteringMethod::Mean>(
                               y, x, k, lambdasGrid, seed, nThreads, threshold)
                         : search<CenteringMethod::None>(
                               y, x, k, lambdasGrid, seed, nThreads, threshold);
  }

  return Rcpp::List::create(Rcpp::Named("CV") = optimalPair.cv,
                            Rcpp::Named("lambda") = optimalPair.lambda);
}
