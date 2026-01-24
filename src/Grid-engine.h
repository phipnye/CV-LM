#ifndef CV_LM_GRID_ENGINE_H
#define CV_LM_GRID_ENGINE_H

#include <RcppEigen.h>

#include "Enums.h"
#include "Grid-Deterministic-Worker.h"
#include "Grid-Generator.h"
#include "Grid-Internal.h"
#include "Grid-LambdaCV.h"
#include "Grid-Stochastic-Worker.h"
#include "ResponseWrapper.h"
#include "Utils-Decompositions.h"
#include "Utils-Folds.h"
#include "Utils-Parallel.h"

namespace Grid {

namespace Deterministic {

// Generalized and leave-one-out grid search
template <Enums::AnalyticMethod Analytic, Enums::CenteringMethod Centering>
[[nodiscard]] LambdaCV search(const Eigen::Map<Eigen::VectorXd>& yOrig,
                              const Eigen::Map<Eigen::MatrixXd>& xOrig,
                              const Generator& lambdasGrid, const int nThreads,
                              const double threshold) {
  // Decompose X = UDV' with thin U computation
  const Eigen::BDCSVD<Eigen::MatrixXd> udvT{
      Utils::Decompositions::svd<Centering>(xOrig, Eigen::ComputeThinU,
                                            threshold)};

  // Center the response data if necessary
  ResponseWrapper<Centering> y{yOrig};

  // Pre-compute shared values across both deterministic methods
  const Internal::SVDPrecompute pre{
      // squared (corrected) singular values
      Utils::Decompositions::getSingularVals(udvT).array().square(),
      // U
      udvT.matrixU(),
      // nrow
      xOrig.rows()};

  // Pre-compute data for closed-form solutions, initialize a worker instance,
  // and execute the grid search in parallel across the lambda values
  if constexpr (Analytic == Enums::AnalyticMethod::GCV) {
    return Internal::runGCV(lambdasGrid, nThreads, y, pre);
  } else {
    Enums::assertExpected<Analytic, Enums::AnalyticMethod::LOOCV>();
    return Internal::runLOOCV(lambdasGrid, nThreads, y, pre);
  }
}

}  // namespace Deterministic

namespace Stochastic {

// K-fold grid search
template <Enums::CenteringMethod Centering>
[[nodiscard]] LambdaCV search(const Eigen::Map<Eigen::VectorXd>& y,
                              const Eigen::Map<Eigen::MatrixXd>& x, const int k,
                              const Generator& lambdasGrid, const int seed,
                              const int nThreads, const double threshold) {
  // Setup folds
  const Utils::Folds::DataSplitter splitter{seed, x.rows(), k};

  // Permute the design matrix and response vector so test observations are
  // stored contiguously (this generates a copy of the R data once at the
  // benefit of copying in the worker using indexed views)
  const Eigen::VectorXi perm{splitter.buildPermutation()};
  const Eigen::VectorXd ySorted{y(perm)};
  const Eigen::MatrixXd xSorted{x(perm, Eigen::all)};

  // Initialize Worker
  Worker<Centering> worker{ySorted, xSorted, splitter, lambdasGrid, threshold};

  // Compute cross-validation results (parallelize over folds)
  Utils::Parallel::reduce(worker, static_cast<std::size_t>(k), nThreads);
  return worker.getOptimalPair();
}

}  // namespace Stochastic

}  // namespace Grid

#endif  // CV_LM_GRID_ENGINE_H
