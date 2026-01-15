#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <utility>

#include "CV-OLS-Fit.h"
#include "CV-Ridge-Fit.h"
#include "CV-Worker.h"
#include "CV-WorkerModel.h"
#include "Enums-enums.h"
#include "Utils-Folds-utils.h"
#include "Utils-Parallel-utils.h"

namespace CV {

namespace Deterministic {

// Generalized and leave-one-out cross-validation for linear and ridge
// regression
template <Enums::FitMethod FitType, Enums::AnalyticMethod CVMethod,
          typename... RidgeArgs>
[[nodiscard]] double computeCV(const Eigen::Map<Eigen::VectorXd>& y,
                               const Eigen::Map<Eigen::MatrixXd>& x,
                               const double threshold,
                               RidgeArgs&&... ridgeArgs) {
  // Generate an OLS or Ridge Fit object for computing closed-form CV solutions
  const auto fit{[&]() {
    if constexpr (FitType == Enums::FitMethod::OLS) {
      return OLS::Fit<CVMethod>{y, x, threshold};
    } else {
      Enums::assertExpected<FitType, Enums::FitMethod::Ridge>();
      return Ridge::Fit<CVMethod>{y, x, threshold,
                                  std::forward<RidgeArgs>(ridgeArgs)...};
    }
  }()};

  return fit.cv();
}

}  // namespace Deterministic

namespace Stochastic {

// Multi-threaded K-fold CV for linear and ridge regression
template <Enums::FitMethod FitType, typename... Lambda>
[[nodiscard]] double computeCV(const Eigen::Map<Eigen::VectorXd>& y,
                               const Eigen::Map<Eigen::MatrixXd>& x,
                               const int k, const int seed, const int nThreads,
                               const double threshold, Lambda&&... lambda) {
  // Setup folds
  const Eigen::Index nrow{x.rows()};
  const Utils::Folds::FoldInfo foldInfo{seed, static_cast<int>(nrow), k};

  // Initialize the templated worker with pre-allocated buffers for computing
  // coefficients and decompostions depending on whether we're using OLS or
  // ridge regression
  auto worker{[&]() {
    if constexpr (FitType == Enums::FitMethod::OLS) {
      return Worker<OLS::WorkerModel>{y, x, foldInfo, threshold};
    } else {
      Enums::assertExpected<FitType, Enums::FitMethod::Ridge>();
      return Worker<Ridge::WorkerModel>{y, x, foldInfo, threshold,
                                        std::forward<Lambda>(lambda)...};
    }
  }()};

  // Compute CV result
  Utils::Parallel::reduce(worker, static_cast<std::size_t>(k), nThreads);
  return worker.getCV();
}

}  // namespace Stochastic

}  // namespace CV
