#ifndef CV_LM_CV_ENGINE_H
#define CV_LM_CV_ENGINE_H

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <utility>

#include "CV-Deterministic-OLS-Fit.h"
#include "CV-Deterministic-Ridge-Fit.h"
#include "CV-Stochastic-Worker.h"
#include "CV-Stochastic-WorkerModel.h"
#include "Enums.h"
#include "Utils-Folds.h"
#include "Utils-Parallel.h"

namespace CV {

namespace Deterministic {

// Generalized and leave-one-out cross-validation for linear and ridge
// regression
template <Enums::FitMethod Fit, Enums::AnalyticMethod Analytic,
          Enums::CenteringMethod Centering, typename... Lambda>
[[nodiscard]] double computeCV(const Eigen::Map<Eigen::VectorXd>& y,
                               const Eigen::Map<Eigen::MatrixXd>& x,
                               const double threshold, Lambda&&... lambda) {
  // Generate an OLS or Ridge Fit object for computing closed-form CV solutions
  const auto fit{[&]() {
    if constexpr (Fit == Enums::FitMethod::OLS) {
      return OLS::Fit<Analytic, Centering>{y, x, threshold};
    } else {
      Enums::assertExpected<Fit, Enums::FitMethod::Ridge>();
      return Ridge::Fit<Analytic, Centering>{y, x, threshold,
                                             std::forward<Lambda>(lambda)...};
    }
  }()};

  return fit.cv();
}

}  // namespace Deterministic

namespace Stochastic {

// Multi-threaded K-fold CV for linear and ridge regression
template <Enums::FitMethod Fit, Enums::CenteringMethod Centering,
          typename... Lambda>
[[nodiscard]] double computeCV(const Eigen::Map<Eigen::VectorXd>& y,
                               const Eigen::Map<Eigen::MatrixXd>& x,
                               const int k, const int seed, const int nThreads,
                               const double threshold, Lambda&&... lambda) {
  // Setup folds
  const Utils::Folds::DataSplitter splitter{seed, x.rows(), k};

  // Permute the design matrix and response vector so test observations are
  // stored contiguously (this generates a copy of the R data once at the
  // benefit of avoiding copies using indexed views)
  const Eigen::VectorXi perm{splitter.buildPermutation()};
  const Eigen::VectorXd ySorted{y(perm)};
  const Eigen::MatrixXd xSorted{x(perm, Eigen::all)};

  // Initialize the templated worker with pre-allocated buffers for computing
  // coefficients and decompostions depending on whether we're using OLS or
  // ridge regression
  auto worker{[&]() {
    if constexpr (Fit == Enums::FitMethod::OLS) {
      return Worker<OLS::WorkerModel, Centering>{ySorted, xSorted, splitter,
                                                 threshold};
    } else {
      Enums::assertExpected<Fit, Enums::FitMethod::Ridge>();
      return Worker<Ridge::WorkerModel, Centering>{
          ySorted, xSorted, splitter, threshold,
          std::forward<Lambda>(lambda)...};
    }
  }()};

  // Compute CV result
  Utils::Parallel::reduce(worker, static_cast<std::size_t>(k), nThreads);
  return worker.getCV();
}

}  // namespace Stochastic

}  // namespace CV

#endif  // CV_LM_CV_ENGINE_H
