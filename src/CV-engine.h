#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <type_traits>
#include <utility>

// ReSharper disable once CppUnusedIncludeDirective
#include "CV-OLS-Fit.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "CV-Ridge-Fit.h"
#include "CV-Utils-utils.h"
#include "CV-Worker.h"
#include "CV-WorkerModel.h"
#include "CV-WorkerModelFactory.h"
#include "Constants.h"

namespace CV {

// Generalized cross-validation for linear and ridge regression
template <template <bool> typename FitType, typename... Args>
[[nodiscard]] double gcv(const Eigen::Map<Eigen::VectorXd>& y,
                         const Eigen::Map<Eigen::MatrixXd>& x, Args&&... args) {
  constexpr bool NeedHat{false};  // GCV doesn't need full diagonal entries of
                                  // the hat matrix, just the trace
  const FitType<NeedHat> fit{y, x, std::forward<Args>(args)...};
  return fit.gcv();
}

// Leave-one-out cross-validation for linear and ridge regression
template <template <bool> typename FitType, typename... Args>
[[nodiscard]] double loocv(const Eigen::Map<Eigen::VectorXd>& y,
                           const Eigen::Map<Eigen::MatrixXd>& x,
                           Args&&... args) {
  constexpr bool NeedHat{
      true};  // LOOCV requires full diagonal entries of the hat matrix
  const FitType<NeedHat> fit{y, x, std::forward<Args>(args)...};
  return fit.loocv();
}

// Multi-threaded CV for linear and ridge regression
template <typename WorkerModelType, typename... Args>
[[nodiscard]] double parCV(const Eigen::Map<Eigen::VectorXd>& y,
                           const Eigen::Map<Eigen::MatrixXd>& x, const int k,
                           const int seed, const int nThreads, Args&&... args) {
  // Setup folds
  const Eigen::Index nrow{x.rows()};
  const auto [testFoldIDs, testFoldSizes]{
      Utils::setupFolds(seed, static_cast<int>(nrow), k)};

  // Pre-calculate fold size bounds (this allows us to allocate data buffers of
  // appropriate size in our worker instances)
  const auto [minTestSize, maxTestSize]{Utils::testSizeExtrema(testFoldSizes)};
  const Eigen::Index maxTrainSize{nrow - minTestSize};

  // Generate a WorkerModelFactory for generating worker models with
  // pre-allocated memory
  const auto factory{[&]() {
    if constexpr (std::is_same_v<WorkerModelType, OLS::WorkerModel>) {
      // OLS model requires the training size for pre-allocation of QR/COD and a
      // threshold at which to consider singular values zero
      return OLS::WorkerModelFactory{x.cols(), maxTrainSize,
                                     std::forward<Args>(args)...};
    } else if constexpr (std::is_same_v<WorkerModelType,
                                        Ridge::Narrow::WorkerModel>) {
      // Narrow ridge model doesn't need train size since we're using cholesky
      // of regularized covariance matrix
      return Ridge::Narrow::WorkerModelFactory{x.cols(),
                                               std::forward<Args>(args)...};
    } else if constexpr (std::is_same_v<WorkerModelType,
                                        Ridge::Wide::WorkerModel>) {
      // Wide ridge model requires train size and lambda since we're using
      // cholesky of regularized gram matrix
      return Ridge::Wide::WorkerModelFactory{maxTrainSize,
                                             std::forward<Args>(args)...};
    } else {
      // Raise a build error if we pass some unexpected configuration
      static_assert(!std::is_same_v<WorkerModelType, WorkerModelType>,
                    "Unsupported model type");
    }
  }()};

  // Initialize the templated worker with a factory object to construct a worker
  // model with pre-allocated memory for the decomposition and coefficient
  // computation
  Worker<WorkerModelType, decltype(factory)> worker{
      y, x, testFoldIDs, testFoldSizes, maxTrainSize, maxTestSize, factory};

  // Compute CV result
  const std::size_t end{static_cast<std::size_t>(k)};
  RcppParallel::parallelReduce(Constants::begin, end, worker,
                               Constants::grainSize, nThreads);
  return worker.mse_;
}

}  // namespace CV
