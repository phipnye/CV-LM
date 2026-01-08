#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <type_traits>
#include <utility>

#include "CV-Utils-utils.h"
#include "CV-Worker.h"
#include "CV-WorkerModel.h"
#include "CV-WorkerModelFactory.h"

namespace CV {

// Generalized cross-validation for linear and ridge regression
template <typename FitType, typename... Args>
[[nodiscard]] double gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                         Args&&... args) {
  constexpr bool needHat{false};  // GCV doesn't need full diagonal entries of
                                  // the hat matrix, just the trace
  const FitType fit{y, x, std::forward<Args>(args)..., needHat};
  return fit.gcv();
}

// Leave-one-out cross-validation for linear and ridge regression
template <typename FitType, typename... Args>
[[nodiscard]] double loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                           Args&&... args) {
  constexpr bool needHat{
      true};  // LOOCV requires full diagonal entries of the hat matrix
  const FitType fit{y, x, std::forward<Args>(args)..., needHat};
  return fit.loocv();
}

// Multi-threaded CV for linear and ridge regression
template <typename WorkerModelType, typename... Args>
[[nodiscard]] double parCV(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                           const int k, const int seed, const int nThreads,
                           Args&&... args) {
  // Setup folds
  const Eigen::Index nrow{x.rows()};
  const auto [foldIDs,
              foldSizes]{Utils::setupFolds(seed, static_cast<int>(nrow), k)};

  // Pre-calculate fold size bounds (this allows us to allocate data buffers of
  // appropriate size in our worker instances)
  const auto [minTestSize, maxTestSize]{Utils::testSizeExtrema(foldSizes)};
  const Eigen::Index maxTrainSize{nrow - minTestSize};

  // Generate a WorkerModelFactory for generating worker models with
  // pre-allocated memory
  const auto factory{[&]() {
    if constexpr (std::is_same_v<WorkerModelType, OLS::WorkerModel>) {
      // OLS model requires the training size for pre-allocation of QR and a
      // threshold at which to consider singular values zero
      return OLS::WorkerModelFactory{x.cols(), maxTrainSize,
                                     std::forward<Args>(args)...};
    } else if constexpr (std::is_same_v<WorkerModelType, Ridge::WorkerModel>) {
      // Ridge model doesn't need train size since we're using cholesky of
      // regularized covariance matrix but does need lambda
      return Ridge::WorkerModelFactory{x.cols(), std::forward<Args>(args)...};
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
      y, x, foldIDs, foldSizes, maxTrainSize, maxTestSize, factory};

  // Compute CV result
  constexpr std::size_t begin{0};
  const std::size_t end{static_cast<std::size_t>(k)};
  constexpr std::size_t grainSize{1};
  RcppParallel::parallelReduce(begin, end, worker, grainSize, nThreads);
  return worker.mse_;
}

}  // namespace CV
