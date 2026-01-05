#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <utility>

#include "CVWorker.h"
#include "utils.h"

namespace CV {

// Generalized cross-validation for linear and ridge regression
template <typename FitType, typename... Args>
double gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, Args&&... args) {
  constexpr bool needHat{false};  // GCV doesn't need full diagonal entries of
                                  // the hat matrix, just the trace
  const FitType fit{y, x, std::forward<Args>(args)..., needHat};
  return fit.gcv();
}

// Leave-one-out cross-validation for linear and ridge regression
template <typename FitType, typename... Args>
double loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
             Args&&... args) {
  constexpr bool needHat{
      true};  // LOOCV requires full diagonal entries of the hat matrix
  const FitType fit{y, x, std::forward<Args>(args)..., needHat};
  return fit.loocv();
}

// Multi-threaded CV for linear and ridge regression
template <typename WorkerFitType, typename... Args>
double parCV(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, const int k,
             const int seed, const int nThreads, Args&&... args) {
  // Setup folds
  const Eigen::Index nrow{x.rows()};
  const auto [foldIDs,
              foldSizes]{CV::Utils::cvSetup(seed, static_cast<int>(nrow), k)};

  // Pre-calculate fold size bounds (this allows us to allocate data buffers of
  // appropriate size in our worker instances)
  const auto [minTestSize, maxTestSize]{CV::Utils::testSizeExtrema(foldSizes)};
  const Eigen::Index maxTrainSize{nrow - minTestSize};

  // Initialize the templated worker with model-specific arguments (like lambda)
  CVWorker<WorkerFitType> worker{y,
                                 x,
                                 foldIDs,
                                 foldSizes,
                                 maxTrainSize,
                                 maxTestSize,
                                 std::forward<Args>(args)...};

  // Compute CV result
  constexpr std::size_t begin{0};
  const std::size_t end{static_cast<std::size_t>(k)};
  constexpr std::size_t grainSize{1};
  RcppParallel::parallelReduce(begin, end, worker, grainSize, nThreads);
  return worker.mse_;
}

}  // namespace CV
