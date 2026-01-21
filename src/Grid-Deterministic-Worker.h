#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cmath>
#include <cstddef>
#include <utility>

#include "Constants.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"

namespace Grid::Deterministic {

// Class for searching grid of deterministic CV (LOOCV and GCV) results
template <typename WorkerModel>
class Worker : public RcppParallel::Worker {
  // Model (owns all calculation-specific data)
  WorkerModel model_;

  // Reduction result: (corresponding lambda, best CV)
  LambdaCV optimalPair_;

  // References
  const Generator& lambdasGrid_;

 public:
  // Main ctor
  explicit Worker(const Generator& lambdasGrid, WorkerModel model)
      : model_{std::move(model)},
        // [lambda, CV] - no designated initializer in C++17
        optimalPair_{0.0, Constants::Inf},
        lambdasGrid_{lambdasGrid} {}

  // Split ctor
  explicit Worker(const Worker& other, const RcppParallel::Split)
      : model_{other.model_},
        optimalPair_{0.0, Constants::Inf},
        lambdasGrid_{other.lambdasGrid_} {}

  // parallelReduce work operator
  void operator()(const std::size_t begin, const std::size_t end) override {
    const Eigen::Index endIdx{static_cast<Eigen::Index>(end)};

    for (Eigen::Index lambdaIdx{static_cast<Eigen::Index>(begin)};
         lambdaIdx < endIdx; ++lambdaIdx) {
      // Retrieve next lambda value from the generator
      const double lambda{lambdasGrid_[lambdaIdx]};

      // Calculate GCV or LOOCV (IEEE 754 evaluates < to false for NaN)
      if (const double cv{model_.computeCV(lambda)}; cv < optimalPair_.cv) {
        optimalPair_.cv = cv;
        optimalPair_.lambda = lambda;
      }
    }
  }

  // Join logic for parallel reduction
  void join(const Worker& other) {
    if (other.optimalPair_.cv < optimalPair_.cv) {
      optimalPair_ = other.optimalPair_;
    }
  }

  // Member access
  [[nodiscard]] LambdaCV getOptimalPair() const noexcept {
    return optimalPair_;
  }
};

}  // namespace Grid::Deterministic
