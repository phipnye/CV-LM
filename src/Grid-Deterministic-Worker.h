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
template <typename WorkerPolicy>
class Worker : public RcppParallel::Worker {
  // Policy (owns all calculation-specific data)
  WorkerPolicy policy_;

  // Reduction result: (corresponding lambda, best CV)
  LambdaCV optimalPair_;

  // References
  const Generator& lambdasGrid_;

 public:
  // Main ctor
  explicit Worker(const Generator& lambdasGrid, WorkerPolicy policy)
      : policy_{std::move(policy)},
        // [lambda, CV] - no designated initializer in C++17
        optimalPair_{0.0, Constants::Inf},
        lambdasGrid_{lambdasGrid} {}
  // Split ctor
  explicit Worker(const Worker& other, const RcppParallel::Split)
      : policy_{other.policy_},
        optimalPair_{0.0, Constants::Inf},
        lambdasGrid_{other.lambdasGrid_} {}

  // parallelReduce work operator
  void operator()(const std::size_t begin, const std::size_t end) override {
    for (Eigen::Index lambdaIdx{static_cast<Eigen::Index>(begin)},
         endIdx{static_cast<Eigen::Index>(end)};
         lambdaIdx < endIdx; ++lambdaIdx) {
      // Retrieve next lambda value from the generator
      const double lambda{lambdasGrid_[lambdaIdx]};

      // Calculate GCV or LOOCV
      if (const double cv{policy_.computeCV(lambda)};
          // IEEE 754 technically evaluates < to false for NaN but added here
          // for precaution
          cv < optimalPair_.cv && !std::isnan(cv)) {
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
