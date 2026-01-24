#ifndef CV_LM_GRID_DETERMINISTIC_WORKER_H
#define CV_LM_GRID_DETERMINISTIC_WORKER_H

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

  // Work operator for parallel reduction - each thread gets its own exclusive
  // range
  void operator()(const std::size_t begin, const std::size_t end) override {
    // This type cast is safe, the grid ctor ensures the size (end) doesn't
    // exceed Index limit
    const Eigen::Index endIdx{static_cast<Eigen::Index>(end)};

    for (Eigen::Index lambdaIdx{static_cast<Eigen::Index>(begin)};
         lambdaIdx < endIdx; ++lambdaIdx) {
      // Retrieve next lambda value from the generator
      const double lambda{lambdasGrid_[lambdaIdx]};

      // Calculate GCV or LOOCV (IEEE 754 evaluates < to false for NaN)
      if (const double cv{model_.computeCV(lambda)}; cv < optimalPair_.cv_) {
        optimalPair_.cv_ = cv;
        optimalPair_.lambda_ = lambda;
      }
    }
  }

  // Reduce results across multiple threads
  void join(const Worker& other) {
    // In the off chance of multiple min CVs, the smaller lambda should be taken
    // because LHS (this) is the earlier sequence compared to RHS (other)
    if (other.optimalPair_.cv_ < optimalPair_.cv_) {
      optimalPair_ = other.optimalPair_;
    }
  }

  // Retrieve optimal result
  [[nodiscard]] LambdaCV getOptimalPair() const noexcept {
    return optimalPair_;
  }
};

}  // namespace Grid::Deterministic

#endif  // CV_LM_GRID_DETERMINISTIC_WORKER_H
