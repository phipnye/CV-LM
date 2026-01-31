#ifndef CV_LM_GRID_DETERMINISTIC_WORKER_H
#define CV_LM_GRID_DETERMINISTIC_WORKER_H

#include <RcppArmadillo.h>
#include <RcppParallel.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"

namespace Grid::Deterministic {

// Class for searching grid of deterministic CV (LOOCV and GCV) results
template <typename Decomp>
class Worker : public RcppParallel::Worker {
  // --- Data members

  // Decomposition object for doing the math
  Decomp decomp_;

  // Reduction result: (corresponding lambda, best CV)
  LambdaCV optimalPair_;

  // References
  const Generator& lambdasGrid_;

 public:
  // Main ctor
  explicit Worker(Decomp decomp, const Generator& lambdasGrid)
      : decomp_{std::move(decomp)},
        // [lambda, CV] - no designated initializer in C++17
        optimalPair_{0.0, std::numeric_limits<double>::infinity()},
        lambdasGrid_{lambdasGrid} {}

  // Split ctor
  explicit Worker(const Worker& other, const RcppParallel::Split)
      : decomp_{other.decomp_},
        optimalPair_{0.0, std::numeric_limits<double>::infinity()},
        lambdasGrid_{other.lambdasGrid_} {}

  // Work operator for parallel reduction - each thread gets its own exclusive
  // range
  void operator()(const std::size_t gridBegin,
                  const std::size_t gridEnd) override {
    // This type cast is safe, the grid ctor ensures the size (end) doesn't
    // exceed uword limit
    const arma::uword endIdx{static_cast<arma::uword>(gridEnd)};

    for (arma::uword lambdaIdx{static_cast<arma::uword>(gridBegin)};
         lambdaIdx < endIdx; ++lambdaIdx) {
      // Retrieve next lambda value from the generator
      const double lambda{lambdasGrid_[lambdaIdx]};
      decomp_.setLambda(lambda);

      // Calculate GCV or LOOCV (IEEE 754 evaluates < to false for NaN)
      if (const double cv{decomp_.cv()}; cv < optimalPair_.cv_) {
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
