#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <limits>
#include <utility>

#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"

namespace Grid::Deterministic {

// Class for searching grid of deterministic CV (LOOCV and GCV) results
template <typename WorkerPolicy>
class Worker : public RcppParallel::Worker {
  // Policy (owns all calculation-specific data)
  WorkerPolicy policy_;

  // Thread-local buffer for repeated denominator computations
  Eigen::ArrayXd denom_;

  // Reduction result: (corresponding lambda, best CV)
  LambdaCV optimalPair_;

  // References
  const Generator& lambdasGrid_;
  const Eigen::ArrayXd& eigenValsSq_;

  // Scalars
  const Eigen::Index nrow_;

  // Flag for whether data was centered in R
  const bool centered_;

 public:
  // Main ctor
  explicit Worker(const Generator& lambdasGrid,
                  const Eigen::ArrayXd& eigenValsSq, const Eigen::Index nrow,
                  const bool centered, WorkerPolicy policy)
      : policy_{std::move(policy)},
        denom_(eigenValsSq.size()),
        // [lambda, CV] - no designated initializer in C++17
        optimalPair_{0.0, std::numeric_limits<double>::infinity()},
        lambdasGrid_{lambdasGrid},
        eigenValsSq_{eigenValsSq},
        nrow_{nrow},
        centered_{centered} {}

  // Split ctor
  explicit Worker(const Worker& other, const RcppParallel::Split)
      : policy_{other.policy_},
        denom_(other.denom_.size()),
        optimalPair_{0.0, std::numeric_limits<double>::infinity()},
        lambdasGrid_{other.lambdasGrid_},
        eigenValsSq_{other.eigenValsSq_},
        nrow_{other.nrow_},
        centered_{other.centered_} {}

  // parallelReduce work operator
  void operator()(const std::size_t begin, const std::size_t end) override {
    for (Eigen::Index lambdaIdx{static_cast<Eigen::Index>(begin)},
         endIdx{static_cast<Eigen::Index>(end)};
         lambdaIdx < endIdx; ++lambdaIdx) {
      // Retrieve next lambda value from the generator
      const double lambda{lambdasGrid_[lambdaIdx]};

      // denom_ is reused to avoid temporary allocations and repeated
      // computations (it is the denominator of df(lambda) in ESL p. 68)
      denom_ = eigenValsSq_ + lambda;

      // Calculate GCV or LOOCV
      const double cv{
          policy_.computeCV(lambda, denom_, eigenValsSq_, nrow_, centered_)};

      if (cv < optimalPair_.cv) {
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
