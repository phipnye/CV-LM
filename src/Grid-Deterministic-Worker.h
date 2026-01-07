#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <limits>
#include <utility>

#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"

namespace Grid::Deterministic {

// Base class for searching grid of deterministic CV (LOOCV and GCV) results
template <typename WorkerPolicy>
struct Worker : public RcppParallel::Worker {
  // References
  const Generator& lambdasGrid_;
  const Eigen::ArrayXd& eigenValsSq_;

  // Sizes
  const Eigen::Index nrow_;

  // Boolean flags
  const bool centered_;

  // Thread-local buffer for repeated denominator computations
  Eigen::ArrayXd denom_;

  // Reduction result: (corresponding lambda, best CV)
  LambdaCV results_;

  // Policy (owns all calculation-specific data)
  WorkerPolicy policy_;

  // Main ctor
  explicit Worker(const Generator& lambdasGrid,
                  const Eigen::ArrayXd& eigenValsSq, const Eigen::Index nrow,
                  const bool centered, WorkerPolicy policy)
      : lambdasGrid_{lambdasGrid},
        eigenValsSq_{eigenValsSq},
        nrow_{nrow},
        centered_{centered},
        denom_(eigenValsSq.size()),
        // [lambda, CV]
        results_{0.0, std::numeric_limits<double>::infinity()},
        policy_{std::move(policy)} {}

  // Split ctor
  explicit Worker(const Worker& other, const RcppParallel::Split)
      : lambdasGrid_{other.lambdasGrid_},
        eigenValsSq_{other.eigenValsSq_},
        nrow_{other.nrow_},
        centered_{other.centered_},
        denom_(other.denom_.size()),
        // [lambda, CV]
        results_{0.0, std::numeric_limits<double>::infinity()},
        policy_{other.policy_} {}

  // parallelReduce work operator
  void operator()(const std::size_t begin, const std::size_t end) {
    for (Eigen::Index lambdaIdx{static_cast<Eigen::Index>(begin)},
         endIdx{static_cast<Eigen::Index>(end)};
         lambdaIdx < endIdx; ++lambdaIdx) {
      const double lambda{lambdasGrid_[lambdaIdx]};

      // denom_ is reused to avoid temporary allocations and repeated
      // computations
      denom_ = eigenValsSq_ + lambda;

      // Calculate GCV or LOOCV
      const double cv{
          policy_.evaluate(lambda, denom_, eigenValsSq_, nrow_, centered_)};

      if (cv < results_.cv) {
        results_.cv = cv;
        results_.lambda = lambda;
      }
    }
  }

  // Join logic for parallel reduction
  void join(const Worker& other) {
    if (other.results_.cv < results_.cv) {
      results_ = other.results_;
    }
  }
};

};  // namespace Grid::Deterministic
