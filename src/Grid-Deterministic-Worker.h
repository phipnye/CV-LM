#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <cstddef>
#include <limits>
#include <utility>

namespace Grid::Deterministic {

// Base class for searching grid of deterministic CV (LOOCV and GCV) results
template <typename WorkerPolicy>
struct Worker : public RcppParallel::Worker {
  // References
  const Eigen::VectorXd& lambdas_;
  const Eigen::ArrayXd& eigenValsSq_;

  // Sizes
  const Eigen::Index nrow_;

  // Boolean flags
  const bool centered_;

  // Thread-local buffer for repeated denominator computations
  Eigen::ArrayXd denom_;

  // Reduction result: (best CV, best lambda)
  std::pair<double, double> results_;

  // Policy (owns all calculation-specific data)
  WorkerPolicy policy_;

  // Main ctor
  explicit Worker(const Eigen::VectorXd& lambdas,
                  const Eigen::ArrayXd& eigenValsSq, const Eigen::Index nrow,
                  const bool centered, WorkerPolicy policy)
      : lambdas_{lambdas},
        eigenValsSq_{eigenValsSq},
        nrow_{nrow},
        centered_{centered},
        denom_(eigenValsSq.size()),
        results_{std::numeric_limits<double>::infinity(), 0.0},
        policy_{std::move(policy)} {}

  // Split ctor
  explicit Worker(const Worker& other, const RcppParallel::Split)
      : lambdas_{other.lambdas_},
        eigenValsSq_{other.eigenValsSq_},
        nrow_{other.nrow_},
        centered_{other.centered_},
        denom_(other.denom_.size()),
        results_{std::numeric_limits<double>::infinity(), 0.0},
        policy_{other.policy_} {}

  // parallelReduce work operator
  void operator()(const std::size_t begin, const std::size_t end) {
    for (Eigen::Index lambdaIdx{static_cast<Eigen::Index>(begin)},
         endIdx{static_cast<Eigen::Index>(end)};
         lambdaIdx < endIdx; ++lambdaIdx) {
      const double lambda{lambdas_[lambdaIdx]};

      // denom_ is reused to avoid temporary allocations and repeated
      // computations
      denom_ = eigenValsSq_ + lambda;

      // Calculate GCV or LOOCV
      const double cv{
          policy_.evaluate(lambda, denom_, eigenValsSq_, nrow_, centered_)};

      if (cv < results_.first) {
        results_.first = cv;
        results_.second = lambda;
      }
    }
  }

  // Join logic for parallel reduction
  void join(const Worker& other) {
    if (other.results_.first < results_.first) {
      results_ = other.results_;
    }
  }
};

};  // namespace Grid::Deterministic
