#pragma once

#include <RcppEigen.h>

#include <utility>

namespace Utils {

// RAII for setting rounding mode
class ScopedRoundingMode {
  const int oldMode_;

 public:
  explicit ScopedRoundingMode(const int mode);
  ~ScopedRoundingMode();
  ScopedRoundingMode(const ScopedRoundingMode&) = delete;
  ScopedRoundingMode& operator=(const ScopedRoundingMode&) = delete;
  ScopedRoundingMode(const ScopedRoundingMode&&) = delete;
  ScopedRoundingMode& operator=(ScopedRoundingMode&&) = delete;
};

// Confirm valid value for K
int kCheck(const int nrow, const int k0);

// Generates fold assignments
std::pair<Eigen::VectorXi, Eigen::VectorXi> cvSetup(const int seed, const int n,
                                                    const int k);

// Determine the min and max test fold sizes
std::pair<Eigen::Index, Eigen::Index> testSizeExtrema(
    const Eigen::VectorXi& foldSizes);

}  // namespace Utils
