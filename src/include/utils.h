#pragma once

#include <RcppEigen.h>

#include <utility>

namespace CV::Utils {

// RAII for setting rounding mode
class ScopedRoundingMode {
  int oldMode_;

 public:
  explicit ScopedRoundingMode(int mode);
  ~ScopedRoundingMode();
  ScopedRoundingMode(const ScopedRoundingMode&) = delete;
  ScopedRoundingMode& operator=(const ScopedRoundingMode&) = delete;
  ScopedRoundingMode(const ScopedRoundingMode&&) = delete;
  ScopedRoundingMode& operator=(ScopedRoundingMode&&) = delete;
};

// Confirm valid value for K
int kCheck(int nrow, int k0);

// Generates fold assignments
std::pair<Eigen::VectorXi, Eigen::VectorXi> cvSetup(int seed, int n, int k);

}  // namespace CV::Utils
