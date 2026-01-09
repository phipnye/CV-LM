#pragma once

#include <RcppEigen.h>

#include <utility>

namespace CV::Utils {

// RAII for setting rounding mode
class ScopedRoundingMode {
  const int oldMode_;

 public:
  explicit ScopedRoundingMode(int mode);
  ~ScopedRoundingMode();
  ScopedRoundingMode(const ScopedRoundingMode&) = delete;
  ScopedRoundingMode& operator=(const ScopedRoundingMode&) = delete;
  ScopedRoundingMode(const ScopedRoundingMode&&) = delete;
  ScopedRoundingMode& operator=(ScopedRoundingMode&&) = delete;
};

// Confirm valid value for K
[[nodiscard]] int kCheck(int nrow, int k0);

// Generates fold assignments
[[nodiscard]] std::pair<Eigen::VectorXi, Eigen::VectorXi> setupFolds(int seed,
                                                                     int nrow,
                                                                     int k);

// Determine the min and max test fold sizes
[[nodiscard]] std::pair<Eigen::Index, Eigen::Index> testSizeExtrema(
    const Eigen::VectorXi& foldSizes);

// Split the training and test data indicies
void testTrainSplit(int testID, const Eigen::VectorXi& foldIDs,
                    Eigen::VectorXi& testIdxs, Eigen::VectorXi& trainIdxs);

// Check for success of LDLT decomposition
void checkLdltStatus(Eigen::ComputationInfo info);

}  // namespace CV::Utils
