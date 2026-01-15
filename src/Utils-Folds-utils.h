#pragma once

#include <RcppEigen.h>

namespace Utils::Folds {

// RAII for setting rounding mode
class ScopedRoundingMode {
  const int oldMode_;

 public:
  explicit ScopedRoundingMode(int mode);
  ~ScopedRoundingMode();

  // Deprecated behavior
  ScopedRoundingMode(const ScopedRoundingMode&) = delete;
  ScopedRoundingMode& operator=(const ScopedRoundingMode&) = delete;
};

// Confirm valid value for the number of folds
[[nodiscard]] int kCheck(int nrow, int k0);

// Container for holding assigned fold information
struct FoldInfo {
  // Eigen objects
  const Eigen::VectorXi testFoldIDs_;
  const Eigen::VectorXi testFoldSizes_;

  // Scalars
  const Eigen::Index maxTestSize_;
  const Eigen::Index maxTrainSize_;
  const int nrow_;

  // Ctor
  explicit FoldInfo(int seed, int nrow, int k);
  FoldInfo(const FoldInfo& other) = delete;

  // Split the training and test data indicies
  void testTrainSplit(int testID, Eigen::VectorXi& testIdxs,
                      Eigen::VectorXi& trainIdxs) const;
};

}  // namespace Utils::Folds
