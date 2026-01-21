#pragma once

#include <RcppEigen.h>

namespace Grid {

// Lightweight description of the grid
class Generator {
  const double maxLambda_;
  const double precision_;
  const Eigen::Index
      internalN;           // number of intermediate steps between (0, maxLambda)
  const bool hasTail_;  // precision and maxLambda don't align so manually force
                        // max value

 public:
  explicit Generator(double maxLambda, double precision);

  [[nodiscard]] Eigen::Index size() const noexcept;
  [[nodiscard]] double operator[](Eigen::Index idx) const noexcept;
};

}  // namespace Grid
