#ifndef CV_LM_GRID_GENERATOR_H
#define CV_LM_GRID_GENERATOR_H

#include <RcppEigen.h>

namespace Grid {

// Lightweight description of the grid of shrinkage parameter values to search
// across
class Generator {
  const double maxLambda_;
  const double precision_;
  const Eigen::Index size_;

 public:
  explicit Generator(double maxLambda, double precision);

  [[nodiscard]] Eigen::Index size() const noexcept;
  [[nodiscard]] double operator[](Eigen::Index idx) const noexcept;
};

}  // namespace Grid

#endif  // CV_LM_GRID_GENERATOR_H
