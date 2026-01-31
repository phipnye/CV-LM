#ifndef CV_LM_GRID_GENERATOR_H
#define CV_LM_GRID_GENERATOR_H

#include <RcppArmadillo.h>

namespace Grid {

// Lightweight description of the grid of shrinkage parameter values to search
// across
class Generator {
  const double maxLambda_;
  const double precision_;
  const arma::uword size_;

 public:
  explicit Generator(double maxLambda, double precision);

  [[nodiscard]] arma::uword size() const noexcept;
  [[nodiscard]] double operator[](arma::uword idx) const noexcept;
};

}  // namespace Grid

#endif  // CV_LM_GRID_GENERATOR_H
