#include "Grid-Generator.h"

#include <RcppEigen.h>

#include <cmath>

namespace Grid {

Generator::Generator(const double maxLambda, const double precision)
    : maxLambda_{maxLambda},
      precision_{precision},
      nFull_{static_cast<Eigen::Index>(std::floor(maxLambda / precision))},
      hasTail_{(maxLambda - (static_cast<double>(nFull_) * precision)) > 1e-9} {
}

Eigen::Index Generator::size() const noexcept { return nFull_ + 1 + hasTail_; }

double Generator::operator[](const Eigen::Index idx) const {
  return idx <= nFull_ ? static_cast<double>(idx) * precision_ : maxLambda_;
}

}  // namespace Grid
