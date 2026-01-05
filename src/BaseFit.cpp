#include "include/BaseFit.h"

#include <RcppEigen.h>

namespace CV {

// GCV = MSE / (1 - trace(H)/n)^2
double BaseFit::gcv() const {
  const double mrl{meanResidualLeverage()};
  return mse() / (mrl * mrl);
}

// LOOCV_error_i = e_i / (1 - h_ii))
double BaseFit::loocv() const {
  return (residuals().array() / (1.0 - hatDiagonal())).square().mean();
}

}  // namespace CV
