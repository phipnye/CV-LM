#pragma once

#include <RcppEigen.h>

namespace CV {

class BaseFit {
 public:
  virtual ~BaseFit() = default;
  double gcv() const;
  double loocv() const;

 protected:
  virtual double mse() const = 0;
  virtual double meanResidualLeverage() const = 0;

  // Return Ref to avoid copying stored members in RidgeFit
  virtual Eigen::Ref<const Eigen::VectorXd> residuals() const = 0;
  virtual const Eigen::ArrayXd& hatDiagonal() const = 0;
};

}  // namespace CV
