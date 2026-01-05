#pragma once

#include <RcppEigen.h>

namespace CV::Ridge {

class Fit {
  const Eigen::Index nrow_;
  const Eigen::Index ncol_;
  const double lambda_;
  const bool centered_;
  const Eigen::MatrixXd xtxLambdaInv_;
  const Eigen::VectorXd resid_;
  Eigen::ArrayXd diagH_{};

 public:
  explicit Fit(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
               const double lambda, const bool centered, const bool needHat);

  double gcv() const;
  double loocv() const;

 private:
  double rss() const;
  double mse() const;
  double meanResidualLeverage() const;
  double traceH() const;
};

}  // namespace CV::Ridge
