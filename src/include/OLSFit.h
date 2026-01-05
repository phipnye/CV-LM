#pragma once

#include <RcppEigen.h>

namespace CV::OLS {

class OLSFit {
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_;
  const Eigen::Index nrow_;
  const Eigen::Index rank_;
  Eigen::VectorXd qty_;
  Eigen::ArrayXd diagH_{};

 public:
  explicit OLSFit(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                  const bool needHat);

  double gcv() const;
  double loocv() const;

 private:
  double rss() const;
  double mse() const;
  double meanResidualLeverage() const;
  Eigen::VectorXd residuals() const;
};

}  // namespace CV::OLS
