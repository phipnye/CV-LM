#pragma once

#include <RcppEigen.h>

namespace CV::OLS {

class Fit {
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_;
  const Eigen::Index nrow_;
  const Eigen::Index rank_;
  Eigen::VectorXd qty_;
  Eigen::ArrayXd diagH_{};

 public:
  explicit Fit(Eigen::VectorXd y, const Eigen::MatrixXd& x, double threshold,
               bool needHat);

  [[nodiscard]] double gcv() const;
  [[nodiscard]] double loocv() const;

 private:
  [[nodiscard]] double rss() const;
  [[nodiscard]] double mse() const;
  [[nodiscard]] double meanResidualLeverage() const;
  [[nodiscard]] Eigen::VectorXd residuals() const;
};

}  // namespace CV::OLS
