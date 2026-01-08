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
               double lambda, bool centered, bool needHat);

  [[nodiscard]] double gcv() const;
  [[nodiscard]] double loocv() const;

 private:
  [[nodiscard]] double rss() const;
  [[nodiscard]] double mse() const;
  [[nodiscard]] double meanResidualLeverage() const;
  [[nodiscard]] double traceH() const;
};

}  // namespace CV::Ridge
