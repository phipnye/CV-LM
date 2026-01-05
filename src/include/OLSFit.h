#pragma once

#include <RcppEigen.h>

#include "BaseFit.h"

namespace CV::OLS {

class OLSFit : public BaseFit {
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_;
  const Eigen::Index nrow_;
  const Eigen::Index rank_;
  Eigen::VectorXd qty_;
  Eigen::ArrayXd diagH_{};

 public:
  explicit OLSFit(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                  const bool needHat);

 private:
  double rss() const;
  double mse() const override;
  double meanResidualLeverage() const override;
  Eigen::Ref<const Eigen::VectorXd> residuals() const override;
  const Eigen::ArrayXd& hatDiagonal() const override;
};

}  // namespace CV::OLS
