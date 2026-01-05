#pragma once

#include <RcppEigen.h>

#include "BaseFit.h"

namespace CV::Ridge {

class RidgeFit : public BaseFit {
  const Eigen::Index nrow_;
  const Eigen::Index ncol_;
  const double lambda_;
  const bool centered_;
  const Eigen::MatrixXd xtxLambdaInv_;
  const Eigen::VectorXd resid_;
  Eigen::ArrayXd diagH_{};

 public:
  explicit RidgeFit(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                    const double lambda, const bool centered,
                    const bool needHat);

 private:
  double rss() const;
  double mse() const override;
  double meanResidualLeverage() const override;
  Eigen::Ref<const Eigen::VectorXd> residuals() const override;
  const Eigen::ArrayXd& hatDiagonal() const override;
  double traceH() const;
};

}  // namespace CV::Ridge
