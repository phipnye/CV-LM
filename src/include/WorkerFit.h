#pragma once

#include <RcppEigen.h>

namespace CV {

namespace OLS {

struct WorkerFit {
  WorkerFit() = default;
  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta) const;
};

}  // namespace OLS

namespace Ridge {

struct WorkerFit {
  const double lambda_;
  mutable Eigen::MatrixXd xtxLambda_;  // mutable allows reuse in const methods

  explicit WorkerFit(const double lambda, const Eigen::Index ncol);
  explicit WorkerFit(const WorkerFit& other);

  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta) const;
};

}  // namespace Ridge

}  // namespace CV
