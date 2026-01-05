#pragma once

#include <RcppEigen.h>

namespace CV {

namespace OLS {

struct WorkerModel {
  WorkerModel() = default;
  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta) const;
};

}  // namespace OLS

namespace Ridge {

struct WorkerModel {
  const double lambda_;
  mutable Eigen::MatrixXd xtxLambda_;  // mutable allows reuse in const methods

  explicit WorkerModel(const double lambda, const Eigen::Index ncol);
  explicit WorkerModel(const WorkerModel& other);

  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta) const;
};

}  // namespace Ridge

}  // namespace CV
