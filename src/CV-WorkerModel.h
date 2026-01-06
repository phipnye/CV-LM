#pragma once

#include <RcppEigen.h>

namespace CV {

namespace OLS {

struct WorkerModel {
  WorkerModel() = default;
  WorkerModel(const WorkerModel&) = default;
  WorkerModel(WorkerModel&&) = default;
  WorkerModel& operator=(const WorkerModel&) = delete;
  WorkerModel& operator=(WorkerModel&&) = delete;

  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta) const;
};

}  // namespace OLS

namespace Ridge {

struct WorkerModel {
  const double lambda_;
  mutable Eigen::MatrixXd
      xtxLambda_;  // mutable allows reuse in beta computation

  explicit WorkerModel(const double lambda, const Eigen::Index ncol);
  explicit WorkerModel(const WorkerModel& other);
  WorkerModel(WorkerModel&&) = default;
  WorkerModel& operator=(const WorkerModel&) = delete;
  WorkerModel& operator=(WorkerModel&&) = delete;

  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta) const;
};

}  // namespace Ridge

}  // namespace CV
