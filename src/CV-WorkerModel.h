#pragma once

#include <RcppEigen.h>

namespace CV {

namespace OLS {

struct WorkerModel {
  Eigen::ComputationInfo info_{
      Eigen::Success};  // CompleteOrthogonalDecomposition::info
                        // always returns success
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_;

  explicit WorkerModel(Eigen::Index ncol, Eigen::Index maxTrainSize,
                       double threshold);
  explicit WorkerModel(const WorkerModel& other);
  WorkerModel(WorkerModel&&) = default;
  WorkerModel& operator=(const WorkerModel&) = delete;
  WorkerModel& operator=(WorkerModel&&) = delete;

  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta);
};

}  // namespace OLS

namespace Ridge {

struct WorkerModel {
  Eigen::ComputationInfo info_;
  const double lambda_;
  Eigen::MatrixXd xtxLambda_;
  Eigen::LDLT<Eigen::MatrixXd> ldlt_;

  explicit WorkerModel(Eigen::Index ncol, double lambda);
  explicit WorkerModel(const WorkerModel& other);
  WorkerModel(WorkerModel&&) = default;
  WorkerModel& operator=(const WorkerModel&) = delete;
  WorkerModel& operator=(WorkerModel&&) = delete;

  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta);
};

}  // namespace Ridge

}  // namespace CV
