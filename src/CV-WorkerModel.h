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

// Use primal form
namespace Narrow {

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

}  // namespace Narrow

namespace Wide {

struct WorkerModel {
  Eigen::ComputationInfo info_;
  const double lambda_;
  Eigen::MatrixXd xxtLambda_;
  Eigen::LDLT<Eigen::MatrixXd> ldlt_;

  // Additional data buffer for dual coefficients where beta = X' * alpha
  Eigen::VectorXd alpha_;

  explicit WorkerModel(Eigen::Index maxTrainSize, double lambda);
  explicit WorkerModel(const WorkerModel& other);
  WorkerModel(WorkerModel&&) = default;
  WorkerModel& operator=(const WorkerModel&) = delete;
  WorkerModel& operator=(WorkerModel&&) = delete;

  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta);
};

}  // namespace Wide

}  // namespace Ridge

}  // namespace CV
