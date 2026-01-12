#pragma once

#include <RcppEigen.h>

namespace CV {

namespace OLS {

class WorkerModel {
 public:
  // Flag indicating whether the decomposition can fail
  static constexpr bool canFail{false};  // COD is always succesful

 private:
  // Members
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_;

 public:
  // Main ctor
  explicit WorkerModel(Eigen::Index ncol, Eigen::Index maxTrainSize,
                       double threshold);

  // Copy ctor for worker splits
  WorkerModel(const WorkerModel& other);

  // Fit OLS coefficients to training set
  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta);
};

}  // namespace OLS

namespace Ridge {

// Use primal form
namespace Narrow {

class WorkerModel {
 public:
  // Flag indicating whether the decomposition can fail
  static constexpr bool canFail{true};  // ldlt can fail because of a zero pivot

 private:
  // Members
  Eigen::LDLT<Eigen::MatrixXd> ldlt_;
  Eigen::MatrixXd xtxLambda_;
  const double lambda_;
  Eigen::ComputationInfo info_;

 public:
  // Main ctor
  explicit WorkerModel(Eigen::Index ncol, double lambda);

  // Copy ctor for worker splits
  WorkerModel(const WorkerModel& other);

  // Fit ridge regression coefficients to training set
  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta);

  // Get decomposition success info
  [[nodiscard]] Eigen::ComputationInfo getInfo() const noexcept;
};

}  // namespace Narrow

namespace Wide {

class WorkerModel {
 public:
  // Flag indicating whether the decomposition can fail
  static constexpr bool canFail{true};  // ldlt can fail because of a zero pivot

 private:
  // Members
  Eigen::LDLT<Eigen::MatrixXd> ldlt_;
  Eigen::MatrixXd xxtLambda_;
  Eigen::VectorXd alpha_;  // additional data buffer for dual coefficients where
                           // beta = X' * alpha
  const double lambda_;
  Eigen::ComputationInfo info_;

 public:
  // Main ctor
  explicit WorkerModel(Eigen::Index maxTrainSize, double lambda);

  // Copy ctor for worker splits
  WorkerModel(const WorkerModel& other);

  // Fit ridge regression coefficients to training set
  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta);

  // Get decomposition success info
  [[nodiscard]] Eigen::ComputationInfo getInfo() const noexcept;
};

}  // namespace Wide

}  // namespace Ridge

}  // namespace CV
