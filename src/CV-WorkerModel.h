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
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> qtz_;

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

class WorkerModel {
 public:
  // Flag indicating whether the decomposition can fail
  static constexpr bool canFail{true};  // BDCSVD can fail

 private:
  // Members
  Eigen::BDCSVD<Eigen::MatrixXd> udvT_;
  Eigen::VectorXd uTy_;
  Eigen::VectorXd singularVals_;
  Eigen::VectorXd singularShrinkFactors_;
  const double lambda_;
  Eigen::ComputationInfo info_;

 public:
  // Main ctor
  explicit WorkerModel(Eigen::Index ncol, Eigen::Index maxTrainSize,
                       double threshold, double lambda);

  // Copy ctor for worker splits
  WorkerModel(const WorkerModel& other);

  // Fit ridge regression coefficients to training set
  void computeBeta(const Eigen::Ref<const Eigen::MatrixXd>& xTrain,
                   const Eigen::Ref<const Eigen::VectorXd>& yTrain,
                   Eigen::VectorXd& beta);

  // Get decomposition success info
  [[nodiscard]] Eigen::ComputationInfo getInfo() const noexcept;
};

}  // namespace Ridge

}  // namespace CV
