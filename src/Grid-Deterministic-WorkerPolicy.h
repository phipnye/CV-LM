#pragma once

#include <RcppEigen.h>

namespace Grid::Deterministic {

namespace GCV {

class WorkerPolicy {
  // Thread-specific buffers
  Eigen::VectorXd coordShrinkFactorsDenom_;

  // References
  const Eigen::VectorXd& uTySq_;
  const Eigen::VectorXd& singularValsSq_;

  // Scalars
  const double rssNull_;
  const Eigen::Index nrow_;

  // Flag for whether data was centered in R
  const bool centered_;

 public:
  // Main ctor
  WorkerPolicy(const Eigen::VectorXd& uTySq,
               const Eigen::VectorXd& singularValsSq, double rssNull,
               Eigen::Index nrow, bool centered);

  // Needs to be copyable for splitting and moveable for std::move in Worker
  // ctor
  WorkerPolicy(const WorkerPolicy&);
  WorkerPolicy(WorkerPolicy&&) = default;

  // Assignments shouldn't be necessary
  WorkerPolicy& operator=(const WorkerPolicy&) = delete;
  WorkerPolicy& operator=(WorkerPolicy&&) = delete;

  // Calculate generalized cross-validation result
  [[nodiscard]] double computeCV(double lambda);
};

}  // namespace GCV

namespace LOOCV {

class WorkerPolicy {
  // Thread-specific buffers
  Eigen::VectorXd coordShrinkFactors_;
  Eigen::VectorXd coordShrinkFactorsDenom_;
  Eigen::VectorXd diagHat_;
  Eigen::VectorXd resid_;

  // References
  const Eigen::VectorXd& yNull_;
  const Eigen::MatrixXd& u_;
  const Eigen::MatrixXd& uSq_;
  const Eigen::VectorXd& uTy_;
  const Eigen::VectorXd& singularValsSq_;

  // Scalars
  Eigen::Index nrow_;

  // Flag for whether data was centered in R
  const bool centered_;

 public:
  // Main ctor
  WorkerPolicy(const Eigen::VectorXd& yNull, const Eigen::MatrixXd& u,
               const Eigen::MatrixXd& uSq, const Eigen::VectorXd& uTy,
               const Eigen::VectorXd& singularValsSq, Eigen::Index nrow,
               bool centered);

  // Needs to be copyable for splitting and moveable for std::move in Worker
  // ctor
  WorkerPolicy(const WorkerPolicy& other);
  WorkerPolicy(WorkerPolicy&&) = default;

  // Assignments shouldn't be necessary
  WorkerPolicy& operator=(const WorkerPolicy&) = delete;
  WorkerPolicy& operator=(WorkerPolicy&&) = delete;

  // Calculate leave-one-out CV result
  [[nodiscard]] double computeCV(double lambda);
};

}  // namespace LOOCV

}  // namespace Grid::Deterministic
