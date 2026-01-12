#pragma once

#include <RcppEigen.h>

namespace Grid::Deterministic {

namespace GCV {

class WorkerPolicy {
  const Eigen::ArrayXd& utySq_;
  const double rssNull_;

 public:
  // Main ctor
  WorkerPolicy(const Eigen::ArrayXd& utySq, double rssNull);

  // Needs to be copyable for splitting and moveable for std::move in Worker
  // ctor
  WorkerPolicy(const WorkerPolicy&) = default;
  WorkerPolicy(WorkerPolicy&&) = default;

  // Assignments shouldn't be necessary
  WorkerPolicy& operator=(const WorkerPolicy&) = delete;
  WorkerPolicy& operator=(WorkerPolicy&&) = delete;

  // Calculate generalized cross-validation result
  [[nodiscard]] double computeCV(double lambda, const Eigen::ArrayXd& denom,
                                 const Eigen::ArrayXd& eigenValsSq,
                                 Eigen::Index nrow, bool centered) const;
};

}  // namespace GCV

namespace LOOCV {

class WorkerPolicy {
  // Thread-specific buffers (mutable for evaluation)
  mutable Eigen::ArrayXd diagS_;
  mutable Eigen::ArrayXd diagH_;
  mutable Eigen::VectorXd resid_;

  // References
  const Eigen::VectorXd& yNull_;
  const Eigen::MatrixXd& u_;
  const Eigen::MatrixXd& uSq_;
  const Eigen::ArrayXd& uty_;

 public:
  // Main ctor
  WorkerPolicy(const Eigen::VectorXd& yNull, const Eigen::MatrixXd& u,
               const Eigen::MatrixXd& uSq, const Eigen::ArrayXd& uty,
               Eigen::Index nrow, Eigen::Index eigenValSize);

  // Needs to be copyable for splitting and moveable for std::move in Worker
  // ctor
  WorkerPolicy(const WorkerPolicy& other);
  WorkerPolicy(WorkerPolicy&&) = default;

  // Assignments shouldn't be necessary
  WorkerPolicy& operator=(const WorkerPolicy&) = delete;
  WorkerPolicy& operator=(WorkerPolicy&&) = delete;

  // Calculate leave-one-out CV result
  [[nodiscard]] double computeCV(double lambda, const Eigen::ArrayXd& denom,
                                 const Eigen::ArrayXd& eigenValsSq,
                                 Eigen::Index nrow, bool centered) const;
};

}  // namespace LOOCV

}  // namespace Grid::Deterministic
