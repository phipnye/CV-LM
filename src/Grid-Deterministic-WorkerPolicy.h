#pragma once

#include <RcppEigen.h>

namespace Grid::Deterministic {

namespace GCV {

struct WorkerPolicy {
  const Eigen::ArrayXd& utySq_;
  const double rssNull_;

  WorkerPolicy(const Eigen::ArrayXd& utySq, const double rssNull);
  WorkerPolicy(const WorkerPolicy&) = default;
  WorkerPolicy(WorkerPolicy&&) = default;
  WorkerPolicy& operator=(const WorkerPolicy&) = delete;
  WorkerPolicy& operator=(WorkerPolicy&&) = delete;

  double evaluate(const double lambda, const Eigen::ArrayXd& denom,
                  const Eigen::ArrayXd& eigenValsSq, const Eigen::Index nrow,
                  const bool centered) const;
};

}  // namespace GCV

namespace LOOCV {

struct WorkerPolicy {
  // References
  const Eigen::VectorXd& yNull_;
  const Eigen::MatrixXd& u_;
  const Eigen::MatrixXd& uSq_;
  const Eigen::VectorXd& uty_;

  // Thread-specific buffers (mutable for evaluation)
  mutable Eigen::ArrayXd diagS_;
  mutable Eigen::ArrayXd diagH_;
  mutable Eigen::VectorXd resid_;

  WorkerPolicy(const Eigen::VectorXd& yNull, const Eigen::MatrixXd& u,
               const Eigen::MatrixXd& uSq, const Eigen::VectorXd& uty,
               const Eigen::Index nrow, const Eigen::Index eigenValSize);
  WorkerPolicy(const WorkerPolicy& other);
  WorkerPolicy(WorkerPolicy&&) = default;
  WorkerPolicy& operator=(const WorkerPolicy&) = delete;
  WorkerPolicy& operator=(WorkerPolicy&&) = delete;

  double evaluate(const double lambda, const Eigen::ArrayXd& denom,
                  const Eigen::ArrayXd& eigenValsSq, const Eigen::Index nrow,
                  const bool centered) const;
};

}  // namespace LOOCV

}  // namespace Grid::Deterministic
