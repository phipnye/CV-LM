#pragma once

#include <RcppEigen.h>

namespace Grid::Deterministic {

namespace GCV {

struct WorkerPolicy {
  const double rssNull_;
  const Eigen::ArrayXd& utySq_;

  WorkerPolicy(const Eigen::ArrayXd& utySq, const double rssNull);

  double evaluate(const double lambda, const Eigen::ArrayXd& denom,
                  const Eigen::ArrayXd& eigenValsSq, const Eigen::VectorXd&,
                  const Eigen::Index nrow, const bool centered) const;
};

}  // namespace GCV

namespace LOOCV {

struct WorkerPolicy {
  const Eigen::VectorXd& yNull_;
  const Eigen::MatrixXd& u_;
  const Eigen::MatrixXd& uSq_;

  // Thread-specific buffers (make them mutable for evaluation)
  mutable Eigen::ArrayXd diagS_;
  mutable Eigen::ArrayXd diagH_;
  mutable Eigen::VectorXd resid_;

  WorkerPolicy(const Eigen::VectorXd& yNull, const Eigen::MatrixXd& u,
               const Eigen::MatrixXd& uSq, const Eigen::Index nrow,
               const Eigen::Index eigenValSize);

  WorkerPolicy(const WorkerPolicy& other);

  double evaluate(const double lambda, const Eigen::ArrayXd& denom,
                  const Eigen::ArrayXd& eigenValsSq, const Eigen::VectorXd& uty,
                  const Eigen::Index nrow, const bool centered) const;
};

}  // namespace LOOCV

}  // namespace Grid::Deterministic
