#ifndef CV_LM_GRID_DETERMINISTIC_WORKERMODEL_H
#define CV_LM_GRID_DETERMINISTIC_WORKERMODEL_H

#include <RcppEigen.h>

#include "Enums.h"
#include "Stats.h"

namespace Grid::Deterministic {

namespace GCV {

template <Enums::CenteringMethod Centering>
class WorkerModel {
  // Thread-specific buffers
  Eigen::VectorXd coordShrinkFactorsDenom_;

  // References
  const Eigen::VectorXd& uTySq_;
  const Eigen::VectorXd& singularValsSq_;

  // Scalars
  const double rssNull_;
  const Eigen::Index nrow_;

 public:
  // Main ctor
  WorkerModel(const Eigen::VectorXd& uTySq,
              const Eigen::VectorXd& singularValsSq, const double rssNull,
              const Eigen::Index nrow)
      : coordShrinkFactorsDenom_{singularValsSq.size()},
        uTySq_{uTySq},
        singularValsSq_{singularValsSq},
        rssNull_{rssNull},
        nrow_{nrow} {}

  // Copy ctor (for splitting)
  WorkerModel(const WorkerModel& other)
      : coordShrinkFactorsDenom_{other.coordShrinkFactorsDenom_.size()},
        uTySq_{other.uTySq_},
        singularValsSq_{other.singularValsSq_},
        rssNull_{other.rssNull_},
        nrow_{other.nrow_} {}

  // Needs to be moveable for worker constructor (buffers stolen, references
  // copied)
  WorkerModel(WorkerModel&&) = default;

  // Assignments shouldn't be necessary
  WorkerModel& operator=(const WorkerModel&) = delete;
  WorkerModel& operator=(WorkerModel&&) = delete;

  // Calculate generalized cross-validation result
  [[nodiscard]] double computeCV(const double lambda) {
    // Coordinate shrinkage factors = d_j^2 / (d_j^2 + lambda)
    coordShrinkFactorsDenom_ = singularValsSq_.array() + lambda;

    // Add 1 for the unpenalized intercept if data was centered
    constexpr double correction{
        Centering == Enums::CenteringMethod::Mean ? 1.0 : 0.0};

    // trace(H) = sum(d_j^2 / (d_j^2 + lambda)) [ESL p. 68]
    const double traceHat{
        (singularValsSq_.array() / coordShrinkFactorsDenom_.array()).sum() +
        correction};

    // RSS = sum_{i=1}^{r}((lambda^2 + (d_i^2 + lambda)^2) * u_i'y^2) +
    // ||(I - UU')y||^2
    const double rss{rssNull_ + ((lambda * lambda) * uTySq_.array() /
                                 coordShrinkFactorsDenom_.array().square())
                                    .sum()};
    return Stats::gcv(rss, traceHat, nrow_);
  }
};

}  // namespace GCV

namespace LOOCV {

template <Enums::CenteringMethod Centering>
class WorkerModel {
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

 public:
  // Main ctor
  WorkerModel(const Eigen::VectorXd& yNull, const Eigen::MatrixXd& u,
              const Eigen::MatrixXd& uSq, const Eigen::VectorXd& uTy,
              const Eigen::VectorXd& singularValsSq, const Eigen::Index nrow)
      : coordShrinkFactors_{singularValsSq.size()},
        coordShrinkFactorsDenom_{singularValsSq.size()},
        diagHat_{nrow},
        resid_{nrow},
        yNull_{yNull},
        u_{u},
        uSq_{uSq},
        uTy_{uTy},
        singularValsSq_{singularValsSq},
        nrow_{nrow} {}

  // Copy ctor (for splitting)
  WorkerModel(const WorkerModel& other)
      : coordShrinkFactors_{other.coordShrinkFactors_.size()},
        coordShrinkFactorsDenom_{other.coordShrinkFactorsDenom_.size()},
        diagHat_{other.diagHat_.size()},
        resid_{other.resid_.size()},
        yNull_{other.yNull_},
        u_{other.u_},
        uSq_{other.uSq_},
        uTy_{other.uTy_},
        singularValsSq_{other.singularValsSq_},
        nrow_{other.nrow_} {}

  // Needs to be moveable for worker constructor (buffers stolen, references
  // copied)
  WorkerModel(WorkerModel&&) = default;

  // Assignments shouldn't be necessary
  WorkerModel& operator=(const WorkerModel&) = delete;
  WorkerModel& operator=(WorkerModel&&) = delete;

  // Calculate leave-one-out CV result
  [[nodiscard]] double computeCV(const double lambda) {
    // Compute the coordinate shrinkage factors [ESL p.66] d_j^2 / (d_j^2 +
    // lambda) - this may result in 0/0 division for a design matrix not of full
    // column rank with lambda = 0.0
    coordShrinkFactorsDenom_ = singularValsSq_.array() + lambda;
    coordShrinkFactors_ =
        singularValsSq_.array() / coordShrinkFactorsDenom_.array();

    // Diagonal of hat matrix = diag(U * shrinkage * U')
    // h_ii = sum_j u_ij^2 (d_j^2 / (d_j^2 + lambda))
    diagHat_.noalias() = uSq_ * coordShrinkFactors_;

    // Add 1/n to account for the unpenalized intercept if the data was centered
    if constexpr (Centering == Enums::CenteringMethod::Mean) {
      diagHat_.array() += (1.0 / static_cast<double>(nrow_));
    }

    // Calculate ridge residuals: e = y - y_hat = yNull + U * [(lambda /
    // (singularVals^2 + lambda)) * U'y]
    resid_.noalias() =
        yNull_ +
        (u_ *
         (lambda * (uTy_.array() / coordShrinkFactorsDenom_.array())).matrix());

    return Stats::loocv(resid_, diagHat_);
  }
};

}  // namespace LOOCV

}  // namespace Grid::Deterministic

#endif  // CV_LM_GRID_DETERMINISTIC_WORKERMODEL_H
