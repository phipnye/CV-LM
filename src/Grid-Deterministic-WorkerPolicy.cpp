#include "Grid-Deterministic-WorkerPolicy.h"

#include "Stats-computations.h"

namespace Grid::Deterministic {

namespace GCV {

// Main ctor
WorkerPolicy::WorkerPolicy(const Eigen::VectorXd& uTySq,
                           const Eigen::VectorXd& singularValsSq,
                           const double rssNull, const Eigen::Index nrow,
                           const bool centered)
    : coordShrinkFactorsDenom_{singularValsSq.size()},
      uTySq_{uTySq},
      singularValsSq_{singularValsSq},
      rssNull_{rssNull},
      nrow_{nrow},
      centered_{centered} {}

// Copy ctor (for splitting)
WorkerPolicy::WorkerPolicy(const WorkerPolicy& other)
    : coordShrinkFactorsDenom_{other.coordShrinkFactorsDenom_.size()},
      uTySq_{other.uTySq_},
      singularValsSq_{other.singularValsSq_},
      rssNull_{other.rssNull_},
      nrow_{other.nrow_},
      centered_{other.centered_} {}

// Compute generalized cross-validation result
double WorkerPolicy::computeCV(const double lambda) {
  // trace(H) = sum(d_j^2 / (d_j^2 + lambda)) [ESL p. 68]
  coordShrinkFactorsDenom_ = singularValsSq_.array() + lambda;
  double traceHat{
      (singularValsSq_.array() / coordShrinkFactorsDenom_.array()).sum()};

  // Add 1 for the unpenalized intercept if data was centered
  if (centered_) {
    traceHat += 1.0;
  }

  // RSS = sum_{i=1}^{r}((lambda^2 + (d_i^2 + lambda)^2) * u_i'y^2) +
  // ||(I - UU')y||^2
  const double rss{rssNull_ + ((lambda * lambda) * uTySq_.array() /
                               coordShrinkFactorsDenom_.array().square())
                                  .sum()};
  return Stats::gcv(rss, traceHat, nrow_);
}

}  // namespace GCV

namespace LOOCV {

// Main ctor
WorkerPolicy::WorkerPolicy(const Eigen::VectorXd& yNull,
                           const Eigen::MatrixXd& u, const Eigen::MatrixXd& uSq,
                           const Eigen::VectorXd& uTy,
                           const Eigen::VectorXd& singularValsSq,
                           const Eigen::Index nrow, const bool centered)
    : coordShrinkFactors_{singularValsSq.size()},
      coordShrinkFactorsDenom_{singularValsSq.size()},
      diagHat_{nrow},
      resid_{nrow},
      yNull_{yNull},
      u_{u},
      uSq_{uSq},
      uTy_{uTy},
      singularValsSq_{singularValsSq},
      nrow_{nrow},
      centered_{centered} {}

// Copy ctor (for splitting)
WorkerPolicy::WorkerPolicy(const WorkerPolicy& other)
    : coordShrinkFactors_{other.coordShrinkFactors_.size()},
      coordShrinkFactorsDenom_{other.coordShrinkFactorsDenom_.size()},
      diagHat_{other.diagHat_.size()},
      resid_{other.resid_.size()},
      yNull_{other.yNull_},
      u_{other.u_},
      uSq_{other.uSq_},
      uTy_{other.uTy_},
      singularValsSq_{other.singularValsSq_},
      nrow_{other.nrow_},
      centered_{other.centered_} {}

// Compute leave-one-out CV result
double WorkerPolicy::computeCV(const double lambda) {
  // Compute the coordinate shrinkage factors [ESL p.66] d_j^2 / (d_j^2 +
  // lambda) - this may result in 0/0 division for a design matrix not of full
  // column rank with lambda = 0.0
  coordShrinkFactorsDenom_ = singularValsSq_.array() + lambda;
  coordShrinkFactors_ =
      singularValsSq_.array() / coordShrinkFactorsDenom_.array();

  // Diagonal of hat matrix = diag(U * shrinkage * U'): optimized as diag(H) =
  // (U^2) * diag(shrinkage)
  diagHat_.noalias() = uSq_ * coordShrinkFactors_;

  // Add 1/n to account for the unpenalized intercept if the data was centered
  if (centered_) {
    diagHat_.array() += (1.0 / static_cast<double>(nrow_));
  }

  // Calculate Ridge residuals: e = y - y_hat = yNull + U * [(lambda /
  // (singularVals^2 + lambda)) * U'y]
  resid_.noalias() =
      yNull_ +
      (u_ *
       (lambda * (uTy_.array() / coordShrinkFactorsDenom_.array())).matrix());

  return Stats::loocv(resid_, diagHat_);
}

}  // namespace LOOCV

}  // namespace Grid::Deterministic
