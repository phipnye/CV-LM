#include "Grid-Deterministic-WorkerPolicy.h"

namespace Grid::Deterministic {

namespace GCV {

// Main ctor
WorkerPolicy::WorkerPolicy(const Eigen::ArrayXd& utySq, const double rssNull)
    : utySq_{utySq}, rssNull_{rssNull} {}

// Compute generalized cross-validation result
double WorkerPolicy::computeCV(const double lambda, const Eigen::ArrayXd& denom,
                               const Eigen::ArrayXd& eigenValsSq,
                               const Eigen::Index nrow,
                               const bool centered) const {
  // trace(H) = sum(eigenVals^2 / (eigenVals^2 + lambda)) [ESL p. 68]
  double traceH{(eigenValsSq / denom).sum()};

  // Add 1 for the unpenalized intercept if data was centered
  if (centered) {
    traceH += 1.0;
  }

  // RSS = sum_{i=1}^{r}((lambda^2 + (eigenVal_i^2 + lambda)^2) * u_i'y^2) +
  // ||(I - UU')y||^2
  const double rss{rssNull_ +
                   ((lambda * lambda) * utySq_ / denom.square()).sum()};

  // GCV = MSE / (1 - trace(H) / n)^2
  const double meanResidLeverage{1.0 - (traceH / static_cast<double>(nrow))};
  return rss /
         (static_cast<double>(nrow) * meanResidLeverage * meanResidLeverage);
}

}  // namespace GCV

namespace LOOCV {

// Main ctor
WorkerPolicy::WorkerPolicy(const Eigen::VectorXd& yNull,
                           const Eigen::MatrixXd& u, const Eigen::MatrixXd& uSq,
                           const Eigen::ArrayXd& uty, const Eigen::Index nrow,
                           const Eigen::Index eigenValSize)
    : diagS_(eigenValSize),
      diagH_(nrow),
      resid_(nrow),
      yNull_{yNull},
      u_{u},
      uSq_{uSq},
      uty_{uty} {}

// Copy ctor (for splitting)
WorkerPolicy::WorkerPolicy(const WorkerPolicy& other)
    : diagS_(other.diagS_.size()),
      diagH_(other.diagH_.size()),
      resid_(other.resid_.size()),
      yNull_{other.yNull_},
      u_{other.u_},
      uSq_{other.uSq_},
      uty_{other.uty_} {}

// Compute leave-one-out CV result
double WorkerPolicy::computeCV(const double lambda, const Eigen::ArrayXd& denom,
                               const Eigen::ArrayXd& eigenValsSq,
                               const Eigen::Index nrow,
                               const bool centered) const {
  // Calculate the diagonal of the Ridge shrinkage matrix = eigenVal_i^2 /
  // (eigenVal_i^2 + lambda)
  diagS_ = eigenValsSq / denom;

  // Diagonal of hat matrix = diag(U * shrinkage * U'): optimized as diag(H) =
  // (U^2) * diag(shrinkage)
  diagH_.matrix().noalias() = uSq_ * diagS_.matrix();

  // Add 1/n to account for the unpenalized intercept if the data was centered
  if (centered) {
    diagH_ += (1.0 / static_cast<double>(nrow));
  }

  // Calculate Ridge residuals: e = y - y_hat = yNull + U * [(lambda /
  // (eigenVals^2 + lambda)) * U'y]
  resid_.noalias() = yNull_ + u_ * ((lambda / denom) * uty_).matrix();

  // LOOCV = mean((e_i / (1 - h_ii))^2)
  return (resid_.array() / (1.0 - diagH_)).square().mean();
}

}  // namespace LOOCV

}  // namespace Grid::Deterministic
