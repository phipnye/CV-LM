#include "Grid-Utils-utils.h"

#include <Rcpp.h>
#include <RcppEigen.h>

namespace Grid::Utils {

// Make sure SVD info report successful decomposition
void checkSvdStatus(const Eigen::ComputationInfo info) {
  if (info == Eigen::Success) {
    return;
  }

  switch (info) {
    case Eigen::NumericalIssue:
      Rcpp::stop(
          "SVD failed: Numerical issue encountered (likely NaN or Inf values "
          "in the input matrix).");
      // break; - Unreachable
    case Eigen::NoConvergence:
      Rcpp::stop(
          "SVD failed: Algorithm failed to converge (likely due to extreme "
          "ill-conditioning or disparate scales).");
      // break;
    case Eigen::InvalidInput:
      Rcpp::stop("SVD failed: Invalid input.");
      // break;
    default:
      // As of time of writing (2026) this line should never execute as all
      // enum values have been accounted for
      Rcpp::stop("SVD failed: An unknown error occurred during decomposition.");
      // break;
  }
}

// Perform singular decomposition of X and compute thin U
Eigen::BDCSVD<Eigen::MatrixXd> svdDecompose(
    const Eigen::Map<Eigen::MatrixXd>& x, const double threshold) {
  // Perform SVD on full data once and retrieve thin U which we need for GCV and
  // LOOCV
  Eigen::BDCSVD<Eigen::MatrixXd> svd{x, Eigen::ComputeThinU};

  // Set threshold at which signular values are considered zero "A singular
  // value will be considered nonzero if its value is strictly greater than
  // |singularvalue|⩽threshold×|maxsingularvalue|." - this threshold only
  // affects other methods like solve and rank, not the decomposition itself
  svd.setThreshold(threshold);

  // Confirm successful decomposition
  checkSvdStatus(svd.info());
  return svd;
}

}  // namespace Grid::Utils
