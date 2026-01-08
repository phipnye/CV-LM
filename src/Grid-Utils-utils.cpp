#include "Grid-Utils-utils.h"

#include <Rcpp.h>
#include <RcppEigen.h>

namespace Grid::Utils {

void checkSvdStatus(const Eigen::ComputationInfo info) {
  if (info == Eigen::Success) {
    return;
  }

  switch (info) {
    case Eigen::NumericalIssue:
      Rcpp::stop(
          "SVD failed: Numerical issue encountered (likely NaN or Inf values "
          "in the input matrix).");
      break;
    case Eigen::NoConvergence:
      Rcpp::stop(
          "SVD failed: Algorithm failed to converge (likely due to extreme "
          "ill-conditioning or disparate scales).");
      break;
    case Eigen::InvalidInput:
      Rcpp::stop("SVD failed: Invalid input.");
      break;
    default:
      Rcpp::stop("SVD failed: An unknown error occurred during decomposition.");
      break;
  }
}

Eigen::BDCSVD<Eigen::MatrixXd> svdDecompose(const Eigen::MatrixXd& x,
                                            const double threshold) {
  // Pre-allocate memory for SVD
  Eigen::BDCSVD<Eigen::MatrixXd> svd(x.rows(), x.cols(), Eigen::ComputeThinU);

  // Set threshold at which signular values are considered zero "A singular
  // value will be considered nonzero if its value is strictly greater than
  // |singularvalue|⩽threshold×|maxsingularvalue|."
  svd.setThreshold(threshold);

  // Perform SVD on full data once (for GCV, we only need singular values and
  // U'y)
  svd.compute(x);

  // Confirm successful decomposition
  checkSvdStatus(svd.info());
  return svd;
}

}  // namespace Grid::Utils
