#include "Grid-Utils-utils.h"

#include <RcppEigen.h>

namespace Grid::Utils {

Eigen::BDCSVD<Eigen::MatrixXd> svdDecompose(const Eigen::MatrixXd& x,
                                            const double threshold) {
  // Pre-allocate memory for SVD
  Eigen::BDCSVD<Eigen::MatrixXd> svd(x.rows(), x.cols(), Eigen::ComputeThinU);

  // Set threshold at which signular values are considered zero
  svd.setThreshold(threshold);

  // Perform SVD on full data once (for GCV, we only need singular values and
  // U'y)
  svd.compute(x);

  // Confirm successful decomposition
  if (const int info{svd.info()}; info != Eigen::Success) {
    const std::string reason{(info == Eigen::NoConvergence)
                                 ? "Convergence failed"
                                 : "Numerical issue/Invalid input"};
    Rcpp::stop("SVD decomposition failed. Reason: " + reason +
               ". Check if the input matrix is extremely poorly scaled.");
  }

  return svd;
}

}  // namespace Grid::Utils
