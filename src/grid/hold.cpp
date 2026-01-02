// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <limits>

namespace Grid {

struct GridResult {
  double bestLambda;
  double minCV;
};

GridResult gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
               const double maxLambda, const double precision) {
  // Compute the (thin) SVD of X, we need U (n x rank(X)) which has orthonormal
  // columns
  const Eigen::BDCSVD<Eigen::MatrixXd> svd{x, Eigen::ComputeThinU};

  // Squared eigen values of X are the eigenvalues of X'X and determine how
  // ridge shrinks each principal direction (the directions in the data space
  // along which the variance is maximized)
  const auto eigenValsSq{svd.singularValues().array().square().eval()};
  const auto& u{svd.matrixU()};
  const auto uty{u.transpose() * y};
  const auto utySq{uty.array().square().eval()};
  const Eigen::Index nrow{x.rows()};

  // ||(I - UU')y||^2 = ||y||^2 - ||U'y||^2 (squared norm of the projection of y
  // onto the orthogonal complement of the column space of X)
  const double rssNull{y.squaredNorm() - uty.squaredNorm()};

  // Variables to hold results
  double minCV{std::numeric_limits<double>::infinity()};
  double bestLambda{0.0};

  for (double lambda{0.0}; lambda <= maxLambda; lambda += precision) {
    // Calculate trace(H) = sum(eigenVals^2 / (eigenVals^2 + lambda))
    const double traceH{(eigenValsSq / (eigenValsSq + lambda)).sum()};

    // RSS = sum_{i=1}^{r}((lambda^2 + (eigenVal_i^2 + lambda)^2) * u_i'y^2) +
    // ||(I - UU')y||^2
    const double rss{
        rssNull +
        ((lambda * lambda) * utySq / (eigenValsSq + lambda).square()).sum()};

    // GCV = MSE / (1 - trance(H) / n)^2
    const double leverage{1.0 - (traceH / nrow)};
    const double gcvVal{rss / (nrow * leverage * leverage)};

    if (gcvVal < minCV) {
      bestLambda = lambda;
      minCV = gcvVal;
    }
  }

  return GridResult{.bestLambda{bestLambda}, .minCV{minCV}};
}

}  // namespace Grid
