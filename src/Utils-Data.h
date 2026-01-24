#ifndef CV_LM_UTILS_DATA_H
#define CV_LM_UTILS_DATA_H

#include <RcppEigen.h>

namespace Utils::Data {

// Assert the data is a column vector
template <typename Derived>
constexpr void assertColumnVector(const Eigen::MatrixBase<Derived>&) {
  static_assert(Derived::ColsAtCompileTime == 1, "Expected a column vector");
}

// Assert the data is a dynamic matrix
template <typename Derived>
constexpr void assertMatrix(const Eigen::MatrixBase<Derived>&) {
  static_assert(Derived::ColsAtCompileTime == Eigen::Dynamic,
                "Expected a dynamic matrix");
}

// Assert x is a matrix and y is a column vector
template <typename DerivedX, typename DerivedY>
constexpr void assertDataStructure(const Eigen::MatrixBase<DerivedX>& x,
                                   const Eigen::MatrixBase<DerivedY>& y) {
  assertMatrix(x);
  assertColumnVector(y);
}

// Center the response vector
template <typename Derived>
[[nodiscard]] Eigen::VectorXd centerResponse(
    const Eigen::MatrixBase<Derived>& y) {
  assertColumnVector(y);
  return Eigen::VectorXd{y.array() - y.mean()};
}

// Center the design matrix
template <typename Derived>
[[nodiscard]] Eigen::MatrixXd centerPredictors(
    const Eigen::MatrixBase<Derived>& x) {
  assertMatrix(x);
  return Eigen::MatrixXd{x.rowwise() - x.colwise().mean()};
}

// Center the design matrix and response vectors into pre-allocated buffers
template <typename DerivedX, typename DerivedY>
void centerData(Eigen::MatrixBase<DerivedX>& x, Eigen::MatrixBase<DerivedY>& y,
                Eigen::VectorXd& xColMeans, double& yMean) {
  // Make sure data is what we expect
  assertDataStructure(x, y);

  // Extract column means
  xColMeans = x.colwise().mean();
  yMean = y.mean();

  // Center the data
  x.rowwise() -= xColMeans.transpose();
  y.array() -= yMean;
}

}  // namespace Utils::Data

#endif  // CV_LM_UTILS_DATA_H
