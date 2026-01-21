#pragma once

#include <RcppEigen.h>

namespace Utils::Data {

// Assert the data is a column vector
template <typename Derived>
void assertColumnVector(const Eigen::MatrixBase<Derived>&) {
  static_assert(Derived::ColsAtCompileTime == 1,
                "Expected y to be a column vector");
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
  return Eigen::MatrixXd{x.rowwise() - x.colwise().mean()};
}

// Center the design matrix and response vectors into pre-allocated buffers
template <typename DerivedX, typename DerivedY>
void centerData(const Eigen::MatrixBase<DerivedX>& x,
                const Eigen::MatrixBase<DerivedY>& y,
                Eigen::MatrixXd& xCentered, Eigen::VectorXd& yCentered,
                Eigen::VectorXd& xColMeans, double& yMean) {
  // Extract column means
  xColMeans = x.colwise().mean();
  assertColumnVector(y);
  yMean = y.mean();

  // Center the data
  const Eigen::Index trainSize{x.rows()};
  xCentered.topRows(trainSize) = x.rowwise() - xColMeans.transpose();
  yCentered.head(trainSize) = y.array() - yMean;
}

}  // namespace Utils::Data
