#ifndef CV_LM_UTILS_DATA_H
#define CV_LM_UTILS_DATA_H

#include <RcppArmadillo.h>

namespace Utils::Data {

// Confirm the object is an arma object
template <typename T>
constexpr void assertArma() {
  static_assert(arma::is_arma_type<T>::value,
                "Passed a non-arma object when one was expected.");
}

// Confirm the object is a generic matrix
template <typename T>
constexpr void assertMat() {
  assertArma<T>();
  static_assert(!T::is_col,
                "Passed a column vector when a generic matrix was expected.");
}

// Confirm the object is a column vector
template <typename T>
constexpr void assertVec() {
  assertArma<T>();
  static_assert(T::is_col,
                "Passed a generic matrix when a column vector was expected.");
}

// Center the response vector
template <typename T>
[[nodiscard]] arma::vec centerResponse(const T& y) {
  assertVec<T>();
  return y - arma::mean(y);
}

// Center the response vector and store the mean
template <typename T>
void centerResponse(const T& y, double& yMean) {
  assertVec<T>();
  yMean = arma::mean(y);
  y -= yMean;
}

// Center the response vector into another buffer and store the mean
template <typename T>
void centerResponse(const T& y, arma::vec& yCentered, double& yMean) {
  assertVec<T>();
  yMean = arma::mean(y);
  yCentered = y - yMean;
}

// Center the design matrix
template <typename T>
[[nodiscard]] arma::mat centerDesign(const T& X) {
  // Subtract column means from the original design matrix
  assertMat<T>();
  arma::mat centeredX{X};
  centeredX.each_row() -= arma::mean(X, 0);
  return centeredX;
}

// Center the design matrix and store the column means
template <typename T>
void centerDesign(T& X, arma::rowvec& XcolMeans) {
  // Extract column means
  assertMat<T>();
  XcolMeans = arma::mean(X, 0);

  // Center the data
  X.each_row() -= XcolMeans;
}

// Center the design matrix into another buffer and store the column means
template <typename T>
void centerDesign(const T& X, arma::mat& Xcentered, arma::rowvec& XcolMeans) {
  assertMat<T>();
  XcolMeans = arma::mean(X, 0);
  Xcentered = X;
  Xcentered.each_row() -= XcolMeans;
}

}  // namespace Utils::Data

#endif  // CV_LM_UTILS_DATA_H
