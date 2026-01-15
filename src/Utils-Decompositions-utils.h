#pragma once

#include <RcppEigen.h>

namespace Utils::Decompositions {

// Check whether singular value decomposition was successful
void checkSvdInfo(Eigen::ComputationInfo info);

// Retrieve properly zeroed-out singular values
[[nodiscard]] Eigen::VectorXd getSingularVals(
    const Eigen::BDCSVD<Eigen::MatrixXd>& udvT);

// Perform singular value decomposition of X
template <bool checkSuccess, typename Derived>
[[nodiscard]] Eigen::BDCSVD<Eigen::MatrixXd> svd(
    const Eigen::MatrixBase<Derived>& x, const unsigned int computationOptions,
    const double threshold) {
  // Decompose X = UDV'
  Eigen::BDCSVD<Eigen::MatrixXd> udv{x, computationOptions};

  // Set threshold at which singular values are considered zero "A singular
  // value will be considered nonzero if its value is strictly greater than
  // |singularvalue|⩽threshold×|maxsingularvalue|." - this threshold only
  // affects other methods like solve and rank, not the decomposition itself
  udv.setThreshold(threshold);

  // Confirm successful decomposition
  if constexpr (checkSuccess) {  // We don't want to call this from a
                                 // multi-threaded context since Rcpp::stop may
                                 // be called
    checkSvdInfo(udv.info());
  }

  return udv;
}

// Perform complete orthogonal decomposition of X
template <typename Derived>
[[nodiscard]] Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(
    const Eigen::MatrixBase<Derived>& x, const double threshold) {
  // Decompose XP = QR = QTZ (there's no need to check the success of this
  // decomposition - the documentation states the info method always returns
  // success)
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> qtz{x};

  // Set threshold at which pivotes are to be considered zero "A pivot will be
  // considered nonzero if its absolute value is strictly greater than
  // |pivot|⩽threshold×|maxpivot| where maxpivot is the biggest pivot."
  qtz.setThreshold(threshold);
  return qtz;
}

}  // namespace Utils::Decompositions
