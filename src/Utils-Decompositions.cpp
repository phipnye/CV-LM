#include <Rcpp.h>
#include <RcppEigen.h>

#include "Utils-Decompositions.h"

namespace Utils::Decompositions {

// Make sure decomposition info reports successful decomposition
void checkSvdInfo(const Eigen::ComputationInfo info) {
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

// Retrieve properly zeroed-out singular values
Eigen::VectorXd getSingularVals(const Eigen::BDCSVD<Eigen::MatrixXd>& udvT) {
  // Zero out the last m - r singular values which are below the object's
  // threshold, this minute difference can have large effects when dealing
  // with tiny division like 1e-30 / 1e-30 which evaluates to 1 instead of NaN
  // when the singular value should be 0 given the user's threshold
  const auto& origSingularVals{udvT.singularValues()};
  Eigen::VectorXd singularVals{Eigen::VectorXd::Zero(origSingularVals.size())};
  const Eigen::Index rank{udvT.rank()};
  singularVals.head(rank) = origSingularVals.head(rank);
  return singularVals;
}

// Retrieve properly zeroed-out singular values into a pre-allocated bu
void getSingularVals(const Eigen::BDCSVD<Eigen::MatrixXd>& udvT,
                     Eigen::VectorXd& singularVals) {
  const Eigen::Index rank{udvT.rank()};
  singularVals.setZero();
  singularVals.head(rank) = udvT.singularValues().head(rank);
}

}  // namespace Utils::Decompositions
