#ifndef CV_LM_UTILS_DECOMPOSITIONS_H
#define CV_LM_UTILS_DECOMPOSITIONS_H

#include <RcppEigen.h>

#include "Enums.h"
#include "Utils-Data.h"

namespace Utils::Decompositions {

// Check whether singular value decomposition was successful
void checkSvdInfo(Eigen::ComputationInfo info);

// Retrieve properly zeroed-out singular values
[[nodiscard]] Eigen::VectorXd getSingularVals(
    const Eigen::BDCSVD<Eigen::MatrixXd>& udvT);

// Retrieve properly zeroed-out singular values into a pre-allocated buffer
template <typename Derived>
void getSingularVals(const Eigen::BDCSVD<Eigen::MatrixXd>& udvT,
                     Eigen::MatrixBase<Derived>& singularVals) {
  Data::assertColumnVector(singularVals);
  const Eigen::Index rank{udvT.rank()};
  singularVals.setZero();
  singularVals.head(rank) = udvT.singularValues().head(rank);
}

// Perform singular value decomposition of X
template <Enums::CenteringMethod Centering, bool checkSucccess = true,
          typename Derived>
[[nodiscard]] Eigen::BDCSVD<Eigen::MatrixXd> svd(
    const Eigen::MatrixBase<Derived>& x, const unsigned int computationOptions,
    const double threshold) {
  // Make sure we passed a matrix-like object
  Data::assertMatrix(x);

  // Decompose X = UDV'
  using SVD = Eigen::BDCSVD<Eigen::MatrixXd>;
  auto udvT{[&]() -> SVD {
    if constexpr (Centering == Enums::CenteringMethod::Mean) {
      return SVD{Data::centerPredictors(x), computationOptions};
    } else {
      Enums::assertExpected<Centering, Enums::CenteringMethod::None>();
      return SVD{x, computationOptions};
    }
  }()};

  // Set threshold at which singular values are considered zero "A singular
  // value will be considered nonzero if its value is strictly greater than
  // |singularvalue|⩽threshold×|maxsingularvalue|." - this threshold only
  // affects other methods like solve and rank, not the decomposition itself
  udvT.setThreshold(threshold);

  // Confirm successful decomposition (we do not want to call this from a
  // multi-threaded context since Rcpp::stop may be called)
  if constexpr (checkSucccess) {
    checkSvdInfo(udvT.info());
  }

  return udvT;
}

// Perform complete orthogonal decomposition of X
template <Enums::CenteringMethod Centering, typename Derived>
[[nodiscard]] Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(
    const Eigen::MatrixBase<Derived>& x, const double threshold) {
  // Make sure we passed a matrix-like object
  Data::assertMatrix(x);

  // Decompose XP = QR = QTZ (there's no need to check the success of this
  // decomposition - the documentation states the info method always returns
  // success)
  using COD = Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>;
  auto qtz{[&]() -> COD {
    if constexpr (Centering == Enums::CenteringMethod::Mean) {
      return COD{Data::centerPredictors(x)};
    } else {
      Enums::assertExpected<Centering, Enums::CenteringMethod::None>();
      return COD{x};
    }
  }()};

  // Set threshold at which pivotes are to be considered zero "A pivot will be
  // considered nonzero if its absolute value is strictly greater than
  // |pivot|⩽threshold×|maxpivot| where maxpivot is the biggest pivot."
  qtz.setThreshold(threshold);
  return qtz;
}

}  // namespace Utils::Decompositions

#endif  // CV_LM_UTILS_DECOMPOSITIONS_H
