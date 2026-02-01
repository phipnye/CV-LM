#ifndef CV_LM_CV_H
#define CV_LM_CV_H

#include <RcppArmadillo.h>

#include <cstddef>

#include "CV-Stochastic-Worker.h"
#include "CompleteOrthogonalDecomposition.h"
#include "DataLoader.h"
#include "Enums.h"
#include "SingularValueDecomposition.h"
#include "Utils-Decompositions.h"
#include "Utils-Parallel.h"

namespace CV {

namespace Deterministic {

// Generalized and leave-one-out cross-validation for linear and ridge
// regression
template <Enums::CrossValidationMethod CV, Enums::CenteringMethod Centering>
[[nodiscard]] double computeCV(const arma::mat& X, const arma::vec& y,
                               const double tolerance, const double lambda) {
  // Make sure we have a deterministic type
  if constexpr (CV != Enums::CrossValidationMethod::GCV) {
    Enums::assertExpected<CV, Enums::CrossValidationMethod::LOOCV>();
  }

  // With OLS, use Complete Orthogonal Decomposition
  if (lambda <= 0.0) {
    CompleteOrthogonalDecomposition<CV, Centering> cod{tolerance};

    if (!Utils::Decompositions::setParams(cod, X, y)) {
      Rcpp::stop(
          "Complete orthogonal decomposition of the design matrix failed.");
    }

    return cod.cv();
  }

  // For ridge regression use Singular Value Decomposition
  SingularValueDecomposition<CV, Centering> svd{tolerance};

  if (!Utils::Decompositions::setParams(svd, X, y, lambda)) {
    Rcpp::stop("Singular value decomposition of the design matrix failed.");
  }

  return svd.cv();
}

}  // namespace Deterministic

namespace Stochastic {

// Multi-threaded K-fold CV for linear and ridge regression
template <Enums::CenteringMethod Centering>
[[nodiscard]] double computeCV(const arma::mat& X, const arma::vec& y,
                               const arma::uword k, const int seed,
                               const int nThreads, const double tolerance,
                               const double lambda) {
  // Setup data loader (handles shuffling and fold indexing)
  const DataLoader loader{X, y, seed, k};

  // With OLS, use Complete Orthogonal Decomposition
  if (lambda <= 0.0) {
    using COD =
        CompleteOrthogonalDecomposition<Enums::CrossValidationMethod::KCV,
                                        Centering>;
    Worker worker{COD{tolerance}, loader};
    Utils::Parallel::reduce(worker, static_cast<std::size_t>(k), nThreads);
    return worker.getCV();
  }

  // For ridge regression, use Singular Value Decomposition
  using SVD =
      SingularValueDecomposition<Enums::CrossValidationMethod::KCV, Centering>;
  Worker worker{SVD{tolerance}, loader, lambda};
  Utils::Parallel::reduce(worker, static_cast<std::size_t>(k), nThreads);
  return worker.getCV();
}

}  // namespace Stochastic

}  // namespace CV

#endif  // CV_LM_CV_H
