#ifndef CV_LM_GRID_H
#define CV_LM_GRID_H

#include <RcppArmadillo.h>

#include <utility>

#include "DataLoader.h"
#include "Enums.h"
#include "Grid-Deterministic-Worker.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-Stochastic-Worker.h"
#include "SingularValueDecomposition.h"
#include "Utils-Parallel.h"

namespace Grid {

namespace Deterministic {

// Generalized and leave-one-out grid search
template <Enums::CrossValidationMethod CV, Enums::CenteringMethod Centering>
[[nodiscard]] LambdaCV search(const arma::mat& X, const arma::vec& y,
                              const Generator& lambdasGrid, const int nThreads,
                              const double tolerance) {
  // Make sure we have a deterministic type
  if constexpr (CV != Enums::CrossValidationMethod::GCV) {
    Enums::assertExpected<CV, Enums::CrossValidationMethod::LOOCV>();
  }

  // Singular value decomposition allows us to decompose the design matrix once
  // allowing for fast parameter searches and is very numerically stable
  // regardless of ill-conditionedness or rank-deficiency
  SingularValueDecomposition<CV, Centering> svd{tolerance};

  if (!svd.setDesign(X)) {
    Rcpp::stop("Singular value decomposition of the design matrix failed.");
  }

  svd.setResponse(y);  // always returns true (no need to check)
  Worker worker{std::move(svd), lambdasGrid};
  Utils::Parallel::reduce(worker, lambdasGrid.size(), nThreads);
  return worker.getOptimalPair();
}

}  // namespace Deterministic

namespace Stochastic {

// K-fold grid search
template <Enums::CenteringMethod Centering>
[[nodiscard]] LambdaCV search(const arma::mat& X, const arma::vec& y,
                              const arma::uword k, const Generator& lambdasGrid,
                              const int seed, const int nThreads,
                              const double tolerance) {
  // Setup data loader (handles shuffling and fold indexing)
  const DataLoader loader{X, y, seed, k};

  // Initialize worker
  using SVD =
      SingularValueDecomposition<Enums::CrossValidationMethod::KCV, Centering>;
  Worker worker{SVD{tolerance}, loader, lambdasGrid};

  // Compute cross-validation results (parallelize over folds)
  Utils::Parallel::reduce(worker, k, nThreads);
  return worker.getOptimalPair();
}

}  // namespace Stochastic

}  // namespace Grid

#endif  // CV_LM_GRID_H
