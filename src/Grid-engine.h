#pragma once

#include <RcppEigen.h>

#include "Enums.h"
#include "Grid-Deterministic-Worker.h"
#include "Grid-Deterministic-WorkerModel.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-Stochastic-Worker.h"
#include "ResponseWrapper.h"
#include "Utils-Data.h"
#include "Utils-Decompositions.h"
#include "Utils-Folds.h"
#include "Utils-Parallel.h"

namespace Grid {

namespace Deterministic {

template <typename Worker>
[[nodiscard]] LambdaCV executeWorker(Worker& worker,
                                     const Generator& lambdasGrid,
                                     const int nThreads) {
  // Parallelize over the grid of lambdas (this cast should be safe - in the
  // ctor for the grid, we check that the grid size does not exceed the max
  // std::size_t value)
  Utils::Parallel::reduce(worker, static_cast<std::size_t>(lambdasGrid.size()),
                          nThreads);
  return worker.getOptimalPair();
}

template <Enums::AnalyticMethod Analytic, Enums::CenteringMethod Centering>
[[nodiscard]] LambdaCV search(const Eigen::Map<Eigen::VectorXd>& yOrig,
                              const Eigen::Map<Eigen::MatrixXd>& xOrig,
                              const Generator& lambdasGrid, const int nThreads,
                              const double threshold) {
  // Decompose X = UDV' with thin U computation
  const Eigen::BDCSVD<Eigen::MatrixXd> udvT{
      Utils::Decompositions::svd<Centering>(xOrig, Eigen::ComputeThinU,
                                            threshold)};

  // Center the response data if necessary
  ResponseWrapper<Centering> y{yOrig};

  // Compute the squared singular values (corrected singular values below
  // threshold criteria to zero)
  const Eigen::VectorXd singularValsSq{
      Utils::Decompositions::getSingularVals(udvT).array().square()};

  // Compute the projection of y onto the left singular vectors of X
  const Eigen::MatrixXd& u{udvT.matrixU()};
  const Eigen::VectorXd uTy{u.transpose() * y.get()};
  const Eigen::Index nrow{xOrig.rows()};

  // Pre-compute data for closed-form solutions and initialize a worker instance
  if constexpr (Analytic == Enums::AnalyticMethod::GCV) {
    using WorkerModel = GCV::WorkerModel<Centering>;
    const Eigen::VectorXd uTySq{uTy.array().square()};

    // ||(I - UU')y||^2 = ||y||^2 - ||U'y||^2 (squared norm of the projection of
    // y onto the orthogonal complement of the column space of X)
    const double rssNull{y.get().squaredNorm() - uTySq.sum()};

    // Initialize worker
    Worker worker{lambdasGrid,
                  WorkerModel{uTySq, singularValsSq, rssNull, nrow}};
    return executeWorker(worker, lambdasGrid, nThreads);
  } else {
    Enums::assertExpected<Analytic, Enums::AnalyticMethod::LOOCV>();
    using WorkerModel = LOOCV::WorkerModel<Centering>;
    const Eigen::MatrixXd uSq{u.array().square()};

    // yNull is the projection of y onto the orthogonal complement of the column
    // space of X (the part of y not explained by the singular vectors)
    const Eigen::VectorXd yNull{y.get() - (u * uTy)};

    // Initialize worker
    Worker worker{lambdasGrid,
                  WorkerModel{yNull, u, uSq, uTy, singularValsSq, nrow}};
    return executeWorker(worker, lambdasGrid, nThreads);
  }
}

}  // namespace Deterministic

namespace Stochastic {

// K-fold CV
template <Enums::CenteringMethod Centering>
[[nodiscard]] LambdaCV search(const Eigen::Map<Eigen::VectorXd>& y,
                              const Eigen::Map<Eigen::MatrixXd>& x, const int k,
                              const Generator& lambdasGrid, const int seed,
                              const int nThreads, const double threshold) {
  // Setup folds
  const Eigen::Index nrow{x.rows()};
  const Utils::Folds::FoldInfo foldInfo{seed, static_cast<int>(nrow), k};

  // Initialize Worker
  Worker<Centering> worker{y, x, foldInfo, lambdasGrid, threshold};

  // Compute cross-validation results (parallelize over folds)
  Utils::Parallel::reduce(worker, static_cast<std::size_t>(k), nThreads);
  return worker.getOptimalPair();
}

}  // namespace Stochastic

}  // namespace Grid
