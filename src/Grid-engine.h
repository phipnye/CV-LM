#pragma once

#include <RcppEigen.h>
#include <RcppParallel.h>

#include "Enums-enums.h"
#include "Grid-Deterministic-Worker.h"
#include "Grid-Deterministic-WorkerPolicy.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Utils-Decompositions-utils.h"
#include "Utils-Parallel-utils.h"

namespace Grid {

namespace Deterministic {

template <typename Worker>
[[nodiscard]] LambdaCV executeWorker(Worker& worker,
                                     const Generator& lambdasGrid,
                                     const int nThreads) {
  // Parallelize over the grid of Lambdas (this cast should be safe - in the
  // ctor, we check that the grid size does not exceed the max std::size_t
  // value)
  Utils::Parallel::reduce(worker, static_cast<std::size_t>(lambdasGrid.size()),
                          nThreads);
  return worker.getOptimalPair();
}

template <Enums::AnalyticMethod CVMethod>
[[nodiscard]] LambdaCV search(const Eigen::Map<Eigen::VectorXd>& y,
                              const Eigen::Map<Eigen::MatrixXd>& x,
                              const Generator& lambdasGrid, const int nThreads,
                              const double threshold, const bool centered) {
  // Decompose X = UDV' with thin U computation
  constexpr bool checkSuccess{true};
  const Eigen::BDCSVD udvT{Utils::Decompositions::svd<checkSuccess>(
      x, Eigen::ComputeThinU, threshold)};

  // Compute the squared singular values (corrected singular values below
  // threshold criteria to zero)
  const Eigen::VectorXd singularValsSq{
      Utils::Decompositions::getSingularVals(udvT).array().square()};

  // Pre-compute data for closed-form solutions and initialize a worker instance
  if constexpr (CVMethod == Enums::AnalyticMethod::GCV) {
    // Projection of y onto the column space of X squared
    const Eigen::VectorXd uTySq{
        (udvT.matrixU().transpose() * y).array().square()};

    // ||(I - UU')y||^2 = ||y||^2 - ||U'y||^2 (squared norm of the projection of
    // y onto the orthogonal complement of the column space of X)
    const double rssNull{y.squaredNorm() - uTySq.sum()};

    // Initialize worker
    Worker worker{lambdasGrid, GCV::WorkerPolicy{uTySq, singularValsSq, rssNull,
                                                 x.rows(), centered}};
    return executeWorker(worker, lambdasGrid, nThreads);
  } else {
    Enums::assertExpected<CVMethod, Enums::AnalyticMethod::LOOCV>();
    const Eigen::MatrixXd& u{udvT.matrixU()};
    const Eigen::MatrixXd uSq{u.array().square()};

    // Projection of y onto the column space of X
    const Eigen::VectorXd uTy{(u.transpose() * y)};

    // yNull is the projection of y onto the orthogonal complement of the column
    // space of X (the part of y not explained by the singular vectors)
    const Eigen::VectorXd yNull{y - (u * uTy)};

    // Initialize worker
    Worker worker{lambdasGrid,
                  LOOCV::WorkerPolicy{yNull, u, uSq, uTy, singularValsSq,
                                      x.rows(), centered}};
    return executeWorker(worker, lambdasGrid, nThreads);
  }
}

}  // namespace Deterministic

namespace Stochastic {

// K-fold CV
[[nodiscard]] LambdaCV search(const Eigen::Map<Eigen::VectorXd>& y,
                              const Eigen::Map<Eigen::MatrixXd>& x, int k,
                              const Generator& lambdasGrid, int seed,
                              int nThreads, double threshold);

}  // namespace Stochastic

}  // namespace Grid
