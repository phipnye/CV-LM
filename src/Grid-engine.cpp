#include "Grid-engine.h"

#include <RcppEigen.h>

#include <utility>

#include "CV-Utils-utils.h"
#include "Constants.h"
#include "Grid-Deterministic-Worker.h"
#include "Grid-Deterministic-WorkerPolicy.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-Stochastic-Worker.h"
#include "Grid-Utils-utils.h"

namespace Grid {

// Generalized CV
LambdaCV gcv(const Eigen::Map<Eigen::VectorXd>& y,
             const Eigen::Map<Eigen::MatrixXd>& x, const Generator& lambdasGrid,
             const int nThreads, const double threshold, const bool centered) {
  // Decompose X with thin U computation
  const Eigen::BDCSVD<Eigen::MatrixXd> svd{Utils::svdDecompose(x, threshold)};

  // Pre-compute data for closed-form solution
  const Eigen::ArrayXd eigenValsSq{svd.singularValues().array().square()};
  const Eigen::ArrayXd utySq{(svd.matrixU().transpose() * y).array().square()};

  // ||(I - UU')y||^2 = ||y||^2 - ||U'y||^2 (squared norm of the projection of y
  // onto the orthogonal complement of the column space of X)
  const double rssNull{y.squaredNorm() - utySq.sum()};

  // Initialize worker
  using Deterministic::GCV::WorkerPolicy;
  Deterministic::Worker<WorkerPolicy> worker{lambdasGrid, eigenValsSq, x.rows(),
                                             centered,
                                             WorkerPolicy{utySq, rssNull}};

  // Parallelize over the grid of Lambdas
  const std::size_t end{static_cast<std::size_t>(lambdasGrid.size())};
  RcppParallel::parallelReduce(Constants::begin, end, worker,
                               Constants::grainSize, nThreads);
  return worker.getOptimalPair();
}

// Leave-one-out CV
LambdaCV loocv(const Eigen::Map<Eigen::VectorXd>& y,
               const Eigen::Map<Eigen::MatrixXd>& x,
               const Generator& lambdasGrid, const int nThreads,
               const double threshold, const bool centered) {
  // Decompose X with thin U computation
  const Eigen::BDCSVD<Eigen::MatrixXd> svd{Utils::svdDecompose(x, threshold)};

  // Pre-calculate U squared for faster diagonal computation in the policy
  const Eigen::MatrixXd& u{svd.matrixU()};
  const Eigen::MatrixXd uSq{u.array().square()};
  const Eigen::ArrayXd uty{(u.transpose() * y).array()};

  // yNull is the projection of y onto the orthogonal complement of the column
  // space of X (the part of y not explained by the singular vectors)
  const Eigen::VectorXd yNull{y - (u * uty.matrix())};
  const Eigen::ArrayXd eigenValsSq{svd.singularValues().array().square()};

  // Initialize worker (LOOCV Policy constructor handles its own internal buffer
  // allocation)
  using Deterministic::LOOCV::WorkerPolicy;
  Deterministic::Worker<WorkerPolicy> worker{
      lambdasGrid, eigenValsSq, x.rows(), centered,
      WorkerPolicy{yNull, u, uSq, uty, x.rows(), eigenValsSq.size()}};

  // Parallelize over the grid of Lambdas
  const std::size_t end{static_cast<std::size_t>(lambdasGrid.size())};
  RcppParallel::parallelReduce(Constants::begin, end, worker,
                               Constants::grainSize, nThreads);
  return worker.getOptimalPair();
}

// K fold CV
LambdaCV kcv(const Eigen::Map<Eigen::VectorXd>& y,
             const Eigen::Map<Eigen::MatrixXd>& x, const int k,
             const Generator& lambdasGrid, const int seed, const int nThreads,
             const double threshold) {
  // Setup folds
  const Eigen::Index nrow{x.rows()};
  const auto [testFoldIDs, testFoldSizes]{
      CV::Utils::setupFolds(seed, static_cast<int>(nrow), k)};

  // Pre-calculate fold size bounds (this allows us to allocate data buffers of
  // appropriate size in our worker instances)
  const auto [minTestSize,
              maxTestSize]{CV::Utils::testSizeExtrema(testFoldSizes)};
  const Eigen::Index maxTrainSize{nrow - minTestSize};

  // Initialize Worker
  Stochastic::Worker worker{y,           x,
                            testFoldIDs, testFoldSizes,
                            lambdasGrid, maxTrainSize,
                            maxTestSize, threshold};

  // Compute cross-validation results (parallelize over folds)
  const std::size_t end{static_cast<std::size_t>(k)};
  RcppParallel::parallelReduce(Constants::begin, end, worker,
                               Constants::grainSize, nThreads);
  return worker.getOptimalPair();
}

}  // namespace Grid
