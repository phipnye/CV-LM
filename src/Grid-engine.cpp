#include "Grid-engine.h"

#include <RcppEigen.h>

#include <utility>

#include "CV-Utils-utils.h"
#include "Grid-Deterministic-Worker.h"
#include "Grid-Deterministic-WorkerPolicy.h"
#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-Stochastic-Worker.h"

namespace Grid {

// Generalized CV
LambdaCV gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
             const Generator& lambdasGrid, const int nThreads,
             const bool centered) {
  // Perform SVD on full data once (for GCV, we only need singular values and
  // U'y)
  const Eigen::BDCSVD<Eigen::MatrixXd> svd{x, Eigen::ComputeThinU};
  const Eigen::ArrayXd eigenValsSq{svd.singularValues().array().square()};
  const Eigen::ArrayXd utySq{(svd.matrixU().transpose() * y).array().square()};

  // ||(I - UU')y||^2 = ||y||^2 - ||U'y||^2 (squared norm of the projection of y
  // onto the orthogonal complement of the column space of X)
  const double rssNull{y.squaredNorm() - utySq.sum()};

  // Initialize worker and policy
  using Deterministic::GCV::WorkerPolicy;
  const WorkerPolicy policy{utySq, rssNull};
  Deterministic::Worker<WorkerPolicy> worker{lambdasGrid, eigenValsSq, x.rows(),
                                             centered, policy};

  // Parallelize over the grid of Lambdas
  constexpr std::size_t begin{0};
  const std::size_t end{static_cast<std::size_t>(lambdasGrid.size())};
  constexpr std::size_t grainSize{1};
  RcppParallel::parallelReduce(begin, end, worker, grainSize, nThreads);
  return worker.results_;
}

// Leave-one-out CV
LambdaCV loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
               const Generator& lambdasGrid, const int nThreads,
               const bool centered) {
  // Perform SVD on full data once (need thin U for diagonal of Hat matrix)
  const Eigen::BDCSVD<Eigen::MatrixXd> svd{x, Eigen::ComputeThinU};
  const Eigen::MatrixXd& u{svd.matrixU()};

  // Pre-calculate U squared for faster diagonal computation in the policy
  const Eigen::MatrixXd uSq{u.array().square()};
  const Eigen::VectorXd uty{u.transpose() * y};

  // yNull is the projection of y onto the orthogonal complement of the column
  // space of X (the part of y not explained by the singular vectors)
  const Eigen::VectorXd yNull{y - (u * uty)};
  const Eigen::ArrayXd eigenValsSq{svd.singularValues().array().square()};

  // Initialize Policy and Worker (LOOCV Policy constructor handles its own
  // internal buffer allocation)
  using Deterministic::LOOCV::WorkerPolicy;
  WorkerPolicy policy{yNull, u, uSq, uty, x.rows(), eigenValsSq.size()};
  Deterministic::Worker<WorkerPolicy> worker{lambdasGrid, eigenValsSq, x.rows(),
                                             centered, std::move(policy)};

  // Parallelize over the grid of Lambdas
  constexpr std::size_t begin{0};
  const std::size_t end{static_cast<std::size_t>(lambdasGrid.size())};
  constexpr std::size_t grainSize{1};
  RcppParallel::parallelReduce(begin, end, worker, grainSize, nThreads);
  return worker.results_;
}

// K fold CV
LambdaCV kcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, const int k,
             const Generator& lambdasGrid, const int seed, const int nThreads) {
  // Setup folds
  const Eigen::Index nrow{x.rows()};
  const auto [foldIDs, foldSizes]{
      CV::Utils::setupFolds(seed, static_cast<int>(nrow), k)};

  // Pre-calculate fold size bounds (this allows us to allocate data buffers of
  // appropriate size in our worker instances)
  const auto [minTestSize, maxTestSize]{CV::Utils::testSizeExtrema(foldSizes)};
  const Eigen::Index maxTrainSize{nrow - minTestSize};

  // Initialize Worker
  Stochastic::Worker worker{
      y, x, foldIDs, foldSizes, lambdasGrid, maxTrainSize, maxTestSize};

  // Compute cross-validation results (parallelize over folds)
  constexpr std::size_t begin{0};
  const std::size_t end{static_cast<std::size_t>(k)};
  constexpr std::size_t grainSize{1};
  RcppParallel::parallelReduce(begin, end, worker, grainSize, nThreads);

  // Find the best lambda from the accumulated MSE vector
  Eigen::Index bestIdx;
  const double minMSE{worker.mses_.minCoeff(&bestIdx)};

  // Designated initializers not supported until C++20
  // return LambdaCV{.lambda{lambdasGrid[bestIdx]}, .cv{minMSE}};
  return LambdaCV{lambdasGrid[bestIdx], minMSE};
}

}  // namespace Grid
