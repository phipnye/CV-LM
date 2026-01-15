#include "Grid-engine.h"

#include <RcppEigen.h>

#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"
#include "Grid-Stochastic-Worker.h"
#include "Utils-Folds-utils.h"
#include "Utils-Parallel-utils.h"

namespace Grid::Stochastic {

// K fold CV
LambdaCV search(const Eigen::Map<Eigen::VectorXd>& y,
                const Eigen::Map<Eigen::MatrixXd>& x, const int k,
                const Generator& lambdasGrid, const int seed,
                const int nThreads, const double threshold) {
  // Setup folds
  const Eigen::Index nrow{x.rows()};
  const Utils::Folds::FoldInfo foldInfo{seed, static_cast<int>(nrow), k};

  // Initialize Worker
  Worker worker{y, x, foldInfo, lambdasGrid, threshold};

  // Compute cross-validation results (parallelize over folds)
  Utils::Parallel::reduce(worker, static_cast<std::size_t>(k), nThreads);
  return worker.getOptimalPair();
}

}  // namespace Grid::Stochastic
