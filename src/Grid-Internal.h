#ifndef CV_LM_GRID_INTERNAL_H
#define CV_LM_GRID_INTERNAL_H

#include <RcppEigen.h>

#include "Grid-Deterministic-Worker.h"
#include "Grid-Deterministic-WorkerModel.h"
#include "Grid-LambdaCV.h"

namespace Grid::Deterministic::Internal {

template <typename Worker>
[[nodiscard]] LambdaCV executeWorker(Worker& worker,
                                     const Generator& lambdasGrid,
                                     const int nThreads) {
  // Parallelize over the grid of lambdas (this cast should be safe as, in the
  // ctor for the grid, we check that the grid size does not exceed the max
  // std::size_t value)
  Utils::Parallel::reduce(worker, static_cast<std::size_t>(lambdasGrid.size()),
                          nThreads);
  return worker.getOptimalPair();
}

// POD container for holding pre-computed values necessary across both
// deterministic methods
struct SVDPrecompute {
  const Eigen::VectorXd singularValsSq_;
  const Eigen::MatrixXd& u_;
  const Eigen::Index nrow_;
};

template <Enums::CenteringMethod Centering>
[[nodiscard]] LambdaCV runGCV(const Generator& lambdasGrid, const int nThreads,
                              const ResponseWrapper<Centering>& y,
                              const SVDPrecompute& pre) {
  // ||(I - UU')y||^2 = ||y||^2 - ||U'y||^2
  const Eigen::VectorXd uTySq{
      (pre.u_.transpose() * y.value()).array().square()};
  const double rssNull{y.value().squaredNorm() - uTySq.sum()};
  Worker worker{lambdasGrid,
                GCV::WorkerModel<Centering>{uTySq, pre.singularValsSq_, rssNull,
                                            pre.nrow_}};
  return executeWorker(worker, lambdasGrid, nThreads);
}

template <Enums::CenteringMethod Centering>
[[nodiscard]] LambdaCV runLOOCV(const Generator& lambdasGrid,
                                const int nThreads,
                                const ResponseWrapper<Centering>& y,
                                const SVDPrecompute& pre) {
  // U'y (the projection of y onto the left singular vectors of X)
  const Eigen::VectorXd uTy{pre.u_.transpose() * y.value()};

  // yNull is the projection of y onto the orthogonal complement of the column
  // space of X (the part of y not explained by the singular vectors)
  const Eigen::VectorXd yNull{y.value() - (pre.u_ * uTy)};
  const Eigen::MatrixXd uSq{pre.u_.array().square()};
  Worker worker{lambdasGrid,
                LOOCV::WorkerModel<Centering>{yNull, pre.u_, uSq, uTy,
                                              pre.singularValsSq_, pre.nrow_}};
  return executeWorker(worker, lambdasGrid, nThreads);
}

}  // namespace Grid::Deterministic::Internal

#endif  // CV_LM_GRID_INTERNAL_H
