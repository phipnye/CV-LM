#pragma once

#include <RcppEigen.h>

#include "Grid-Generator.h"
#include "Grid-LambdaCV.h"

namespace Grid {

// Generalized CV
[[nodiscard]] LambdaCV gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                           const Generator& lambdasGrid, int nThreads,
                           double threshold, bool centered);

// Leave-one-out CV
[[nodiscard]] LambdaCV loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                             const Generator& lambdasGrid, int nThreads,
                             double threshold, bool centered);

// K fold CV
[[nodiscard]] LambdaCV kcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
                           int k, const Generator& lambdasGrid, int seed,
                           int nThreads, double threshold);

}  // namespace Grid
