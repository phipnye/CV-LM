#pragma once

#include <RcppEigen.h>

namespace Grid::Utils {

// Check whether SVD was successful
void checkSvdStatus(Eigen::ComputationInfo info);

// Perform singular value decomposition of X and compute thin U
[[nodiscard]] Eigen::BDCSVD<Eigen::MatrixXd> svdDecompose(
    const Eigen::Map<Eigen::MatrixXd>& x, double threshold);

}  // namespace Grid::Utils
