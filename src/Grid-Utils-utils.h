#pragma once

#include <RcppEigen.h>

namespace Grid::Utils {

// Check whether SVD was successful
void checkSvdStatus(Eigen::ComputationInfo info);

[[nodiscard]] Eigen::BDCSVD<Eigen::MatrixXd> svdDecompose(
    const Eigen::MatrixXd& x, double threshold);

}  // namespace Grid::Utils
