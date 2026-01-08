#pragma once

#include <RcppEigen.h>

namespace Grid::Utils {

[[nodiscard]] Eigen::BDCSVD<Eigen::MatrixXd> svdDecompose(
    const Eigen::MatrixXd& x, double threshold);

}
