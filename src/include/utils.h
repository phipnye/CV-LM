#pragma once

#include <RcppEigen.h>

#include <utility>

namespace CV::Utils {

// Confirm valid value for K
int kCheck(int n, int k0);

// Calculate MSE
double cost(const Eigen::VectorXd& y, const Eigen::VectorXd& yHat);

// Generates fold assignments
std::pair<Eigen::VectorXi, Eigen::VectorXd> cvSetup(int seed, int n, int k);

}  // namespace CV::Utils
