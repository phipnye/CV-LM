#pragma once

#include <RcppEigen.h>

namespace CV::Ridge {

// Generalized cross-validation for ridge regression
double gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, double lambda);

// Leave-one-out cross-validation for ridge regression
double loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, double lambda);

// Multi-threaded CV for ridge regression
double parCV(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, int k,
             double lambda, int seed, int nThreads);

}  // namespace CV::Ridge
