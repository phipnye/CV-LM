#pragma once

#include <RcppEigen.h>

namespace CV::Ridge {

// Generalized cross-validation for ridge regression
double gcv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
           const double lambda, const bool centered);

// Leave-one-out cross-validation for ridge regression
double loocv(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
             const double lambda, const bool centered);

// Multi-threaded CV for ridge regression
double parCV(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, const int k,
             const double lambda, const int seed, const int nThreads);

}  // namespace CV::Ridge
