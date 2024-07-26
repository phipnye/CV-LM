#ifndef CV_LM_H
#define CV_LM_H

#include <Rcpp.h>
#include <RcppEigen.h>

using namespace Rcpp;

DataFrame cvLM(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K0, const bool &generalized,
               const int &seed, const int &nthreads);

#endif
