// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <Rcpp.h>
#include <RcppEigen.h>

#include "cvLM.h"
#include "cvRidge.h"

using namespace Rcpp;

// [[Rcpp::export(name="cv.lm.rcpp")]]
DataFrame cvLMrcpp(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K0, const double &lambda,
                   const bool &generalized, const int &seed, const int &nthreads) {
  DataFrame cvRes;
  if (lambda == 0) {
    cvRes = cvLM(y, X, K0, generalized, seed, nthreads);
  } else {
    cvRes = cvRidge(y, X, K0, lambda, generalized, seed, nthreads);
  }
  return cvRes;
}
