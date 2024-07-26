// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "cvRidge.h"

#include <Rcpp.h>
#include <RcppEigen.h>

#include "FUNScvLM.h"

using namespace Rcpp;

DataFrame cvRidge(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K0, const double &lambda,
                  const bool &generalized, const int &seed, const int &nthreads) {
  double CV;
  int K;
  if (generalized) {
    K = K0;
    CV = gcvRidge(y, X, lambda);
  } else {
    int n = X.rows();
    K = Kcheck(n, K0);
    if (K == n) {
      CV = loocvRidge(y, X, lambda);
    } else if (nthreads > 1) {
      CV = parcvRidge(y, X, K, lambda, seed, nthreads);
    } else {
      CV = cvRidge(y, X, K, lambda, seed);
    }
  }
  return DataFrame::create(_["K"] = K, _["CV"] = CV, _["seed"] = seed);
}
