// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "cvLM.h"

#include <Rcpp.h>
#include <RcppEigen.h>

#include "FUNScvLM.h"

using namespace Rcpp;

DataFrame cvLM(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K0, const bool &generalized,
               const int &seed, const int &nthreads) {
  double CV;
  int K;
  if (generalized) {
    K = K0;
    CV = gcvOLS(y, X);
  } else {
    int n = X.rows();
    K = Kcheck(n, K0);
    if (K == n) {
      CV = loocvOLS(y, X);
    } else if (nthreads > 1) {
      CV = parcvOLS(y, X, K, seed, nthreads);
    } else {
      CV = cvOLS(y, X, K, seed);
    }
  }
  return DataFrame::create(_["K"] = K, _["CV"] = CV, _["seed"] = seed);
}
