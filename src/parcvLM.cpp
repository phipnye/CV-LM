// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "FUNScvLM.h"
#include "cvLMWorker.h"

using namespace Rcpp;

// [[Rcpp::export(name = "par.cv.lm")]]
List parcvLM(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, int K, const int& seed, const String& pivot = "full", const bool& rankCheck = true) {
  int n = X.rows();
  int d = X.cols();
  if (n != y.size()) {
    stop("Non-conforming sizes for y & X.");
  }
  if ((K > n) || (K <= 1)) {
    stop("'K' outside allowable range");
  }
  if (pivot != "full" && pivot != "col" && pivot != "none") {
    stop("Invalid pivot argument. Must be either 'full', 'col', or 'none'.");
  }
  if (pivot == "none" && rankCheck) {
    stop("QR without pivoting is not a rank-revealing decomposition.");
  }
  
  int K0 = K;
  int holdKvalsLen = floor(n / 2);
  IntegerVector holdKvals(holdKvalsLen);
  for (int i = 0; i < holdKvalsLen; ++i) {
    holdKvals[i] = round(n / (i + 1));
  }
  IntegerVector Kvals = unique(holdKvals);
  IntegerVector temp(Kvals.size());
  bool anyZero = false;
  for (int i = 0; i < Kvals.size(); i++) {
    int absDiff = abs(Kvals[i] - K);
    temp[i] = absDiff;
    if (absDiff == 0) {
      anyZero = true;
    }
  }
  if (!anyZero) {
    IntegerVector minKvals = Kvals[temp == min(temp)];
    K = minKvals[0];
  }
  if (K != K0) {
    warning("The value for K has changed. See returned value for the value of K used.");
  }
  
  double CV = 0.0;
  if (K == n) {
    CV = loocvLM(n, d, pivot, X, y, rankCheck);
  }
  else {
    List Partitions = cvSetup(seed, n, K);
    int ms = Partitions["ms"];
    Eigen::VectorXi s = Partitions["s"];
    NumericVector ns = Partitions["ns"];
    cvLMWorker CVLMW(y, X, s, ns, n, pivot, rankCheck);
    RcppParallel::parallelReduce(0, ms, CVLMW);
    CV = CVLMW.MSE;
  }
  return List::create(_["K"] = K, _["CV"] = CV, _["seed"] = seed);
}
