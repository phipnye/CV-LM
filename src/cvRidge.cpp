// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "cvLMExport.h"
#include "FUNScvLM.h"

using namespace Rcpp;

// [[Rcpp::export(name="cv.ridge")]]
List cvRidge(const Eigen::VectorXd& y, const Eigen::MatrixXd& X, const double& lambda, int K, const int& seed, const bool& pivot = true) {
  int n = X.rows();
  int d = X.cols();
  if (n != y.size()) {
    stop("Non-conforming sizes for y & X.");
  }
  if ((K > n) || (K <= 1)) {
    stop("'K' outside allowable range");
  }
  if (lambda < 0.0) {
    stop("'lambda' must be non-negative");
  }
  if (lambda == 0.0) {
    List cvRes = cvLM(y, X, K, seed, "col", true);
    return cvRes;
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
    CV = loocvRidge(n, d, pivot, X, y, lambda);
  }
  else {
    List Partitions = cvSetup(seed, n, K);
    int ms = Partitions["ms"];
    Eigen::VectorXi s = Partitions["s"];
    NumericVector ns = Partitions["ns"];
    for (int i = 0; i < ms; ++i) {
      Eigen::MatrixXd XinS = XinSample(X, s, i);
      Eigen::VectorXd yinS = yinSample(y, s, i);
      Eigen::MatrixXd XoutS = XoutSample(X, s, i);
      Eigen::VectorXd youtS = youtSample(y, s, i);
      Eigen::VectorXd W = Ridgecoef(XinS, yinS, pivot, lambda);
      Eigen::VectorXd yhat = XoutS * W;
      double costI = cost(youtS, yhat);
      double alphaI = ns[i] / n;
      CV += (alphaI * costI);
    }
  }
  return List::create(_["K"] = K, _["CV"] = CV, _["seed"] = seed);
}