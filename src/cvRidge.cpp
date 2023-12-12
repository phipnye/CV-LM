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
  int f = ceil(static_cast<double>(n) / K);
  Function setSeed("set.seed");
  setSeed(seed);
  IntegerVector x = rep(seq(1, K), f);
  IntegerVector s = sampleCV(x, n);
  Eigen::VectorXi sEigen = as<Eigen::Map<Eigen::VectorXi>>(s);
  IntegerVector ns(K);
  for (int i = 0; i < K; ++i) {
    ns[i] = sum(s == (i + 1));
  }
  int ms = max(s);
  double CV = 0.0;
  for (int i = 0; i < ms; ++i) {
    Eigen::MatrixXd XinS = XinSample(X, sEigen, i);
    Eigen::VectorXd yinS = yinSample(y, sEigen, i);
    Eigen::MatrixXd XoutS = XoutSample(X, sEigen, i);
    Eigen::VectorXd youtS = youtSample(y, sEigen, i);
    Eigen::VectorXd W = Ridgecoef(XinS, yinS, pivot, lambda);
    Eigen::VectorXd yhat = XoutS * W;
    double costI = cost(youtS, yhat);
    double alphaI = static_cast<double>(ns[i]) / n;
    CV += alphaI * costI;
  }
  return List::create(_["K"] = K, _["CV"] = CV, _["seed"] = seed);
}