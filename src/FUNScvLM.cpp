// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "FUNScvLM.h"

using namespace Rcpp;

// Calculate MSE
double cost(const Eigen::VectorXd& y, const Eigen::VectorXd& yhat) {
  double mse = (y - yhat).array().square().mean();
  return mse;
}

// Generate OLS coefficients
Eigen::VectorXd OLScoef(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const String& pivot, const bool& check) {
  int d = X.cols();
  Eigen::VectorXd W(d);
  if (pivot == "full") {
    Eigen::FullPivHouseholderQR<Eigen::MatrixXd> QR = X.fullPivHouseholderQr();
    if (check && (QR.rank() != d)) {
      stop("OLS coefficients not produced because X is not full column rank.");
    }
    W = QR.solve(y);
  }
  else if (pivot == "col") {
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> QR = X.colPivHouseholderQr();
    if (check && (QR.rank() != d)) {
      stop("OLS coefficients not produced because X is not full column rank.");
    }
    W = QR.solve(y);
  }
  else {
    Eigen::HouseholderQR<Eigen::MatrixXd> QR = X.householderQr();
    W = QR.solve(y);
  }
  return W;
}

// Generate Ridge regression coefficients
Eigen::VectorXd Ridgecoef(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const bool& pivot, const double& lambda) {
  Eigen::MatrixXd XT = X.transpose();
  Eigen::MatrixXd XTX = XT * X;
  XTX.diagonal().array() += lambda;
  Eigen::VectorXd XTy = XT * y;
  Eigen::VectorXd W(X.cols());
  if (pivot) {
    Eigen::LDLT<Eigen::MatrixXd> Cholesky(XTX);
    W = Cholesky.solve(XTy);
  }
  else {
    Eigen::LLT<Eigen::MatrixXd> Cholesky(XTX);
    W = Cholesky.solve(XTy);
  }
  return W;
}

// Extract elements of our features that are in-sample
Eigen::MatrixXd XinSample(const Eigen::MatrixXd& X, const Eigen::VectorXi& s, const int& i) {
  Eigen::VectorXi mask = (s.array() != (i + 1)).cast<int>();
  Eigen::MatrixXd XinS(mask.sum(), X.cols());
  int newRow = 0;
  for (int j = 0; j < X.rows(); ++j) {
    if (mask[j]) {
      XinS.row(newRow++) = X.row(j);
    }
  }
  return XinS;
}

// Extract elements of our target that are in-sample
Eigen::VectorXd yinSample(const Eigen::VectorXd& y, const Eigen::VectorXi& s, const int& i) {
  Eigen::VectorXi mask = (s.array() != (i + 1)).cast<int>();
  Eigen::VectorXd yinS(mask.sum());
  int newInd = 0;
  for (int j = 0; j < y.size(); ++j) {
    if (mask[j]) {
      yinS(newInd++) = y(j);
    }
  }
  return yinS;
}

// Extract elements of our features that are out-of-sample
Eigen::MatrixXd XoutSample(const Eigen::MatrixXd& X, const Eigen::VectorXi& s, const int& i) {
  Eigen::VectorXi mask = (s.array() == (i + 1)).cast<int>();
  Eigen::MatrixXd XoutS(mask.sum(), X.cols());
  int newRow = 0;
  for (int j = 0; j < X.rows(); ++j) {
    if (mask[j]) {
      XoutS.row(newRow++) = X.row(j);
    }
  }
  return XoutS;
}

// Extract elements of our target that are out-of-sample
Eigen::VectorXd youtSample(const Eigen::VectorXd& y, const Eigen::VectorXi& s, const int& i) {
  Eigen::VectorXi mask = (s.array() == (i + 1)).cast<int>();
  Eigen::VectorXd yout(mask.sum());
  int newIndex = 0;
  for (int j = 0; j < y.size(); ++j) {
    if (mask[j]) {
      yout(newIndex++) = y(j);
    }
  }
  return yout;
}

// Sampling assignment for CV
IntegerVector sampleCV(const IntegerVector& x, const int& size) {
  Function sample("sample");
  IntegerVector indices = sample(x, size);
  return indices;
}

// Setup partitions for CV
List cvSetup(const int& seed, const int& n, const int& K) {
  Function setSeed("set.seed");
  setSeed(seed);
  int f = ceil(static_cast<double>(n) / K);
  IntegerVector x = rep(seq(1, K), f);
  IntegerVector s = sampleCV(x, n);
  Eigen::VectorXi sEigen = as<Eigen::Map<Eigen::VectorXi>>(s);
  IntegerVector ns(K);
  for (int i = 0; i < K; ++i) {
    ns[i] = sum(s == (i + 1));
  }
  NumericVector nsDouble = as<NumericVector>(ns);
  int ms = max(s);
  return List::create(_["ms"] = ms, _["s"] = sEigen, _["ns"] = nsDouble);
}
