// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "FUNScvLM.h"
#include "CVWorker.h"

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
  double CV = 0.0;
  if (K == n) {
    Eigen::VectorXd yhat(n);
    Eigen::VectorXd diagH(n);
    Eigen::VectorXd W(n);
    Eigen::MatrixXd Q(n, d);
    if (pivot == "full") {
      Eigen::FullPivHouseholderQR<Eigen::MatrixXd> QR(X);
      if(rankCheck && QR.rank() != d) {
        stop("OLS coefficients not produced because X is not full column rank.");
      }
      W = QR.solve(y);
      Q = QR.matrixQ().leftCols(d);
      yhat = X * W;
      diagH = Q.rowwise().squaredNorm();
    }
    else if(pivot == "col") {
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> QR(X);
      if(rankCheck && QR.rank() != d) {
        stop("OLS coefficients not produced because X is not full column rank.");
      }
      W = QR.solve(y);
      Q = QR.householderQ() * Eigen::MatrixXd::Identity(n, d);
      yhat = X * W;
      diagH = Q.rowwise().squaredNorm();
    }
    else {
      Eigen::HouseholderQR<Eigen::MatrixXd> QR(X);
      W = QR.solve(y);
      Q = QR.householderQ() * Eigen::MatrixXd::Identity(n, d);
      yhat = X * W;
      diagH = Q.rowwise().squaredNorm();
    }
    for (int i = 0; i < n; ++i) {
      double error = y[i] - yhat[i];
      double leverage = 1 - diagH[i];
      CV += pow(error / leverage, 2);
    }
    CV /= n;
  }
  else {
    Function setSeed("set.seed");
    setSeed(seed);
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
      anyZero = (anyZero || (absDiff == 0));
    }
    if (!anyZero) {
      IntegerVector minKvals = Kvals[temp == min(temp)];
      K = minKvals[0];
    }
    if (K != K0) {
      Rcout << "Warning: K has been set to " << K << std::endl;
    }
    int f = ceil(static_cast<double>(n) / K);
    IntegerVector x = rep(seq(1, K), f);
    IntegerVector s = sampleCV(x, n);
    Eigen::VectorXi sEigen = as<Eigen::Map<Eigen::VectorXi>>(s);
    IntegerVector ns(K);
    for (int i = 0; i < K; ++i) {
      ns[i] = sum(s == (i + 1));
    }
    int ms = max(s);
    CVWorker cvWorker(y, X, sEigen, ns, n, pivot, rankCheck);
    RcppParallel::parallelReduce(0, ms, cvWorker);
    CV = cvWorker.MSE;
  }
  return List::create(_["K"] = K, _["CV"] = CV, _["seed"] = seed);
}
