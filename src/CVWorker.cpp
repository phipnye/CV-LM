// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "FUNScvLM.h"
#include "CVWorker.h"


CVWorker::CVWorker(const Eigen::VectorXd& y_, const Eigen::MatrixXd& X_, const Eigen::VectorXi& s_, const IntegerVector& ns_, const int& n_, const String& pivot_, const bool& rankCheck_)
  : y(y_), X(X_), s(s_), ns(ns_), n(n_), pivot(pivot_), rankCheck(rankCheck_), MSE(0.0) {}
CVWorker::CVWorker(const CVWorker& CVw, RcppParallel::Split)
  : y(CVw.y), X(CVw.X), s(CVw.s), ns(CVw.ns), n(CVw.n), pivot(CVw.pivot), rankCheck(CVw.rankCheck), MSE(0.0) {}

void CVWorker::operator()(std::size_t begin, std::size_t end) {
  for (std::size_t i = begin; i < end; ++i) {
    Eigen::MatrixXd XinS = XinSample(X, s, i);
    Eigen::VectorXd yinS = yinSample(y, s, i);
    Eigen::MatrixXd XoutS = XoutSample(X, s, i);
    Eigen::VectorXd youtS = youtSample(y, s, i);
    Eigen::VectorXd W = OLScoef(XinS, yinS, pivot, rankCheck);
    Eigen::VectorXd yhat = XoutS * W;
    double costI = cost(youtS, yhat);
    double alphaI = static_cast<double>(ns[i]) / n;
    MSE += alphaI * costI;
  }
}

void CVWorker::join(const CVWorker& CVw) {
  MSE += CVw.MSE;
}
