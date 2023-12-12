// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>
#include "FUNScvLM.h"
#include "cvLMWorker.h"

cvLMWorker::cvLMWorker(const Eigen::VectorXd& y_, const Eigen::MatrixXd& X_, const Eigen::VectorXi& s_, const IntegerVector& ns_, const int& n_, const String& pivot_, const bool& rankCheck_)
  : y(y_), X(X_), s(s_), ns(ns_), n(n_), pivot(pivot_), rankCheck(rankCheck_), MSE(0.0) {}
cvLMWorker::cvLMWorker(const cvLMWorker& CVLMW, RcppParallel::Split)
  : y(CVLMW.y), X(CVLMW.X), s(CVLMW.s), ns(CVLMW.ns), n(CVLMW.n), pivot(CVLMW.pivot), rankCheck(CVLMW.rankCheck), MSE(0.0) {}

void cvLMWorker::operator()(std::size_t begin, std::size_t end) {
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

void cvLMWorker::join(const cvLMWorker& CVLMW) {
  MSE += CVLMW.MSE;
}
