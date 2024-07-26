// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "FUNScvLM.h"

#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppParallel.h>

#include "cvLMWorker.h"
#include "cvRidgeWorker.h"

using namespace Rcpp;

// Confirm valid value for K
int Kcheck(const int &n, const int &K0) {
  int K = K0;
  double nDbl = static_cast<double>(n);
  int n2 = static_cast<int>(floor(nDbl / 2));
  NumericVector KvalsR(n2);
  for (int i = 0; i < n2; ++i) {
    KvalsR[i] = round(nDbl / (i + 1));
  }
  IntegerVector Kvals = as<IntegerVector>(KvalsR);
  Kvals = sort_unique(Kvals, true);
  int temp = n;
  int minDiff = temp - K;
  bool anyK = false;
  for (int i = 0; i < Kvals.size(); ++i) {
    int Diff = Kvals[i] - K0;
    int absDiff = (Diff < 0) ? -Diff : Diff;
    if (absDiff == 0) {
      anyK = true;
      break;
    } else if (absDiff < minDiff) {
      minDiff = absDiff;
      temp = Kvals[i];
    }
  }
  if (!anyK) {
    K = temp;
    warning("K has been changed from " + std::to_string(K0) + " to " + std::to_string(K) + ".");
  }
  return K;
}

// Calculate MSE
double cost(const Eigen::VectorXd &y, const Eigen::VectorXd &yhat) {
  double mse = (y - yhat).array().square().mean();
  return mse;
}

// Generate OLS coefficients
Eigen::VectorXd OLScoef(const Eigen::VectorXd &y, const Eigen::MatrixXd &X) {
  int p = X.cols();
  Eigen::VectorXd w(p);
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> QRPT(X);
  int r = QRPT.rank();
  if (r == p) {  // full-rank
    w = QRPT.solve(y);
  } else {  // rank-deficient
    Eigen::MatrixXd Rinv(
        QRPT.matrixQR().topLeftCorner(r, r).triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(r, r)));
    Eigen::VectorXd coef(QRPT.householderQ().transpose() * y);
    w.setZero();
    w.head(r) = Rinv * coef.head(r);
    w = QRPT.colsPermutation() * w;
  }
  return w;
}

// Generate Ridge regression coefficients
Eigen::VectorXd Ridgecoef(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const double &lambda) {
  int p = X.cols();
  Eigen::MatrixXd XT = X.transpose();
  Eigen::MatrixXd XTXlambda = Eigen::MatrixXd(p, p).setZero().selfadjointView<Eigen::Lower>().rankUpdate(XT);
  XTXlambda.diagonal().array() += lambda;
  Eigen::LDLT<Eigen::MatrixXd> PTLDLTP(XTXlambda);
  Eigen::VectorXd w = PTLDLTP.solve(XT * y);
  return w;
}

// Extract elements of our features that are in-sample
Eigen::MatrixXd XinSample(const Eigen::MatrixXd &X, const Eigen::VectorXi &s, const int &i) {
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
Eigen::VectorXd yinSample(const Eigen::VectorXd &y, const Eigen::VectorXi &s, const int &i) {
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
Eigen::MatrixXd XoutSample(const Eigen::MatrixXd &X, const Eigen::VectorXi &s, const int &i) {
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
Eigen::VectorXd youtSample(const Eigen::VectorXd &y, const Eigen::VectorXi &s, const int &i) {
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
IntegerVector sampleCV(const IntegerVector &x, const int &size) {
  Function sample("sample");
  IntegerVector indices = sample(x, size);
  return indices;
}

// Setup partitions for CV
List cvSetup(const int &seed, const int &n, const int &K) {
  Function setSeed("set.seed");
  setSeed(seed);
  int f = ceil(static_cast<double>(n) / K);
  IntegerVector x = rep(seq(1, K), f);
  IntegerVector samp = sampleCV(x, n);
  Eigen::VectorXi s = as<Eigen::Map<Eigen::VectorXi>>(samp);
  Eigen::VectorXi nsInt(K);
  for (int i = 0; i < K; ++i) {
    nsInt[i] = sum(samp == (i + 1));
  }
  Eigen::VectorXd ns = nsInt.cast<double>();
  return List::create(_["s"] = s, _["ns"] = ns);
}

// Generalized cross-validation for linear regression
double gcvOLS(const Eigen::VectorXd &y, const Eigen::MatrixXd &X) {
  int n = X.rows();
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> QRPT(X);
  Eigen::VectorXd w = QRPT.solve(y);
  Eigen::VectorXd yhat = X * w;
  int df = QRPT.rank();
  double leverage = 1 - (static_cast<double>(df) / n);
  double CV = ((y - yhat).array() / leverage).square().mean();
  return CV;
}

// Generalized cross-validation for ridge regression
double gcvRidge(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const double &lambda) {
  int n = X.rows();
  int p = X.cols();
  Eigen::BDCSVD<Eigen::MatrixXd> SVD(X, Eigen::ComputeThinU);
  Eigen::MatrixXd U = SVD.matrixU();
  Eigen::VectorXd evSq = SVD.singularValues().array().square();
  Eigen::VectorXd df = evSq.array() / (evSq.array() + lambda);
  Eigen::VectorXd yhat = Eigen::VectorXd(n).setZero();
  for (int j = 0; j < p; ++j) {
    Eigen::VectorXd Uj = U.col(j);
    yhat += ((df[j] * Uj) * (Uj.transpose() * y));
  }
  double leverage = 1 - (df.sum() / n);
  double CV = ((y - yhat).array() / leverage).square().mean();
  return CV;
}

// Leave-one-out cross-validation for linear regression
double loocvOLS(const Eigen::VectorXd &y, const Eigen::MatrixXd &X) {
  int n = X.rows();
  int p = X.cols();
  Eigen::VectorXd w(p);
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> QRPT(X);
  int r = QRPT.rank();
  if (r == p) {  // full-rank
    w = QRPT.solve(y);
  } else {  // rank-deficient
    warning("Received a rank-deficient design matrix.");
    Eigen::MatrixXd Rinv(
        QRPT.matrixQR().topLeftCorner(r, r).triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(r, r)));
    Eigen::VectorXd coef(QRPT.householderQ().transpose() * y);
    w.setZero();
    w.head(r) = Rinv * coef.head(r);
    w = QRPT.colsPermutation() * w;
  }
  Eigen::VectorXd yhat = X * w;
  Eigen::MatrixXd Q = QRPT.householderQ() * Eigen::MatrixXd::Identity(n, r);
  Eigen::VectorXd diagH = Q.rowwise().squaredNorm();
  double CV = ((y - yhat).array() / (1 - diagH.array())).square().mean();
  return CV;
}

// Leave-one-out cross-validation for ridge regression
double loocvRidge(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const double &lambda) {
  int n = X.rows();
  int p = X.cols();
  Eigen::BDCSVD<Eigen::MatrixXd> SVD(X, Eigen::ComputeThinU);
  Eigen::MatrixXd U = SVD.matrixU();
  Eigen::VectorXd evSq = SVD.singularValues().array().square();
  Eigen::VectorXd df = evSq.array() / (evSq.array() + lambda);
  Eigen::VectorXd S = df.array().sqrt();
  Eigen::MatrixXd US = U * S.asDiagonal();
  Eigen::VectorXd diagH = US.rowwise().squaredNorm();
  Eigen::VectorXd yhat = Eigen::VectorXd(n).setZero();
  for (int j = 0; j < p; ++j) {
    Eigen::VectorXd Uj = U.col(j);
    yhat += ((df[j] * Uj) * (Uj.transpose() * y));
  }
  double CV = ((y - yhat).array() / (1 - diagH.array())).square().mean();
  return CV;
}

// Multi-threaded CV for linear regression
double parcvOLS(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K, const int &seed,
                const int &nthreads) {
  int n = X.rows();
  List Partitions = cvSetup(seed, n, K);
  Eigen::VectorXi s = Partitions["s"];
  Eigen::VectorXd ns = Partitions["ns"];
  cvLMWorker CVLMW(y, X, s, ns, n);
  RcppParallel::parallelReduce(0, K, CVLMW, 1, nthreads);
  double CV = CVLMW.MSE;
  return CV;
}

// Multi-threaded CV for ridge regression
double parcvRidge(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K, const double &lambda,
                  const int &seed, const int &nthreads) {
  int n = X.rows();
  List Partitions = cvSetup(seed, n, K);
  Eigen::VectorXi s = Partitions["s"];
  Eigen::VectorXd ns = Partitions["ns"];
  cvRidgeWorker CVRW(y, X, lambda, s, ns, n);
  RcppParallel::parallelReduce(0, K, CVRW, 1, nthreads);
  double CV = CVRW.MSE;
  return CV;
}

// CV for linear regression
double cvOLS(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K, const int &seed) {
  int n = X.rows();
  List Partitions = cvSetup(seed, n, K);
  Eigen::VectorXi s = Partitions["s"];
  Eigen::VectorXd ns = Partitions["ns"];
  double CV = 0;
  for (int i = 0; i < K; ++i) {
    Eigen::MatrixXd XinS = XinSample(X, s, i);
    Eigen::VectorXd yinS = yinSample(y, s, i);
    Eigen::MatrixXd XoutS = XoutSample(X, s, i);
    Eigen::VectorXd youtS = youtSample(y, s, i);
    Eigen::VectorXd winS = OLScoef(yinS, XinS);
    Eigen::VectorXd yhat = XoutS * winS;
    double costI = cost(youtS, yhat);
    double alphaI = ns[i] / n;
    CV += (alphaI * costI);
  }
  return CV;
}

// CV for ridge regression
double cvRidge(const Eigen::VectorXd &y, const Eigen::MatrixXd &X, const int &K, const double &lambda,
               const int &seed) {
  int n = X.rows();
  List Partitions = cvSetup(seed, n, K);
  Eigen::VectorXi s = Partitions["s"];
  Eigen::VectorXd ns = Partitions["ns"];
  double CV = 0;
  for (int i = 0; i < K; ++i) {
    Eigen::MatrixXd XinS = XinSample(X, s, i);
    Eigen::VectorXd yinS = yinSample(y, s, i);
    Eigen::MatrixXd XoutS = XoutSample(X, s, i);
    Eigen::VectorXd youtS = youtSample(y, s, i);
    Eigen::VectorXd winS = Ridgecoef(yinS, XinS, lambda);
    Eigen::VectorXd yhat = XoutS * winS;
    double costI = cost(youtS, yhat);
    double alphaI = ns[i] / n;
    CV += (alphaI * costI);
  }
  return CV;
}
