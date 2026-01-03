// [[Rcpp::depends(RcppEigen, RcppParallel)]]

#include "include/DeterministicGridWorker.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include <limits>
#include <utility>

namespace Grid {

// --- Base implementation

DeterministicGridWorker::DeterministicGridWorker(
    double maxLambda, double precision, bool centered,
    const Eigen::ArrayXd& eigenValsSq, const Eigen::VectorXd& uty,
    const Eigen::Index nrow)
    : maxLambda_{maxLambda},
      precision_{precision},
      centered_{centered},
      eigenValsSq_{eigenValsSq},
      uty_{uty},
      nrow_{nrow},
      results_{std::numeric_limits<double>::infinity(), 0.0},
      denom_(eigenValsSq.size()) {}

void DeterministicGridWorker::join(const DeterministicGridWorker& other) {
  if (other.results_.first < results_.first) {
    results_ = other.results_;
  }
}

// --- GCV Implementation

GCVGridWorker::GCVGridWorker(double maxLambda, double precision, bool centered,
                             const Eigen::ArrayXd& eigenValsSq,
                             const Eigen::VectorXd& uty,
                             const Eigen::Index nrow, const double rssNull,
                             const Eigen::ArrayXd& utySq)
    : DeterministicGridWorker{maxLambda,   precision, centered,
                              eigenValsSq, uty,       nrow},
      rssNull_{rssNull},
      utySq_{utySq} {}

GCVGridWorker::GCVGridWorker(const GCVGridWorker& other, RcppParallel::Split)
    : DeterministicGridWorker{other.maxLambda_, other.precision_,
                              other.centered_,  other.eigenValsSq_,
                              other.uty_,       other.nrow_},
      rssNull_{other.rssNull_},
      utySq_{other.utySq_} {}

void GCVGridWorker::operator()(std::size_t begin, std::size_t end) {
  for (std::size_t step{begin}; step < end; ++step) {
    const double lambda{static_cast<double>(step) * precision_};

    // Calculate trace(H) = sum(eigenVals^2 / (eigenVals^2 + lambda))
    // TO DO: Handle case of collinearity with zero shrinkage
    denom_ = eigenValsSq_ + lambda;
    double traceH{(eigenValsSq_ / denom_).sum()};

    // Add 1 for the unpenalized intercept if data was centered
    if (centered_) {
      traceH += 1.0;
    }

    // RSS = sum_{i=1}^{r}((lambda^2 + (eigenVal_i^2 + lambda)^2) * u_i'y^2) +
    // ||(I - UU')y||^2
    const double rss{rssNull_ +
                     ((lambda * lambda) * utySq_ / denom_.square()).sum()};

    // GCV = MSE / (1 - trance(H) / n)^2
    const double leverage{1.0 - (traceH / nrow_)};
    const double gcv{rss / (nrow_ * leverage * leverage)};

    if (gcv < results_.first) {
      results_.second = lambda;
      results_.first = gcv;
    }
  }
}

// --- LOOCV Implementation

LOOCVGridWorker::LOOCVGridWorker(double maxLambda, double precision,
                                 bool centered,
                                 const Eigen::ArrayXd& eigenValsSq,
                                 const Eigen::VectorXd& uty, Eigen::Index nrow,
                                 const Eigen::VectorXd& yNull,
                                 const Eigen::MatrixXd& u,
                                 const Eigen::MatrixXd& uSq)
    : DeterministicGridWorker{maxLambda,   precision, centered,
                              eigenValsSq, uty,       nrow},
      yNull_{yNull},
      u_{u},
      uSq_{uSq},
      diagD_(eigenValsSq_.size()),
      diagH_(nrow_),
      resid_(nrow_) {}

LOOCVGridWorker::LOOCVGridWorker(const LOOCVGridWorker& other,
                                 RcppParallel::Split)
    : DeterministicGridWorker{other.maxLambda_, other.precision_,
                              other.centered_,  other.eigenValsSq_,
                              other.uty_,       other.nrow_},
      yNull_{other.yNull_},
      u_{other.u_},
      uSq_{other.uSq_},
      diagD_(other.diagD_.size()),
      diagH_(other.diagH_.size()),
      resid_(other.resid_.size()) {}

void LOOCVGridWorker::operator()(std::size_t begin, std::size_t end) {
  for (std::size_t step{begin}; step < end; ++step) {
    const double lambda{static_cast<double>(step) * precision_};
    // TO DO: Handle case of collinearity with zero shrinkage
    // Calculate the diagonal of the Ridge shrinkage matrix = eigenVal_i^2 /
    // (eigenVal_i^2 + lambda)
    denom_ = eigenValsSq_ + lambda;  // denom_ is reused to avoid temporary
                                     // allocations and repeated computations
    diagD_ = eigenValsSq_ / denom_;

    // Diagonal of hat matrix = diag(U * shrinkage * U'): optimized as diag(H) =
    // (U^2) * diag(shrinkage)
    diagH_.matrix().noalias() = uSq_ * diagD_.matrix();

    // Add 1/n to account for the unpenalized intercept if the data was centered
    if (centered_) {
      diagH_.array() += (1.0 / nrow_);
    }

    // Calculate Ridge residuals: e = y - y_hat = yNull + U * [(lambda /
    // (eigenVals^2 + lambda)) * U'y]
    resid_ = yNull_;
    resid_.noalias() += u_ * ((lambda / denom_) * uty_.array()).matrix();

    // Leave-One-Out CV formula: CV = mean((e_i / (1 - h_ii))^2)
    const double loocv{(resid_.array() / (1.0 - diagH_)).square().mean()};

    // Track the lambda that minimizes the LOOCV error
    if (loocv < results_.first) {
      results_.second = lambda;
      results_.first = loocv;
    }
  }
}

}  // namespace Grid
