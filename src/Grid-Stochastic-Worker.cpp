#include "Grid-Stochastic-Worker.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

namespace Grid::Stochastic {

// Ctor
Worker::Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
               const Eigen::VectorXi& foldIDs, const Eigen::VectorXi& foldSizes,
               const Eigen::VectorXd& lambdas, const Eigen::Index nrow,
               const Eigen::Index maxTrainSize, const Eigen::Index maxTestSize)
    : y_{y},
      x_{x},
      foldIDs_{foldIDs},
      foldSizes_{foldSizes},
      lambdas_{lambdas},
      nrow_{nrow},
      maxTrainSize_{maxTrainSize},
      maxTestSize_{maxTestSize},
      mses_(Eigen::VectorXd::Zero(lambdas.size())),
      trainIdxs_(maxTrainSize_),
      testIdxs_(maxTestSize_),
      uty_(x.cols()),
      eigenVals_(x.cols()),
      eigenValsSq_(x.cols()),
      diagD_(x.cols()),
      beta_(x.cols()),
      resid_(maxTestSize_) {}

// Split ctor
Worker::Worker(const Worker& other, const RcppParallel::Split)
    : y_{other.y_},
      x_{other.x_},
      foldIDs_{other.foldIDs_},
      foldSizes_{other.foldSizes_},
      lambdas_{other.lambdas_},
      nrow_{other.nrow_},
      maxTrainSize_{other.maxTrainSize_},
      maxTestSize_{other.maxTestSize_},
      mses_(Eigen::VectorXd::Zero(other.lambdas_.size())),
      trainIdxs_(maxTrainSize_),
      testIdxs_(maxTestSize_),
      uty_(other.uty_.size()),
      eigenVals_(other.eigenVals_.size()),
      eigenValsSq_(other.eigenValsSq_.size()),
      diagD_(other.diagD_.size()),
      beta_(other.beta_.size()),
      resid_(maxTestSize_) {}

// Work operator
void Worker::operator()(const std::size_t begin, const std::size_t end) {
  // Casting from std::size_t to int is safe here (end is the number of folds
  // which is a 32-bit integer from R)
  for (int foldID{static_cast<int>(begin)}, endID{static_cast<int>(end)};
       foldID < endID; ++foldID) {
    const Eigen::Index testSize{foldSizes_[foldID]};
    const Eigen::Index trainSize{nrow_ - testSize};

    // Prepare training and testing containers
    Eigen::Index tr{0};
    Eigen::Index ts{0};

    for (int row{0}; row < nrow_; ++row) {
      if (foldIDs_[row] == foldID) {
        testIdxs_[ts++] = row;
      } else {
        trainIdxs_[tr++] = row;
      }
    }

    // SVD module does not support in-place matrix decomposition
    const Eigen::BDCSVD<Eigen::MatrixXd> svd{
        x_(trainIdxs_.head(trainSize), Eigen::all),
        Eigen::ComputeThinU | Eigen::ComputeThinV};
    const auto& u{svd.matrixU()};
    const auto& v{svd.matrixV()};
    uty_.noalias() = u.transpose() * y_(trainIdxs_.head(trainSize));
    eigenVals_ = svd.singularValues();
    eigenValsSq_ = eigenVals_.square();

    for (Eigen::Index lambdaIdx{0}, nLambda{lambdas_.size()};
         lambdaIdx < nLambda; ++lambdaIdx) {
      // TO DO: Handle rank-deficient and 0 shrinkage case
      // diag(D)_i = diag(eigenVal_i / eigenVal_i^2 + lambda)
      diagD_ = eigenVals_ / (eigenValsSq_ + lambdas_[lambdaIdx]);

      // beta = V * diagD * U'y
      beta_.noalias() = v * (diagD_ * uty_.array()).matrix();

      // Evaluate performance on hold-out fold (MSE)
      resid_.head(testSize) = y_(testIdxs_.head(testSize));
      resid_.head(testSize).noalias() -=
          (x_(testIdxs_.head(testSize), Eigen::all) * beta_);
      const double foldMSE{resid_.head(testSize).squaredNorm() / testSize};

      // Weighted MSE contribution
      const double alpha{static_cast<double>(testSize) / nrow_};
      mses_[lambdaIdx] += (alpha * foldMSE);
    }
  }
}

// reduce results
void Worker::join(const Worker& other) { mses_ += other.mses_; }

};  // namespace Grid::Stochastic
