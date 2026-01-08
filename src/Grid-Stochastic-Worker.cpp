#include "Grid-Stochastic-Worker.h"

#include <RcppEigen.h>
#include <RcppParallel.h>

#include "Grid-Generator.h"

namespace Grid::Stochastic {

// Ctor
Worker::Worker(const Eigen::VectorXd& y, const Eigen::MatrixXd& x,
               const Eigen::VectorXi& foldIDs, const Eigen::VectorXi& foldSizes,
               const Generator& lambdasGrid, const Eigen::Index maxTrainSize,
               const Eigen::Index maxTestSize)
    : y_{y},
      x_{x},
      foldIDs_{foldIDs},
      foldSizes_{foldSizes},
      lambdasGrid_{lambdasGrid},
      nrow_{x_.rows()},
      mses_(Eigen::VectorXd::Zero(lambdasGrid.size())),
      uty_(x.cols()),
      beta_(x.cols()),
      resid_(maxTestSize),
      eigenVals_(x.cols()),
      eigenValsSq_(x.cols()),
      diagD_(x.cols()),
      trainIdxs_(maxTrainSize),
      testIdxs_(maxTestSize),
      svd_(maxTrainSize, x.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV)
      {}

// Split ctor
Worker::Worker(const Worker& other, const RcppParallel::Split)
    : y_{other.y_},
      x_{other.x_},
      foldIDs_{other.foldIDs_},
      foldSizes_{other.foldSizes_},
      lambdasGrid_{other.lambdasGrid_},
      nrow_{other.nrow_},
      mses_(Eigen::VectorXd::Zero(other.mses_.size())),
      uty_(other.uty_.size()),
      beta_(other.beta_.size()),
      resid_(other.resid_.size()),
      eigenVals_(other.eigenVals_.size()),
      eigenValsSq_(other.eigenValsSq_.size()),
      diagD_(other.diagD_.size()),
      trainIdxs_(other.trainIdxs_.size()),
      testIdxs_(other.testIdxs_.size()),
      svd_(other.svd_.rows(), other.svd_.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV)
      {}

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

    // Perform SVD on training set
    svd_.compute(x_(trainIdxs_.head(trainSize), Eigen::all));
    const auto& u{svd_.matrixU()};
    const auto& v{svd_.matrixV()};
    uty_.noalias() = u.transpose() * y_(trainIdxs_.head(trainSize));
    eigenVals_ = svd_.singularValues();
    eigenValsSq_ = eigenVals_.square();

    for (Eigen::Index lambdaIdx{0}, nLambda{lambdasGrid_.size()};
         lambdaIdx < nLambda; ++lambdaIdx) {
      // TO DO: Handle rank-deficient and 0 shrinkage case
      // diag(D)_i = diag(eigenVal_i / eigenVal_i^2 + lambda)
      diagD_ = eigenVals_ / (eigenValsSq_ + lambdasGrid_[lambdaIdx]);

      // beta = V * diagD * U'y
      beta_.noalias() = v * (diagD_ * uty_.array()).matrix();

      // Evaluate performance on hold-out fold (MSE)
      resid_.head(testSize) = y_(testIdxs_.head(testSize));
      resid_.head(testSize).noalias() -=
          (x_(testIdxs_.head(testSize), Eigen::all) * beta_);
      const double foldMSE{resid_.head(testSize).squaredNorm() /
                           static_cast<double>(testSize)};

      // Weighted MSE contribution
      const double alpha{static_cast<double>(testSize) /
                         static_cast<double>(nrow_)};
      mses_[lambdaIdx] += (alpha * foldMSE);
    }
  }
}

// reduce results
void Worker::join(const Worker& other) { mses_ += other.mses_; }

}  // namespace Grid::Stochastic
