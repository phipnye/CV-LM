#include "DataLoader.h"

#include <RcppArmadillo.h>

#include <algorithm>
#include <cmath>
#include <tuple>

DataLoader::DataLoader(const arma::mat& X, const arma::vec& y, const int seed,
                       const arma::uword k)
    : Xsorted_(X.n_rows, X.n_cols),
      ySorted_(y.n_elem),
      testIDs_(X.n_rows),
      testStarts_{arma::zeros<arma::uvec>(k)},
      testSizes_{arma::zeros<arma::uvec>(k)},
      nrow_{X.n_rows},
      ncol_{X.n_cols} {
  // --- Assign test IDs

  // Call back into R for sample and set.seed to guarantee the exact same
  // random partitions as boot::cv.glm (using C++ RNG would break
  // reproducibility)
  const Rcpp::Function setSeed{"set.seed"};
  const Rcpp::Function sampleR{"sample"};
  std::ignore = setSeed(seed);  // returns NULL that we can ignore

  /*
   * Replicate fold assignment from boot::cv.glm:
   *   f <- ceiling(n/K)
   *   s <- sample0(rep(1L:K, f), n)
   */
  const int repeats{
      static_cast<int>(std::ceil(static_cast<double>(nrow_) / k))};
  const Rcpp::IntegerVector seqVec{Rcpp::rep(Rcpp::seq(1, k), repeats)};
  const Rcpp::IntegerVector sampled{sampleR(seqVec, static_cast<int>(nrow_))};

  // --- Compute test sizes and starting offsets

  // Compute test sizes
  for (arma::uword idx{0}; idx < nrow_; ++idx) {
    // Convert to zero-based indexing
    testIDs_[idx] = static_cast<arma::uword>(sampled[idx] - 1);
    maxTestSize_ = std::max(maxTestSize_, ++testSizes_[testIDs_[idx]]);
  }

  // Max train = n - max test
  maxTrainSize_ = nrow_ - testSizes_.min();

  // Compute test starts
  for (arma::uword foldIdx{1}; foldIdx < k; ++foldIdx) {
    testStarts_[foldIdx] = testStarts_[foldIdx - 1] + testSizes_[foldIdx - 1];
  }

  // --- Create sorted X and y with test sets stored contiguously
  arma::uvec nextIdxs{testStarts_};  // running counters

  for (arma::uword idx{0}; idx < nrow_; ++idx) {
    arma::uword pos{nextIdxs[testIDs_[idx]]++};
    Xsorted_.row(pos) = X.row(idx);
    ySorted_[pos] = y[idx];
  }
}

DataLoader::LoadValues DataLoader::load(const arma::uword testID,
                                        arma::mat& XtrainBuf,
                                        arma::vec& yTrainBuf) const {
  // Retrieve the indexes of the test set
  const arma::uword testStart{testStarts_[testID]};
  const arma::uword testSize{testSizes_[testID]};
  const arma::uword trainSize{nrow_ - testSize};

  // --- Load training data into buffers

  // Copy rows before the test fold
  XtrainBuf.head_rows(testStart) = Xsorted_.head_rows(testStart);
  yTrainBuf.head(testStart) = ySorted_.head(testStart);

  // Copy rows after the test fold into the remaining buffer space
  if (const arma::uword nTrailing{trainSize - testStart}; nTrailing > 0) {
    XtrainBuf.rows(testStart, testStart + nTrailing - 1) =
        Xsorted_.tail_rows(nTrailing);
    yTrainBuf.subvec(testStart, testStart + nTrailing - 1) =
        ySorted_.tail(nTrailing);
  }

  // Return views of the test data
  return {Xsorted_.rows(testStart, testStart + testSize - 1),
          ySorted_.subvec(testStart, testStart + testSize - 1), testSize,
          trainSize};
}

arma::uword DataLoader::maxTrain() const noexcept { return maxTrainSize_; }
arma::uword DataLoader::maxTest() const noexcept { return maxTestSize_; }
arma::uword DataLoader::nrow() const noexcept { return nrow_; }
arma::uword DataLoader::ncol() const noexcept { return ncol_; }
