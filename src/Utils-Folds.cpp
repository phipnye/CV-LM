#include "Utils-Folds.h"

#include <Rcpp.h>
#include <RcppEigen.h>

#include <cfenv>
#include <cmath>
#include <string>
#include <tuple>
#include <utility>

namespace Utils::Folds {

// RAII guard for the rounding mode
ScopedRoundingMode::ScopedRoundingMode(const int mode)
    : oldMode_{std::fegetround()} {
  std::fesetround(mode);
}

// Restore original rounding mode when object goes out-of-scope and gets
// destroyed
ScopedRoundingMode::~ScopedRoundingMode() { std::fesetround(oldMode_); }

// Confirm valid value for K
// Mimics boot::cv.glm logic to find a K that evenly (or nearly evenly) divides
// n (uses FE_TONEAREST to match R's banker rounding)
int kCheck(const int nrow, const int k0, const bool generalized) {
  // GCV or LOOCV
  if (generalized || nrow == k0) {
    return k0;
  }

  /*
   * We're essentially trying to mimic (from boot::cv.glm):
   *   kvals <- unique(round(n/(1L:floor(n/2))))
   *   temp <- abs(kvals - K)
   *   if (!any(temp == 0))
   *     K <- kvals[temp == min(temp)][1L]
   *   if (K != K.o)
   *     warning(gettextf("'K' has been set to %f", K), domain = NA)
   */

  // FE_TONEAREST -> round to nearest, ties to even
  [[maybe_unused]] const ScopedRoundingMode roundGuard{FE_TONEAREST};
  const double nDbl{static_cast<double>(nrow)};
  const int floorHalfN{nrow / 2};

  // Start with den = 1 (we already checked for no difference at beginning of
  // function)
  int closestK{nrow};
  int minDiff{closestK - k0};

  // Consider k values between n and 2 (iterates through possible denominators
  // to find a K value that fits n)
  for (int den{2}; den <= floorHalfN; ++den) {
    // Use banker's rounding
    const int kVal{static_cast<int>(std::nearbyint(nDbl / den))};
    const int absDiff{std::abs(k0 - kVal)};

    // Per: K <- kvals[temp == min(temp)][1L], take first instance
    if (absDiff == 0) {
      return k0;
    }

    if (absDiff < minDiff) {
      minDiff = absDiff;
      closestK = kVal;
    }
  }

  Rcpp::warning("K has been changed from %d to %d.", k0, closestK);
  return closestK;
}

DataSplitter::DataSplitter(const int seed, const Eigen::Index nrow, const int k)
    : testIDs_{[&]() {
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
            static_cast<int>(std::ceil(static_cast<double>(nrow) / k))};
        const Rcpp::IntegerVector seqVec{Rcpp::rep(Rcpp::seq(1, k), repeats)};
        const Rcpp::IntegerVector sampled{sampleR(seqVec, nrow)};
        Eigen::VectorXi testIDs{Rcpp::as<Eigen::VectorXi>(sampled)};

        // Convert to zero-indexing
        testIDs.array() -= 1;
        return testIDs;
      }()},

      // The number of test observations per fold
      testSizes_{[&]() -> Eigen::VectorXi {
        // R's internal documentation states the number of rows for a matrix are
        // limited to 32-bit values so VectorXi is safe here
        Eigen::VectorXi testSizes{Eigen::VectorXi::Zero(k)};

        // Store test fold sizes to calculate the weighted CV estimate:
        // sum((n_i / n) * cost_i)
        for (Eigen::Index idx{0}; idx < nrow; ++idx) {
          ++testSizes[testIDs_[idx]];
        }

        return testSizes;
      }()},

      // Record where the test indices for a given fold start
      testStarts_{[&]() -> Eigen::VectorXi {
        Eigen::VectorXi testStarts{Eigen::VectorXi::Zero(k)};

        for (Eigen::Index idx{1}; idx < k; ++idx) {
          testStarts[idx] = testStarts[idx - 1] + testSizes_[idx - 1];
        }

        return testStarts;
      }()},

      // buffer sizes in worker instances
      maxTestSize_{static_cast<Eigen::Index>(testSizes_.maxCoeff())},
      maxTrainSize_{nrow - testSizes_.minCoeff()},
      nrow_{nrow} {}

Eigen::VectorXi DataSplitter::buildPermutation() const {
  // Store the current index/offset for each fold to write an observation to
  Eigen::VectorXi offsets{testStarts_};
  Eigen::VectorXi perm{nrow_};

  for (int idx{0}; idx < nrow_; ++idx) {
    perm[offsets[testIDs_[idx]]++] = idx;
  }

  return perm;
}

std::pair<int, int> DataSplitter::operator[](const Eigen::Index idx) const {
  return std::make_pair(testStarts_[idx], testSizes_[idx]);
}

Eigen::Index DataSplitter::maxTrain() const noexcept { return maxTrainSize_; }
Eigen::Index DataSplitter::maxTest() const noexcept { return maxTestSize_; }

}  // namespace Utils::Folds
