#include "Utils-Folds.h"

#include <RcppArmadillo.h>

#include <cfenv>
#include <cmath>

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
arma::uword kCheck(const arma::uword nrow, const arma::uword k0,
                   const bool generalized) {
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
  const arma::uword floorHalfN{nrow / 2};

  // Start with den = 1 (we already checked for no difference at beginning of
  // function)
  arma::uword closestK{nrow};
  arma::uword minDiff{closestK - k0};

  // Consider k values between n and 2 (iterates through possible denominators
  // to find a K value that fits n)
  for (arma::uword den{2}; den <= floorHalfN; ++den) {
    // Use banker's rounding
    const arma::uword kVal{
        static_cast<arma::uword>(std::nearbyint(nDbl / den))};
    const arma::uword absDiff{(k0 > kVal) ? (k0 - kVal) : (kVal - k0)};

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

}  // namespace Utils::Folds
