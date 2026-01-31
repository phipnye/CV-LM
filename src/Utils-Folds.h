#ifndef CV_LM_UTILS_FOLDS_H
#define CV_LM_UTILS_FOLDS_H

#include <RcppArmadillo.h>

namespace Utils::Folds {

// RAII guard for setting rounding mode
class ScopedRoundingMode {
  // Store the original rounding mode so we can revert back to it once our
  // object goes out-of-scope
  const int oldMode_;

 public:
  explicit ScopedRoundingMode(int mode);
  ~ScopedRoundingMode();
};

// Confirm valid value for the number of folds
[[nodiscard]] arma::uword kCheck(arma::uword nrow, arma::uword k0,
                                 bool generalized);

}  // namespace Utils::Folds

#endif  // CV_LM_UTILS_FOLDS_H
