#ifndef CV_LM_CLOSEDFORM_H
#define CV_LM_CLOSEDFORM_H

#include <RcppArmadillo.h>

namespace ClosedForm {

// Closed-form solution for generalized cross-validation
[[nodiscard]] double gcv(double rss, double traceHat, arma::uword nrow);

// Closed-form solution for leave-one-out cross-validation
[[nodiscard]] double loocv(const arma::vec& residuals,
                           const arma::vec& diagHat);

}  // namespace ClosedForm

#endif  // CV_LM_CLOSEDFORM_H
