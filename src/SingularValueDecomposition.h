#ifndef CV_LM_SINGULARVALUEDECOMPOSITION_H
#define CV_LM_SINGULARVALUEDECOMPOSITION_H

#include <RcppArmadillo.h>

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "ClosedForm.h"
#include "ConstexprOptional.h"
#include "Enums.h"
#include "Utils-Data.h"

template <Enums::CrossValidationMethod CV, Enums::CenteringMethod Centering>
class SingularValueDecomposition {
 public:
  static constexpr bool requiresLambda{true};

 private:
  static constexpr bool meanCenter{Centering == Enums::CenteringMethod::Mean};
  static constexpr bool kcv{CV == Enums::CrossValidationMethod::KCV};
  static constexpr bool gcv{CV == Enums::CrossValidationMethod::GCV};
  static constexpr bool lcv{CV == Enums::CrossValidationMethod::LOOCV};

  // Design matrix state
  arma::mat U_{};
  arma::vec diagD_{};
  arma::vec diagDsq_{};
  ConstexprOptional<kcv, arma::mat> V_{};
  ConstexprOptional<kcv && meanCenter, arma::rowvec> XtrainColMeans_{};
  arma::uword nrow_{0};
  arma::uword ncol_{0};
  arma::uword rank_{0};
  double tolerance_;

  // Response state
  ConstexprOptional<lcv, arma::vec> y_{};
  ConstexprOptional<kcv || lcv, arma::vec> UTy_{};
  ConstexprOptional<gcv, arma::vec> UTySq_{};
  ConstexprOptional<meanCenter && kcv, double> yTrainMean_{0.0};
  ConstexprOptional<gcv, double> tss_{0.0};

  // Penalty state
  ConstexprOptional<kcv, arma::vec> singularShrink_{};
  ConstexprOptional<gcv || lcv, arma::vec> coordShrink_{};

  // Flags
  bool useOLS_{false};
  bool isDesignSet_{false};
  bool isResponseSet_{false};
  bool isLambdaSet_{false};
  bool success_{true};

 public:
  // Main ctor
  explicit SingularValueDecomposition(const double tolerance)
      : tolerance_{tolerance} {}

  // Copy ctor (copying is allowed for this class since determinstic workers
  // require the full decomposition - "cloning" is insufficient)
  SingularValueDecomposition(const SingularValueDecomposition& other) = default;

  // Create a new decomposition object sharing only the tolerance parameter
  [[nodiscard]] SingularValueDecomposition clone() const {
    return SingularValueDecomposition{tolerance_};
  }

  // Move ctor
  SingularValueDecomposition(SingularValueDecomposition&&) = default;

  // Dtor
  ~SingularValueDecomposition() = default;

  // Assigments shouldn't be necessary with this class
  SingularValueDecomposition& operator=(const SingularValueDecomposition&) =
      delete;
  SingularValueDecomposition& operator=(SingularValueDecomposition&&) = delete;

  // Set the design matrix and decompose X = UDV'
  template <typename T>
  [[nodiscard]] bool setDesign(const T& X0) {
    Utils::Data::assertMat<T>();

    // Potentially centered design matrix
    using DesignType =
        std::conditional_t<meanCenter, const arma::mat, const T&>;
    DesignType X{[&]() -> DesignType {
      if constexpr (meanCenter) {
        if constexpr (kcv) {
          arma::mat Xcentered{X0};
          // Store train colmeans so we can apply to test set
          Utils::Data::centerDesign(Xcentered, XtrainColMeans_.value());
          return Xcentered;
        } else {
          return Utils::Data::centerDesign(X0);
        }
      } else {
        return X0;
      }
    }()};

    // We use economic SVD (U [n x p], V [p x p]) - only compute V for K-fold so
    // we can compute coefficients that can be applied to out-of-sample
    // observations
    if constexpr (decltype(V_)::isEnabled) {
      success_ = arma::svd_econ(U_, diagD_, V_.value(), X);
    } else {
      // Only compute left singular vectors
      arma::mat Vplaceholder{};
      success_ = arma::svd_econ(U_, diagD_, Vplaceholder, X, "left");
    }

    // Make sure the decomposition was successful
    if (!success_) {
      return success_;
    }

    nrow_ = X.n_rows;
    ncol_ = X.n_cols;

    // Estimate rank: A singular value will be considered nonzero if its value
    // is strictly greater than tolerance x maxsingularvalue
    const double threshold{tolerance_ * diagD_[0]};
    diagD_.clean(threshold);
    rank_ = arma::accu(diagD_ != 0.0);
    diagDsq_ = arma::square(diagD_);
    isDesignSet_ = success_;
    return success_;
  }

  // Set the response vector (always returns true - no LAPACK calls)
  template <typename T>
  bool setResponse(const T& y0) {
    Utils::Data::assertVec<T>();
    assert(isDesignSet_ && "Must set design matrix before setting a response");
    assert(y0.n_elem == nrow_);

    // Potentially centered response vector
    if constexpr (decltype(y_)::isEnabled) {
      y_.value() = meanCenter ? Utils::Data::centerResponse(y0) : y0;
    }

    // Avoid generating additional copies of the response
    using ResponseType = std::conditional_t<
        !meanCenter, const T&,
        std::conditional_t<lcv, const arma::vec&, const arma::vec>>;
    ResponseType y{[&]() -> ResponseType {
      if constexpr (!meanCenter) {
        return y0;
      } else if constexpr (decltype(y_)::isEnabled) {
        return y_.value();
      } else {
        return Utils::Data::centerResponse(y0);
      }
    }()};

    // Projection of y onto left singular values of X (U'y)
    if constexpr (decltype(UTy_)::isEnabled) {
      UTy_.value() = U_.t() * y;
    }

    // Squared projection values
    if constexpr (decltype(UTySq_)::isEnabled) {
      if constexpr (decltype(UTy_)::isEnabled) {
        UTySq_.value() = arma::square(UTy_.value());
      } else {
        UTySq_.value() = arma::square(U_.t() * y);
      }
    }

    // Response average
    if constexpr (decltype(yTrainMean_)::isEnabled) {
      yTrainMean_.value() = arma::mean(y0);
    }

    // Total sum of squares
    if constexpr (decltype(tss_)::isEnabled) {
      tss_.value() = arma::dot(y, y);
    }

    isResponseSet_ = true;
    return true;  // nothing lapack-related in this function
  }

  // Set lambda
  void setLambda(const double lambda) {
    assert(isDesignSet_ && isResponseSet_ &&
           "Must set design matrix and response vector before setting lambda");

    // OLS (solved separately to provide minimum norm solutions)
    useOLS_ = lambda <= 0.0;

    if (useOLS_) {
      if constexpr (decltype(singularShrink_)::isEnabled) {
        // Singular shrinkage values simplify to 1 / diag(D) for lambda == 0.0
        singularShrink_.value().zeros(diagD_.n_elem);
        singularShrink_.value().head(rank_) = 1.0 / diagD_.head(rank_);
      }

      if constexpr (decltype(coordShrink_)::isEnabled) {
        // Coordinate shrinkage factors simplify to 1 for lamdab == 0.0
        coordShrink_.value() = arma::ones(diagD_.n_elem);
      }

      isLambdaSet_ = true;
      return;
    }

    // Singular shrinkage factor = d_j / (d_j^2 + lambda)
    if constexpr (decltype(singularShrink_)::isEnabled) {
      singularShrink_.value() = diagD_ / (diagDsq_ + lambda);
    }

    // Coordinate shrinkage factors = d_j^2 / (d_j^2 + lambda) see ESL p.66
    if constexpr (decltype(coordShrink_)::isEnabled) {
      if constexpr (decltype(singularShrink_)::isEnabled) {
        coordShrink_.value() = diagD_ % singularShrink_.value();
      } else {
        coordShrink_.value() = diagDsq_ / (diagDsq_ + lambda);
      }
    }

    isLambdaSet_ = true;
  }

  // Determinstic cross-validation methods
  template <bool deterministic = gcv || lcv,
            typename = std::enable_if_t<deterministic>>
  [[nodiscard]] double cv() const {
    // The only failures for SVD should come from svd_econ which should be
    // checked by the user immediately upon setting the design matrix, other
    // instances are misuses
    assert(isReady() &&
           "Attempting to compute deterministic CV values while SVD is not in "
           "a complete state.");

    if constexpr (gcv) {
      return ClosedForm::gcv(rss(), traceHat(), nrow_);
    } else {
      Enums::assertExpected<CV, Enums::CrossValidationMethod::LOOCV>();
      return ClosedForm::loocv(residuals(), diagHat());
    }
  }

  // MSE for test set (for stochastic (K-Fold) cross-validation)
  template <bool stochastic = kcv, typename = std::enable_if_t<stochastic>>
  [[nodiscard]] double testMSE(const arma::subview<double>& Xtest,
                               const arma::subview_col<double>& yTest) const {
    // The only failures for SVD should come from svd_econ which should be
    // checked by the user immediately upon setting the design matrix, other
    // instances are misuses
    Enums::assertExpected<CV, Enums::CrossValidationMethod::KCV>();
    assert(isReady() &&
           "Attempting to evaluate out-of-sample performance while SVD is not "
           "in a complete state.");

    // Beta is computed on training set
    const arma::vec beta{solve()};

    if constexpr (meanCenter) {
      // Pre-calculate the scalar offset: y_mean - X_means * beta (accounts
      // for the centering shift without copying the whole test matrix)
      const double offset{yTrainMean_.value() -
                          arma::dot(XtrainColMeans_.value(), beta)};

      // Residual = y_test - (X_test * beta + offset)
      return arma::mean(arma::square(yTest - ((Xtest * beta) + offset)));
    } else {
      // Standard calculation for non-centered data
      return arma::mean(arma::square(yTest - (Xtest * beta)));
    }
  }

 private:
  // --- Internal modular calculations

  // Minimum-norm least squares or ridge regression coefficients
  [[nodiscard]] arma::vec solve() const {
    if (useOLS_) {
      return V_.value().head_cols(rank_) *
             (singularShrink_.value().head(rank_) % UTy_.value().head(rank_));
    }

    // See ESL p.66
    return V_.value() * (singularShrink_.value() % UTy_.value());
  }

  // Sum of squared residuals
  [[nodiscard]] double rss() const {
    // OLS
    if (useOLS_) {
      // Fully saturated linear regression model
      if (rank_ + (meanCenter ? 1u : 0u) == nrow_) {
        return 0.0;
      }

      const double ess{arma::accu(UTySq_.value().head(rank_))};
      return std::max(tss_.value() - ess, 0.0);
    }

    /*
     * rss = ||resid||^2
     *     = ||y - U D(D^2 + LI)^-1 DU'y||^2
     *     = ||y - USU'y||^2
     *     = [y - USU'y]'[y - USU'y]
     *     = [y' - (USU'y)'][y - USU'y]
     *     = y'y - y'USU'y - (USU'y)'y + (USU'y)'USU'y
     *     = ||y||^2 - y'USU'y - y'USU'y + y'US'U'USU'y
     *     = ||y||^2 - 2 * y'USU'y + y'U S^2 U'y
     *     = ||y||^2 - (U'y)' 2S U'y + (U'y)' S^2 U'y
     *     = ||y||^2 - sum_{j} (2 s_j - s_j^2) (U'y)_j^2
     */
    const double essRidge{
        arma::accu((coordShrink_.value() % (2.0 - coordShrink_.value())) %
                   UTySq_.value())};
    return std::max(tss_.value() - essRidge, 0.0);
  }

  [[nodiscard]] arma::vec residuals() const {
    // OLS
    if (useOLS_) {
      // Fully saturated linear regression model
      if (rank_ + (meanCenter ? 1u : 0u) == nrow_) {
        return arma::zeros(nrow_);
      }

      return y_.value() - (U_.head_cols(rank_) * UTy_.value().head(rank_));
    }

    /*
     * [See ESL p.66]
     * resid = y - X * beta
     *       = y - X * (X'X + LI)^-1 X'y
     *       = y - U D(D^2 + LI)^-1 DU'y
     *       = y - sum_j u_j (d_j^2) / (d_j^2 + lambda) u_j'y
     */
    return y_.value() - (U_ * (coordShrink_.value() % UTy_.value()));
  }

  [[nodiscard]] double traceHat() const {
    // If we are centering the data, we dropped the intercept term in R and need
    // to add one to correct the rank to the rank of the original design matrix
    constexpr double correction{meanCenter ? 1.0 : 0.0};

    if (useOLS_) {
      return static_cast<double>(rank_) + correction;  // tr(H) = rank(X)
    }

    // Trace(H) = sum_{j=1}^{p} d_j^2 / (d_j^2 + lambda) [See ESL p.68]
    return arma::accu(coordShrink_.value()) + correction;
  }

  [[nodiscard]] arma::vec diagHat() const {
    arma::vec diagH{[&]() -> arma::vec {
      // In OLS case, H = UU', so diagonals again equate to squared row norms
      // [see ESL p.66]
      if (useOLS_) {
        return arma::sum(arma::square(U_.head_cols(rank_)), 1);
      }

      // Otherwise, h_ii = sum_j (u_ij^2 * (d_j^2 / (d_j^2 + lambda)))
      return arma::square(U_) * coordShrink_.value();
    }()};

    // If the data was centered, we need to add 1/n (diag(11')/n) to the
    // diagonal entries to capture the dropped intercept column
    if constexpr (meanCenter) {
      diagH += (1.0 / static_cast<double>(nrow_));
    }

    return diagH;
  }

  // Determine if the decomposition is ready for modular calculations
  [[nodiscard]] bool isReady() const noexcept {
    return isDesignSet_ && isResponseSet_ && isLambdaSet_ && success_;
  }
};

#endif  // CV_LM_SINGULARVALUEDECOMPOSITION_H
