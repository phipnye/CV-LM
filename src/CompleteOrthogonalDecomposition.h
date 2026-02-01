#ifndef CV_LM_COMPLETEORTHOGONALDECOMPOSITION_H
#define CV_LM_COMPLETEORTHOGONALDECOMPOSITION_H

#include <RcppArmadillo.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "ClosedForm.h"
#include "ConstexprOptional.h"
#include "Enums.h"
#include "Utils-Data.h"

// Armadillo doesn't expose dormqr
extern "C" {
void F77_NAME(dormqr)(const char* side, const char* trans, const int* m,
                      const int* n, const int* k, const double* a,
                      const int* lda, const double* tau, double* c,
                      const int* ldc, double* work, const int* lwork, int* info,
                      std::size_t, std::size_t);
}

template <Enums::CrossValidationMethod CV, Enums::CenteringMethod Centering>
class CompleteOrthogonalDecomposition {
 public:
  static constexpr bool requiresLambda{false};

 private:
  static constexpr bool meanCenter{Centering == Enums::CenteringMethod::Mean};
  static constexpr bool kcv{CV == Enums::CrossValidationMethod::KCV};
  static constexpr bool gcv{CV == Enums::CrossValidationMethod::GCV};
  static constexpr bool lcv{CV == Enums::CrossValidationMethod::LOOCV};

  // Design matrix state
  arma::mat QR_{}, ZU_{};
  arma::vec Qtau_{}, Ztau_{};
  arma::uvec piv_{};
  ConstexprOptional<kcv && meanCenter, arma::rowvec> XtrainColMeans_{};
  arma::uword nrow_{0};
  arma::uword ncol_{0};
  arma::uword rank_{0};
  double tolerance_;

  // Response state
  arma::vec QTy_{};
  ConstexprOptional<kcv && meanCenter, double> yTrainMean_{0.0};
  ConstexprOptional<gcv, double> tss_{0.0};

  // Flags
  bool isDesignSet_{false};
  bool isResponseSet_{false};

  // mutable: LAPACK applications may fail even in logically-const operations
  mutable bool success_{true};

  // Enum for differentiating the householders Q and Z
  enum class Householder : std::int8_t { Q, Z };

 public:
  // Main ctor
  explicit CompleteOrthogonalDecomposition(const double tolerance)
      : tolerance_{tolerance} {}

  // For our use, we don't need a full copy of the data (any such use would be
  // erroneous) - instead we just need something to "clone" the tolerance
  CompleteOrthogonalDecomposition(const CompleteOrthogonalDecomposition&) =
      delete;

  // Create a new decomposition object sharing only the tolerance parameter
  [[nodiscard]] CompleteOrthogonalDecomposition clone() const {
    return CompleteOrthogonalDecomposition{tolerance_};
  }

  // Move ctor
  CompleteOrthogonalDecomposition(CompleteOrthogonalDecomposition&&) = default;

  // Dtor
  ~CompleteOrthogonalDecomposition() = default;

  // Assigments shouldn't be necessary with this class
  CompleteOrthogonalDecomposition& operator=(
      const CompleteOrthogonalDecomposition&) = delete;
  CompleteOrthogonalDecomposition& operator=(
      CompleteOrthogonalDecomposition&&) = delete;

  // Set the design matrix and decompose XP = QR = QTZ (we don't actually
  // form the full decomposition but mimic it for min norm solutions)
  template <typename T>
  [[nodiscard]] bool setDesign(const T& X0) {
    // Possibly centered design matrix
    Utils::Data::assertMat<T>();

    if constexpr (decltype(XtrainColMeans_)::isEnabled) {
      // Store the column means and center X into QR_
      Utils::Data::centerDesign(X0, QR_, XtrainColMeans_.value());
    } else {
      QR_ = meanCenter ? Utils::Data::centerDesign(X0) : X0;
    }

    nrow_ = QR_.n_rows;
    ncol_ = QR_.n_cols;

    // Decompose XP = QR
    if (!colPivotQR()) {
      return (success_ = false);
    }

    // Estimate rank: A pivot will be considered nonzero if its absolute value
    // is strictly greater than tolerance x |maxpivot|
    arma::diagview pivots{QR_.diag()};

    // Diagonal entries of R are ordered from largest to smallest magnitude
    const double threshold{tolerance_ * std::abs(pivots[0])};
    pivots.clean(threshold);
    rank_ = arma::accu(pivots != 0.0);

    // --- Use QR decomposition of R1' to compute ZU

    // Extract R1 (the rank x p upper triangular part)
    ZU_ = arma::zeros(rank_, ncol_);

    if (rank_ > 0) {
      for (arma::uword col{0}; col < ncol_; ++col) {
        // Ensure row does not exceed the rank or the physical rows of QR_
        const arma::uword maxRow{std::min(col, rank_ - 1)};

        for (arma::uword row{0}; row <= maxRow; ++row) {
          ZU_.at(row, col) = QR_.at(row, col);
        }
      }
    }

    arma::inplace_trans(ZU_);
    success_ = econQR();  // decompose R1' = ZU
    isDesignSet_ = success_;
    return success_;
  }

  // Set the response vector
  template <typename T>
  [[nodiscard]] bool setResponse(const T& y0) {
    Utils::Data::assertVec<T>();
    assert(isDesignSet_ && "Must set design matrix before setting a response");
    assert(y0.n_elem == nrow_);

    // Potentially centered response vector
    using ResponseType =
        std::conditional_t<meanCenter, const arma::vec, const T&>;
    ResponseType y{[&]() -> ResponseType {
      if constexpr (meanCenter) {
        return Utils::Data::centerResponse(y0);
      } else {
        return y0;
      }
    }()};

    // Projection of y onto orthonormal basis for column space of X see ESL p.55
    QTy_ = y;

    if (!applyHouseholderOnLeft(QTy_, Householder::Q, true)) {
      return (success_ = false);
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
    return success_;
  }

  // Determinstic cross-validation methods
  template <bool deterministic = gcv || lcv,
            typename = std::enable_if_t<deterministic>>
  [[nodiscard]] double cv() const {
    // For COD, lapack applications could fail either during one of the two QR
    // decompositions or applying householder transformations to a matrix or
    // vector, hence more careful success checking is required

    if constexpr (gcv) {
      // rss() and traceHat() do not call lapack and hence the unsuccessful
      // states should have already been handled
      assert(isReady() &&
             "Attempting to compute GCV values while COD is not in a complete "
             "state.");
      return ClosedForm::gcv(rss(), traceHat(), nrow_);
    } else {
      // residuals() and diagHat() both call lapack, so we gather their values
      // first and then check
      Enums::assertExpected<CV, Enums::CrossValidationMethod::LOOCV>();
      assert(
          isReady() &&
          "Attempting to compute LOOCV values while COD is not in a complete "
          "state.");
      const arma::vec resid{residuals()};

      if (!isReady()) {
        return arma::datum::nan;
      }

      const arma::vec diagH{diagHat()};
      return isReady() ? ClosedForm::loocv(resid, diagH) : arma::datum::nan;
    }
  }

  // MSE for test set (for stochastic (K-Fold) cross-validation) - no mean
  // centering
  template <typename TX, typename TY, bool stochastic = kcv,
            typename = std::enable_if_t<stochastic>>
  [[nodiscard]] double testMSE(const TX& Xtest, const TY& yTest) const {
    Enums::assertExpected<CV, Enums::CrossValidationMethod::KCV>();
    Utils::Data::assertMat<TX>();
    Utils::Data::assertVec<TY>();
    assert(isReady() &&
           "Attempting to evaluate out-of-sample performance while COD is not "
           "in a complete state.");

    // Beta is computed on training set
    const arma::vec beta{solve()};

    if constexpr (meanCenter) {
      // Pre-calculate the scalar offset: y_mean - X_means * beta (accounts
      // for the centering shift without copying the whole test matrix)
      const double offset{yTrainMean_.value() -
                          arma::dot(XtrainColMeans_.value(), beta)};

      // Residual = y_test - ((X_test * beta) + offset)
      return arma::mean(arma::square(yTest - ((Xtest * beta) + offset)));
    } else {
      // Standard calculation for non-centered data
      return arma::mean(arma::square(yTest - (Xtest * beta)));
    }
  }

 private:
  // --- Internal modular calculations

  // Solve for minimum-norm least squares solution
  [[nodiscard]] arma::vec solve() const {
    // Solve the triangular system: U'w = Q_thin'y (ZU_ is (p x r))
    arma::vec w{arma::zeros(ncol_)};
    w.head(rank_) =
        arma::solve(arma::trimatu(ZU_.head_rows(rank_)).t(), QTy_.head(rank_));

    // Map back to full space: beta * P = Z * w
    success_ = applyHouseholderOnLeft(w, Householder::Z, false);

    // Un-pivot to original order
    arma::vec beta(ncol_);
    beta(piv_) = w;
    return beta;
  }

  // Compute sum of squared residuals
  [[nodiscard]] double rss() const {
    // Fully-saturated model (including implicit intercept if centered)
    if (rank_ + (meanCenter ? 1u : 0u) == nrow_) {
      return 0.0;
    }

    // We can compute RSS = TSS - ESS = ||y||^2 - ||Q_thin'y||^2
    // [see "Matrix Computations" Golub p.263 4th ed.]
    const double ess{arma::accu(arma::square(QTy_.head(rank_)))};
    const double rss{tss_.value() - ess};
    return std::max(rss, 0.0);
  }

  [[nodiscard]] arma::vec residuals() const {
    // Fully-saturated model (including implicit intercept if centered)
    if (rank_ + (meanCenter ? 1u : 0u) == nrow_) {
      return arma::zeros(nrow_);
    }

    // Zero out the components in the column space (the first 'rank' elements,
    // leaving only the components in the orthogonal complement) [see "Matrix
    // Computations" Golub p.263 4th ed.]
    arma::vec resid{QTy_};
    resid.head(rank_).zeros();

    // Transform back to original space: resid = Q * [0, Q'y.tail(n - rank)]'
    success_ = applyHouseholderOnLeft(resid, Householder::Q, false);
    return resid;
  }

  [[nodiscard]] double traceHat() const noexcept {
    // If we are centering the data, we dropped the intercept term in R and need
    // to add one to correct the rank to the rank of the original design matrix
    constexpr double correction{meanCenter ? 1.0 : 0.0};
    return static_cast<double>(rank_) + correction;  // tr(H) = rank(X)
  }

  // Diagonal of hat matrix
  [[nodiscard]] arma::vec diagHat() const {
    // Leverage values: h_ii = [X(X'X)^-1 X']_ii
    // Using QR=QTZ, H = Q_1Q_1' so h_ii = sum_{j=1}^{rank} q_{ij}^2
    // (rowwise squared norm of thin Q)
    arma::mat Qthin{arma::eye(nrow_, rank_)};
    success_ = applyHouseholderOnLeft(Qthin, Householder::Q, false);
    arma::vec diagH{arma::sum(arma::square(Qthin), 1)};

    // If the data was centered, we need to add 1/n (diag(11')/n) to the
    // diagonal entries to capture the dropped intercept column
    if constexpr (meanCenter) {
      diagH += (1.0 / static_cast<double>(nrow_));
    }

    return diagH;
  }

  // Determine if the decomposition is ready for modular calculations
  [[nodiscard]] bool isReady() const noexcept {
    return isDesignSet_ && isResponseSet_ && success_;
  }

  // Internal helper to do column-pivoted QR decomposition in lapack without
  // forming full n x n matrix Q
  [[nodiscard]] bool colPivotQR() {
    // --- LAPACK dgeqp3: Column-pivoted QR
    /* From the docs:
     * subroutine dgeqp3(
     * integer m, number of rows
     * integer n, number of column
     * double, dimension(lda, *) a, double array [dimension (LDA,N)]
     * integer lda, The leading dimension of the array A
     * integer, dimension(*) jpvt, integer array [dimenson N]
     * double, dimension(*) tau, double array [dimension min(M,N)]
     * double, dimension(*) work, double array [dimension max(1,LWORK)]
     * integer lwork, the dimension of the array WORK
     * integer info, 0 -> success; -i -> i-th argument had an illegal value
     * )
     */
    int m{static_cast<int>(nrow_)};
    int n{static_cast<int>(ncol_)};
    int lda{m};
    arma::ivec jpvt{arma::zeros<arma::ivec>(ncol_)};
    Qtau_.set_size(std::min(nrow_, ncol_));
    int lwork{-1};  // indicates a workspace query to dgeqp3
    int info{0};

    // Workspace query
    double workQuery;
    arma::lapack::geqp3(&m, &n, QR_.memptr(), &lda, jpvt.memptr(),
                        Qtau_.memptr(), &workQuery, &lwork, &info);

    if (info != 0) {
      return false;
    }

    lwork = static_cast<int>(workQuery);
    arma::vec work(lwork);

    // Decomposition
    arma::lapack::geqp3(&m, &n, QR_.memptr(), &lda, jpvt.memptr(),
                        Qtau_.memptr(), work.memptr(), &lwork, &info);
    const bool success{info == 0};

    if (!success) {
      return success;
    }

    // Convert to zero-indexing
    piv_ = arma::conv_to<arma::uvec>::from(jpvt) - 1;
    return success;
  }

  // Internal helper to do economic QR decomposition in lapack without
  // forming the full orthogonal matrix
  [[nodiscard]] bool econQR() {
    /* --- LAPACK dgeqrf: Economic QR factorization
     * From the docs:
     * subroutine dgeqrf(
     * integer m, number of rows
     * integer n, number of column
     * double, dimension(lda, *) a, double array [dimension (LDA,N)]
     * integer lda, The leading dimension of the array A
     * double, dimension(*) tau, double array [dimension min(M,N)]
     * double, dimension(*) work, double array [dimension max(1,LWORK)]
     * integer lwork, the dimension of the array WORK
     * integer info, 0 -> success; -i -> i-th argument had an illegal value
     * )
     */
    int m{static_cast<int>(ZU_.n_rows)};
    int n{static_cast<int>(ZU_.n_cols)};
    int lda{m};
    Ztau_.set_size(std::min(m, n));
    int lwork{-1};
    int info{0};
    double workQuery;

    // Workspace query
    arma::lapack::geqrf(&m, &n, ZU_.memptr(), &lda, Ztau_.memptr(), &workQuery,
                        &lwork, &info);

    if (info != 0) {
      return false;
    }

    lwork = static_cast<int>(workQuery);
    arma::vec work(lwork);

    // Decomposition
    arma::lapack::geqrf(&m, &n, ZU_.memptr(), &lda, Ztau_.memptr(),
                        work.memptr(), &lwork, &info);
    return info == 0;  // indicates success
  }

  // Internal helper to apply a householder reflection using lapack's dormqr
  [[nodiscard]] bool applyHouseholderOnLeft(arma::mat& A,
                                            const Householder householder,
                                            const bool transpose) const {
    /*
     * subroutine dormqr(
     * character side,
     * character trans,
     * integer m,
     * integer n,
     * integer k,
     * double precision, dimension(lda, *) a,
     * integer lda,
     * double precision, dimension(*) tau,
     * double precision, dimension(ldc, *) c,
     * integer ldc,
     * double precision, dimension(*) work,
     * integer lwork,
     * integer info
     * )
     */
    const arma::mat& Q{householder == Householder::Q ? QR_ : ZU_};
    const arma::vec& tau{householder == Householder::Q ? Qtau_ : Ztau_};

    // Make sure dimensions align (for left-application, A and Q must have
    // matching row dimensions)
    if (A.n_rows != Q.n_rows) {
      return false;
    }

    constexpr char side{'L'};
    const char trans{transpose ? 'T' : 'N'};
    const int m{static_cast<int>(Q.n_rows)};
    const int n{static_cast<int>(A.n_cols)};
    const int k{static_cast<int>(tau.n_elem)};
    const int lda{m};
    const int ldc{m};
    int lwork{-1};
    int info{0};
    double workQuery{};

    // Fortran character lengths for dealing with null-terminated characters
    constexpr std::size_t charLen{1};  // both side and trans are a single char

    // Query workspace
    F77_CALL(dormqr)
    (&side, &trans, &m, &n, &k, Q.memptr(), &lda, tau.memptr(), A.memptr(),
     &ldc, &workQuery, &lwork, &info, charLen, charLen);

    if (info != 0) {
      return false;
    }

    lwork = static_cast<int>(workQuery);
    arma::vec work(lwork);

    // Apply Q on left in-place
    F77_CALL(dormqr)
    (&side, &trans, &m, &n, &k, Q.memptr(), &lda, tau.memptr(), A.memptr(),
     &ldc, work.memptr(), &lwork, &info, charLen, charLen);

    return info == 0;  // whether the application was successful or not
  }
};

#endif  // CV_LM_COMPLETEORTHOGONALDECOMPOSITION_H
