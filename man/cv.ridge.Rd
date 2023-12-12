\name{cv.ridge}
\alias{cv.ridge}
\title{Cross-Validation for Linear Models}
\description{
    Implements cross-validation techniques for ridge regression models using C++, Rcpp, RcppParallel, and Eigen libraries. This function enables Leave-One-Out Cross-Validation (LOOCV) and K-Fold Cross-Validation for ridge regression, utilizing Eigen matrices.
}
\usage{
    cv.ridge(y, X, lambda, K, seed, pivot = TRUE)
}
\seealso{
    \code{\link{par.cv.ridge}} for the parallel version of cross-validation.
}
\arguments{
    \item{y}{Vector of response values.}

    \item{X}{Matrix of predictor variables.}
    
    \item{lambda}{Non-negative ridge parameter.}

    \item{K}{Number of folds for cross-validation.}

    \item{seed}{Seed for reproducibility.}

    \item{pivot}{Boolean for determining whether to perform pivoting for Cholesky decomposition. Options are TRUE (higher stability but potentially slower) and FALSE (potentially faster but less stability).}
}
\value{
    A list containing the value for K used, the cross-validation value, and the seed used.
}
\details{
    The \code{cv.ridge} function uses Eigen matrices from RcppEigen to perform Cholesky decomposition. It supports two decomposition methods:

    \itemize{
        \item When \code{pivot} is set to TRUE, \code{cv.ridge} employs Eigen's robust Cholesky decomposition with pivoting: \deqn{X=P^TLDL^TP}
      
        \item When \code{pivot} is set to FALSE, \code{cv.ridge} employs Eigen's standard Cholesky decomposition without pivoting: \deqn{X=LL^T}
    }

    For more technical details, refer to the \href{https://eigen.tuxfamily.org/dox/group__Cholesky__Module.html}{Eigen Cholesky Module documentation}.
}

\examples{
\dontrun{
library(cvLM)

# Initialize data
set.seed(1234)
n.obs <- 7986L
DF <- data.frame(y = runif(n.obs), x1 = rnorm(n.obs), x2 = rbinom(n.obs, 50, 0.2), x3 = rpois(n.obs, 6))
DF.MAT <- data.matrix(DF)
X <- DF.MAT[, -1L]
y <- DF.MAT[, 1L, drop = FALSE]

# K-fold CV (10 folds)
KCV.pivot <- cv.ridge(y, X, K = 10L, lambda = 0.1, seed = 1L, pivot = TRUE)
KCV.nopivot <- cv.ridge(y, X, K = 10L, lambda = 0.1, seed = 1L, pivot = FALSE)
KCV.pivot.par <- par.cv.ridge(y, X, K = 10L, lambda = 0.1, seed = 1L, pivot = TRUE)
KCV.nopivot.par <- par.cv.ridge(y, X, K = 10L, lambda = 0.1, seed = 1L, pivot = FALSE)
all.equal(
  KCV.pivot$CV,
  KCV.nopivot$CV,
  KCV.pivot.par$CV,
  KCV.nopivot.par$CV
) # TRUE

# microbenchmark::microbenchmark(
#   KCV.pivot = cv.ridge(y, X, K = 10L, lambda = 0.1, seed = 1L, pivot = TRUE),
#   KCV.nopivot = cv.ridge(y, X, K = 10L, lambda = 0.1, seed = 1L, pivot = FALSE),
#   KCV.pivot.par = par.cv.ridge(y, X, K = 10L, lambda = 0.1, seed = 1L, pivot = TRUE),
#   KCV.nopivot.par = par.cv.ridge(y, X, K = 10L, lambda = 0.1, seed = 1L, pivot = FALSE),
#   times = 30, unit = "ms", check = "equal"
# )
# Unit: milliseconds
#             expr      min       lq      mean    median       uq      max neval
#        KCV.pivot 1.236226 1.266457 1.2924697 1.2839205 1.317493 1.398075    30
#      KCV.nopivot 1.234401 1.271082 1.2955565 1.2830830 1.319476 1.379757    30
#    KCV.pivot.par 0.825754 0.879792 0.9093268 0.9122625 0.937225 1.042925    30
#  KCV.nopivot.par 0.811157 0.854554 0.8898809 0.8828740 0.935031 0.991763    30

# LOOCV
LOOCV.pivot <- cv.ridge(y, X, K = n.obs, lambda = 0.1, seed = 1L, pivot = TRUE)
LOOCV.nopivot <- cv.ridge(y, X, K = n.obs, lambda = 0.1, seed = 1L, pivot = FALSE)
LOOCV.pivot.par <- par.cv.ridge(y, X, K = n.obs, lambda = 0.1, seed = 1L, pivot = TRUE)
LOOCV.nopivot.par <- par.cv.ridge(y, X, K = n.obs, lambda = 0.1, seed = 1L, pivot = FALSE)
all.equal(
  LOOCV.pivot$CV,
  LOOCV.nopivot$CV,
  LOOCV.pivot.par$CV,
  LOOCV.nopivot.par$CV
) # TRUE

# microbenchmark::microbenchmark(
#   LOOCV.pivot = cv.ridge(y, X, K = n.obs, lambda = 0.1, seed = 1L, pivot = TRUE),
#   LOOCV.nopivot = cv.ridge(y, X, K = n.obs, lambda = 0.1, seed = 1L, pivot = FALSE),
#   LOOCV.pivot.par = par.cv.ridge(y, X, K = n.obs, lambda = 0.1, seed = 1L, pivot = TRUE),
#   LOOCV.nopivot.par = par.cv.ridge(y, X, K = n.obs, lambda = 0.1, seed = 1L, pivot = FALSE),
#   times = 30, unit = "ms", check = "equal"
# )
# Unit: milliseconds
#               expr      min       lq     mean   median       uq      max neval
#        LOOCV.pivot 646.2455 656.4456 725.2745 712.4955 802.4751 878.1579    30
#      LOOCV.nopivot 649.4966 683.6722 735.0574 719.2485 810.6322 843.2977    30
#    LOOCV.pivot.par 191.2899 206.2856 221.8631 216.0781 223.5576 317.1762    30
#  LOOCV.nopivot.par 185.6204 203.9079 240.3663 217.2960 295.6244 320.6512    30

# Internally calls cv.lm if lambda is set to 0
identical(
  cv.lm(y, X, K = 10L, seed = 1L, pivot = "col", rankCheck = TRUE),
  cv.ridge(y, X, lambda = 0, K = 10L, seed = 1L, pivot = TRUE)
) # TRUE

identical(
  cv.lm(y, X, K = 10L, seed = 1L, pivot = "col", rankCheck = TRUE),
  cv.ridge(y, X, lambda = .Machine$double.xmin, K = 10L, seed = 1L, pivot = TRUE)
) # FALSE

identical(
  par.cv.lm(y, X, K = 10L, seed = 1L, pivot = "col", rankCheck = TRUE),
  par.cv.ridge(y, X, lambda = 0, K = 10L, seed = 1L, pivot = TRUE)
) # TRUE

identical(
  par.cv.lm(y, X, K = 10L, seed = 1L, pivot = "col", rankCheck = TRUE),
  par.cv.ridge(y, X, lambda = .Machine$double.xmin, K = 10L, seed = 1L, pivot = TRUE)
) # FALSE
}
}
