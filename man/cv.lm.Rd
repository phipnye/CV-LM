\name{cv.lm}
\alias{cv.lm}
\title{Cross-Validation for Linear Models}
\description{
    Implements cross-validation techniques for linear regression models using C++, Rcpp, RcppParallel, and Eigen libraries. This function enables Leave-One-Out Cross-Validation (LOOCV) and K-Fold Cross-Validation for linear regression, utilizing Eigen matrices.
}
\usage{
    cv.lm(y, X, K, seed, pivot = "full", rankCheck = TRUE)
}
\seealso{
    \code{\link{par.cv.lm}} for the parallel version of cross-validation.
}
\arguments{
    \item{y}{Vector of response values.}

    \item{X}{Matrix of predictor variables.}

    \item{K}{Number of folds for cross-validation.}

    \item{seed}{Seed for reproducibility. Ignored when K equals the number of observations (N).}

    \item{pivot}{Type of pivot for QR decomposition. Options are "full" (highest stability but slowest), "none" (fastest but least stability), and "col" (moderate stability and speed). \emph{NOTE:} User should only use "none" when they are certain the data matrix is of full rank.}

    \item{rankCheck}{Boolean for telling the program to ensure the data matrix has full column rank. Not available when pivot is set to "none".}
}
\value{
    A list containing the value for K used, the cross-validation value, and the seed used.
}
\details{
    The \code{cv.lm} function uses Eigen matrices from RcppEigen to perform QR decomposition. It supports three decomposition methods:

    \itemize{
        \item When \code{pivot} is set to "full," \code{cv.lm} employs Eigen's Householder QR decomposition with full pivoting \deqn{PXP^T=QR}
      
        \item Setting \code{pivot} to "col" triggers the use of Eigen's Householder QR decomposition with column-pivoting \deqn{XP=QR}
      
        \item If \code{pivot} is set to "none," \code{cv.lm} uses Eigen's Householder QR decomposition without pivoting \deqn{X=QR}This method does not reveal the matrix's rank, hence the absence of the \code{rankCheck} feature.
    }
  
    For more technical details, refer to the \href{https://eigen.tuxfamily.org/dox/group__QR__Module.html}{Eigen QR Module documentation}.
}

\examples{
\dontrun{
# install.packages("boot")
library(cvLM)

# Initialize data
set.seed(1234)
n.obs <- 7986L
DF <- data.frame(y = runif(n.obs), x1 = rnorm(n.obs), x2 = rbinom(n.obs, 50, 0.2), x3 = rpois(n.obs, 6))
DF.MAT <- data.matrix(DF)
X <- DF.MAT[, -1L]
y <- DF.MAT[, 1L, drop = FALSE]
glm1 <- glm(y ~ 0 + x1 + x2 + x3, data = DF)

# K-fold CV (10 folds)
KCV.full <- cv.lm(y, X, K = 10L, seed = 1L, pivot = "full", rankCheck = FALSE)
KCV.col <- cv.lm(y, X, K = 10L, seed = 1L, pivot = "col", rankCheck = FALSE)
KCV.none <- cv.lm(y, X, K = 10L, seed = 1L, pivot = "none", rankCheck = FALSE)
KCV.full.par <- par.cv.lm(y, X, K = 10L, seed = 1L, pivot = "full", rankCheck = FALSE)
KCV.col.par <- par.cv.lm(y, X, K = 10L, seed = 1L, pivot = "col", rankCheck = FALSE)
KCV.none.par <- par.cv.lm(y, X, K = 10L, seed = 1L, pivot = "none", rankCheck = FALSE)
all.equal(
  KCV.full$CV,
  KCV.col$CV,
  KCV.none$CV,
  KCV.full.par$CV,
  KCV.col.par$CV,
  KCV.none.par$CV
) # TRUE

# microbenchmark::microbenchmark(
#   KCV.full = cv.lm(y, X, K = 10L, seed = 1L, pivot = "full", rankCheck = FALSE)$CV,
#   KCV.col = cv.lm(y, X, K = 10L, seed = 1L, pivot = "col", rankCheck = FALSE)$CV,
#   KCV.none = cv.lm(y, X, K = 10L, seed = 1L, pivot = "none", rankCheck = FALSE)$CV,
#   KCV.full.par = par.cv.lm(y, X, K = 10L, seed = 1L, pivot = "full", rankCheck = FALSE)$CV,
#   KCV.col.par = par.cv.lm(y, X, K = 10L, seed = 1L, pivot = "col", rankCheck = FALSE)$CV,
#   KCV.none.par = par.cv.lm(y, X, K = 10L, seed = 1L, pivot = "none", rankCheck = FALSE)$CV,
#   KCV.boot = {set.seed(1); boot::cv.glm(DF, glm1, K = 10L)$delta[1L]},
#   unit = "ms", check = "equal", times = 30
# )
# Unit: milliseconds
#          expr       min        lq       mean    median        uq       max neval
#      KCV.full  1.458800  1.491326  1.5120193  1.508614  1.532102  1.581422    30
#       KCV.col  1.273212  1.298542  1.3233712  1.316124  1.336572  1.483381    30
#      KCV.none  1.215489  1.240593  1.2667424  1.264472  1.282532  1.451063    30
#  KCV.full.par  0.868885  0.940312  0.9686063  0.972878  0.989499  1.052555    30
#   KCV.col.par  0.789293  0.891411  0.9167764  0.917442  0.938410  1.017844    30
#  KCV.none.par  0.755575  0.852524  0.8897130  0.880669  0.922519  1.029536    30
#      KCV.boot 42.913467 45.350799 49.0524512 46.498171 48.137839 84.452491    30

# LOOCV (seed gets ignored & no difference between regular and parallelized cvLM)
LOOCV.full <- cv.lm(y, X, K = n.obs, seed = 1L, pivot = "full", rankCheck = FALSE)
LOOCV.col <- cv.lm(y, X, K = n.obs, seed = 1L, pivot = "col", rankCheck = FALSE)
LOOCV.none <- cv.lm(y, X, K = n.obs, seed = 1L, pivot = "none", rankCheck = FALSE)
LOOCV.full.par <- par.cv.lm(y, X, K = n.obs, seed = 1L, pivot = "full", rankCheck = FALSE)
LOOCV.col.par <- par.cv.lm(y, X, K = n.obs, seed = 1L, pivot = "col", rankCheck = FALSE)
LOOCV.none.par <- par.cv.lm(y, X, K = n.obs, seed = 1L, pivot = "none", rankCheck = FALSE)
all.equal(
  LOOCV.full$CV,
  LOOCV.col$CV,
  LOOCV.none$CV,
  LOOCV.full.par$CV,
  LOOCV.col.par$CV,
  LOOCV.none.par$CV
) # TRUE

# microbenchmark::microbenchmark(
#   LOOCV.full = cv.lm(y, X, K = n.obs, seed = 1L, pivot = "full", rankCheck = FALSE)$CV,
#   LOOCV.col = cv.lm(y, X, K = n.obs, seed = 1L, pivot = "col", rankCheck = FALSE)$CV,
#   LOOCV.none = cv.lm(y, X, K = n.obs, seed = 1L, pivot = "none", rankCheck = FALSE)$CV,
#   LOOCV.full.par = par.cv.lm(y, X, K = n.obs, seed = 1L, pivot = "full", rankCheck = FALSE)$CV,
#   LOOCV.col.par = par.cv.lm(y, X, K = n.obs, seed = 1L, pivot = "col", rankCheck = FALSE)$CV,
#   LOOCV.none.par = par.cv.lm(y, X, K = n.obs, seed = 1L, pivot = "none", rankCheck = FALSE)$CV,
#   LOOCV.boot = boot::cv.glm(DF, glm1, K = n.obs)$delta[1L],
#   unit = "s", check = "equal", times = 30
# )
# Unit: seconds
#            expr          min           lq          mean        median           uq          max neval
#      LOOCV.full  0.279313522  0.282662232  0.2892461736  0.2879040780  0.294450729  0.309953378    30
#       LOOCV.col  0.000117561  0.000130233  0.0001422143  0.0001359870  0.000150702  0.000185887    30
#      LOOCV.none  0.000111287  0.000118757  0.0001384967  0.0001330330  0.000158779  0.000179657    30
#  LOOCV.full.par  0.278623021  0.282994375  0.2890490861  0.2882803910  0.291682394  0.304004216    30
#   LOOCV.col.par  0.000118576  0.000127662  0.0001460429  0.0001369255  0.000166326  0.000200325    30
#  LOOCV.none.par  0.000110593  0.000116321  0.0001362101  0.0001300215  0.000156504  0.000186387    30
#      LOOCV.boot 89.579348579 92.129032810 92.9732501921 92.8946394023 93.202399023 96.349058349    30
}
}
