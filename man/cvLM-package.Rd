\name{cvLM-package}
\alias{cvLM-package}
\alias{cvLM}
\alias{cvLM.formula}
\alias{cvLM.lm}
\alias{cvLM.glm}
\docType{package}
\title{Cross-validation for linear and ridge regression models}
\description{
\code{cvLM} provides a high-performance framework for cross-validating linear models using \code{RcppEigen}
and \code{RcppParallel}. It implements numerically stable algorithms to compute K-fold, Leave-One-Out (LOO),
and Generalized Cross-Validation (GCV) metrics for both linear and ridge regression.
}
\usage{
cvLM(object, ...)

\method{cvLM}{formula}(object, data, subset, na.action, K.vals = 10L, 
      lambda = 0, generalized = FALSE, seed = 1L, n.threads = 1L, 
      tol = 1e-7, center = TRUE, ...)

\method{cvLM}{lm}(object, data, K.vals = 10L, lambda = 0,
      generalized = FALSE, seed = 1L, n.threads = 1L, 
      tol = 1e-7, center = TRUE, ...)

\method{cvLM}{glm}(object, data, K.vals = 10L, lambda = 0,
      generalized = FALSE, seed = 1L, n.threads = 1L, 
      tol = 1e-7, center = TRUE, ...)
}
\arguments{
  \item{object}{a \code{\link[stats]{formula}}, \code{\link[stats]{lm}}, or \code{\link[stats]{glm}} object.}
  \item{data}{a data frame, list or environment containing the variables in the model. See 
  \code{\link[stats]{model.frame}}.}
  \item{subset}{a specification of the rows/observations to be used. For \code{lm} and \code{glm} methods,
  this is inherited from \code{object}. See \code{\link[stats]{model.frame}}.}
  \item{na.action}{a function indicating how \code{NA} values should be handled. For \code{lm} and \code{glm}
  methods, this is inherited from \code{object}. See \code{\link[stats]{model.frame}}.}
  \item{K.vals}{an integer vector specifying the number of folds. For Leave-One-Out CV, set \code{K.vals}
  equal to the number of observations.}
  \item{lambda}{a non-negative numeric scalar for the ridge regularization parameter. Defaults to 0 (OLS).}
  \item{generalized}{if \code{TRUE}, the Generalized Cross-Validation (GCV) statistic is computed.
  \code{K.vals} is ignored.}
  \item{seed}{an integer used to initialize the random number generator for reproducible fold splits.}
  \item{n.threads}{an integer specifying the number of threads for parallel computation. Set to -1 to use the
  \code{RcppParallel} default. See \code{\link[RcppParallel]{defaultNumThreads}}.}
  \item{tol}{a numeric scalar specifying the tolerance for rank estimation in the underlying orthogonal and
  singular value decompositions. See 'Details' below.}
  \item{center}{if \code{TRUE} (the default), the predictors and response are mean-centered prior to fitting.
  See 'Details' below.}
  \item{\dots}{currently disregarded.}
}
\details{
  \bold{\emph{Numerical Stability and Underdetermined Systems}}
  
  The newest release of \code{cvLM} prioritizes numerical robustness. For Ordinary Least Squares 
  (\eqn{\lambda = 0}), the package employs Complete Orthogonal Decomposition (COD). In the presence of column
  rank-deficiency—where predictors are perfectly collinear or  the number of parameters \eqn{p} exceeds the
  number of observations \eqn{n}—COD allows for the identification of the unique minimum \eqn{L_2} norm
  solution, providing a stable platform for cross-validation in high-dimensional or  ill-conditioned settings.
  The numerical rank is determined by the \code{tol} argument; a pivot is considered nonzero if its absolute
  value is strictly greater than the threshold relative to the largest pivot:
  \deqn{|\text{pivot}| > \text{tol} \times |\text{max\_pivot}|}
  This approach differs from R's \code{lm}, which handles column rank-deficiency by zeroing out coefficients
  for redundant columns, and may result in different coefficient estimates and subsequent predictions when the
  design matrix is not of full-column rank.

  For ridge regression (\eqn{\lambda > 0}), the package utilizes Singular Value Decomposition (SVD). By
  decomposing the design matrix into \eqn{X = UDV^T}, the ridge solution can be computed with high precision,
  avoiding the potential information loss associated with forming the cross-product matrix \eqn{X^TX} required
  for Cholesky-based methods which we used in the first package release. Similar to the COD, a singular value
  is considered nonzero only if it satisfies:
  \deqn{|\sigma_i| > \text{tol} \times |\sigma_{\text{max}}|}
  While this numerical stability comes at a speed cost, it allows for consistency with methods used for
  efficient shrinkage parameter search used by \code{\link{grid.search}}.

  \bold{\emph{Regularization and Data Centering}}
  
  Following the methodology in \emph{The Elements of Statistical Learning} (Hastie et al.), \code{cvLM} now
  defaults to centering the data (\code{center = TRUE}). Centering the predictors \eqn{X} and the response
  \eqn{y} by their respective means is mathematically equivalent to excluding the intercept term from the
  regularization penalty. 

  In the \bold{uncentered} case (\code{center = FALSE}), the objective function penalizes all coefficients,
  including the intercept \eqn{\beta_0}:
  \deqn{\hat{\beta} = \arg\min_{\beta} \|y - X\beta\|^2 + \lambda \|\beta\|^2}

  In the \bold{centered} case (\code{center = TRUE}), the optimization problem is effectively reformulated as:
  \deqn{\hat{\beta} = \arg\min_{\beta} \left\{ \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} x_{ij}
  \beta_j \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 \right\}}

  \bold{\emph{Efficiency of LOOCV and GCV}}
  
  For Leave-One-Out (LOOCV) and Generalized Cross-Validation (GCV), \code{cvLM} avoids the computational cost
  of \eqn{n} refits by utilizing closed-form solutions derived through use of the hat matrix.
}
\value{
  A \code{\link{data.frame}} consisting of columns representing the number of folds, the cross-validation
  metric, and the seed used for the computation.
}
\author{
  Philip Nye, \email{phipnye@gmail.com}
}
\references{
  Bates D, Eddelbuettel D (2013). "Fast and Elegant Numerical Linear Algebra Using the RcppEigen Package." 
  Journal of Statistical Software, 52(5), 1-24. \doi{10.18637/jss.v052.i05}.
  
  Aggarwal, C. C. (2020). Linear Algebra and Optimization for Machine Learning: A Textbook. Springer Cham.
  \doi{10.1007/978-3-030-40344-7}.
  
  Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining,
  Inference, and Prediction (2nd ed.). Springer New York, NY. \doi{10.1007/978-0-387-84858-7}.
  
  Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.). Johns Hopkins University Press,
  Baltimore, MD. \doi{10.56021/9781421407944}.
}
\keyword{package}
\seealso{
  \code{\link{grid.search}}
}
\examples{
data(mtcars)
n <- nrow(mtcars)

# 10-fold CV for a linear regression model
cvLM(mpg ~ ., data = mtcars, K.vals = 10)

# Comparing 5-fold, 10-fold, and Leave-One-Out CV configurations using 2 threads
cvLM(mpg ~ ., data = mtcars, K.vals = c(5, 10, 32), n.threads = 2)

# Ridge regression with analytic GCV (using lm interface)
fitted.lm <- lm(mpg ~ ., data = mtcars)
cvLM(fitted.lm, data = mtcars, lambda = 0.5, generalized = TRUE)

}
