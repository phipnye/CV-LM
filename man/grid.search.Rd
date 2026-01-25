\name{grid.search}
\alias{grid.search}
\title{Efficient Grid Search for Optimal Ridge Regularization}
\usage{
grid.search(formula, data, subset, na.action, K = 10L, 
            generalized = FALSE, seed = 1L, n.threads = 1L, 
            tol = 1e-7, max.lambda = 10000, precision = 0.1, 
            center = TRUE)
}
\arguments{
  \item{formula}{a \code{\link[stats]{formula}} specifying the model.}
  \item{data}{a data frame, list or environment containing the variables in the model. See 
  \code{\link[stats]{model.frame}}.}
  \item{subset}{a specification of the rows/observations to be used. See \code{\link[stats]{model.frame}}.}
  \item{na.action}{a function indicating how \code{NA} values should be handled. See
  \code{\link[stats]{model.frame}}.}
  \item{K}{an integer specifying the number of folds. For Leave-One-Out CV, set \code{K} equal to the number
  of observations.}
  \item{generalized}{if \code{TRUE}, the Generalized Cross-Validation (GCV) statistic is computed. \code{K} is
  ignored.}
  \item{seed}{an integer used to initialize the random number generator for reproducible K-fold splits.}
  \item{n.threads}{an integer specifying the number of threads. For K-fold CV, parallelization occurs across
  folds; for GCV/LOOCV, it occurs across the \code{lambda} grid.}
  \item{tol}{numeric scalar specifying the tolerance for rank estimation in the SVD. See \code{\link{cvLM}}.}
  \item{max.lambda}{numeric scalar for the maximum \eqn{\lambda} value in the search grid.}
  \item{precision}{numeric scalar specifying the step size (increment) between \eqn{\lambda} values in the
  grid.}
  \item{center}{if \code{TRUE} (the default), the predictors and response are mean-centered, effectively
  excluding the intercept from the ridge penalty. See \code{\link{cvLM}}.}
}
\description{
  Performs an optimized grid search to find the regularization parameter \eqn{\lambda} that minimizes the
  cross-validation metric for ridge regression.
}
\details{
  \code{grid.search} is designed for high-performance parameter tuning. Unlike naive implementations that
  refit the model for every grid point, this function utilizes Singular Value Decomposition (SVD) of the
  design matrix to evaluate the entire grid analytically.

  For Generalized Cross-Validation (GCV) and Leave-One-Out (LOOCV), the SVD is computed once. Each
  \eqn{\lambda} in the grid is then evaluated by updating the diagonal "shrinkage" matrix, reducing the cost
  of each grid point evaluation from \eqn{O(np^2)} to \eqn{O(\min(n,p))}. 

  The search begins at \eqn{\lambda = 0} and increments by \code{precision} until \code{max.lambda} is reached
  (inclusive). The function returns the \eqn{\lambda} that achieves the minimum cross-validation metric across
  the scheme.
}
\value{
  A \code{\link{list}} with the following components:
  \describe{
    \item{CV}{the minimum cross-validation metric.}
    \item{lambda}{the value of \eqn{\lambda} associated with the minimum metric.}
  }
}
\seealso{
  \code{\link{cvLM}}
}
\examples{
data(mtcars)
grid.search(
  formula = mpg ~ ., 
  data = mtcars,
  K = 5L,           # Use 5-fold CV
  max.lambda = 100, # Search values between 0 and 100
  precision = 0.01  # Increment in steps of 0.01
)

grid.search(
  formula = mpg ~ ., 
  data = mtcars,
  K = nrow(mtcars),    # Use LOOCV
  max.lambda = 10000,  # Search values between 0 and 10000
  precision = 0.5      # Increment in steps of 0.5
)

}
