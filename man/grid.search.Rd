\name{grid.search}
\alias{grid.search}
\title{Grid Search for Optimal Lambda in Ridge Regression}
\usage{
grid.search(formula, data, subset, na.action, K = 10L, 
            generalized = FALSE, seed = 1L, n.threads = 1L,
            max.lambda = 10000, precision = 0.1, verbose = TRUE)
}
\arguments{
  \item{formula}{a \code{\link{formula}} specifying the model.}
  \item{data}{a \code{\link{data.frame}} containing the variables in the model.}
  \item{subset}{an optional vector specifying a subset of observations to be used in the fitting process. See \code{\link{model.frame}} for more details.}
  \item{na.action}{a function that indicates what should happen when the data contain \code{NA}s. See \code{\link{model.frame}} for more details.}
  \item{K}{an integer specifying the number of folds for cross-validation.}
  \item{generalized}{a logical indicating whether to compute generalized or ordinary cross-validation.}
  \item{seed}{an integer specifying the seed for random number generation.}
  \item{n.threads}{an integer specifying the number of threads for parallel computation.}
  \item{max.lambda}{a numeric specifying the maximum value of lambda to search.}
  \item{precision}{a numeric specifying the precision of the lambda search.}
  \item{verbose}{a logical indicating whether to display a progress bar during the computation process.}
}
\description{
  Performs a grid search to find the optimal lambda value for ridge regression.
}
\details{
  This function performs a grid search to determine the optimal regularization parameter lambda for ridge regression. It evaluates lambda values starting from 0, incrementing by the specified precision up to the maximum lambda provided. The function utilizes cross-validation to assess the performance of each lambda value and selects the one that minimizes the cross-validation error.
}
\value{
  A \code{\link{list}} with the following components:
  \describe{
    \item{CV}{the minimum cross-validation error.}
    \item{lambda}{the corresponding optimal lambda value.}
  }
}
\examples{
data(mtcars)
grid.search(
  formula = mpg ~ ., 
  data = mtcars,
  K = 5L,           # Use 5-fold CV
  max.lambda = 100, # Search values between 0 and 100
  precision = 0.01, # Increment in steps of 0.01
  verbose = FALSE,  # Disable progress bar
)

\dontrun{
set.seed(1L)
n <- 50002L

DF <- data.frame(
  x1 = rnorm(n),
  x2 = runif(n),
  x3 = rlogis(n),
  x4 = rnorm(n),
  x5 = rcauchy(n),
  x6 = rexp(n)
)

DF <- transform(
  DF,
  y = 3.1 * x1 + 1.8 * x2 + 0.9 * x3 + 1.2 * x4 + x5 + 0.6 * x6 + rnorm(n, sd = 10)
)

grid.search(
  formula = y ~ ., 
  data = DF, 
  K = 10L,          # Use 10-fold CV
  max.lambda = 100, # Search values between 0 and 100
  precision = 1,    # Increment in steps of 1
  n.threads = 10L   # Use 10 threads
)

grid.search(
  formula = y ~ ., 
  data = DF, 
  K = n, generalized = TRUE, # Use generalized CV
  max.lambda = 10000,        # Search values between 0 and 10000
  precision = 10,            # Increment in steps of 10
)
}
}
