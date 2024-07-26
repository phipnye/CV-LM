\name{cvLM-package}
\alias{cvLM-package}
\alias{cvLM}
\alias{cvLM.formula}
\alias{cvLM.lm}
\alias{cvLM.glm}
\docType{package}
\title{Cross-validation for linear and ridge regression models}
\description{
This package provides efficient implementations of cross-validation techniques for linear and ridge regression models, leveraging C++ code with \code{Rcpp}, \code{RcppParallel}, and \code{Eigen} libraries. It supports leave-one-out, generalized, and \code{K}-fold cross-validation methods, utilizing Eigen matrices for high performance.
}
\usage{
cvLM(object, ...)
\method{cvLM}{formula}(object, data, subset, na.action, K.vals = 10L, lambda = 0,
     generalized = FALSE, seed = 1L, n.threads = 1L, ...)
\method{cvLM}{lm}(object, data, K.vals = 10L, lambda = 0,
     generalized = FALSE, seed = 1L, n.threads = 1L, ...)
\method{cvLM}{glm}(object, data, K.vals = 10L, lambda = 0,
     generalized = FALSE, seed = 1L, n.threads = 1L, ...)
}
\arguments{
  \item{object}{a \code{\link{formula}}, a linear model (\code{\link{lm}}), or a generalized linear model (\code{\link{glm}}) object.}
  \item{data}{a \code{\link{data.frame}} containing the variables in the model.}
  \item{subset}{an optional vector specifying a subset of observations to be used in the fitting process. See \code{\link{model.frame}} for more details.}
  \item{na.action}{a function that indicates how to handle \code{NA} values in the data. See \code{\link{model.frame}} for more details.}
  \item{K.vals}{an integer vector specifying the number of folds for cross-validation.}
  \item{lambda}{a non-negative numeric scalar specifying the regularization parameter for ridge regression.}
  \item{generalized}{a logical value indicating whether to compute generalized or ordinary cross-validation. Defaults to \code{FALSE} for ordinary cross-validation.}
  \item{seed}{a single integer value specifying the seed for random number generation.}
  \item{n.threads}{a single positive integer value specifying the number of threads for parallel computation.}
  \item{\dots}{additional arguments. Currently, these do not affect the function's behavior.}
}
\details{
  The \code{cvLM} function is a generic function that dispatches to specific methods based on the class of the \code{object} argument.
  
  The \code{cvLM.formula} method performs cross-validation for linear and ridge regression models specified using a formula interface.
  
  The \code{cvLM.lm} method performs cross-validation for linear regression models.
  
  The \code{cvLM.glm} method performs cross-validation for generalized linear models. It currently supports only gaussian family with identity link function.
  
  The cross-validation process involves splitting the data into \code{K} folds, fitting the model on \code{K-1} folds, and evaluating its performance on the remaining fold. This process is repeated \code{K} times, each time with a different fold held out for testing.
  
  The \code{cvLM} functions use closed-form solutions for leave-one-out and generalized cross-validation and efficiently handle the K-fold cross-validation process, optionally using multithreading for faster computation when working with larger data.
}
\value{
  A \code{\link{data.frame}} consisting of columns representing the number of folds, the cross-validation result, and the seed used for the computation.
}
\author{
  Philip Nye, \email{phipnye@proton.me}
}
\references{
  Bates D, Eddelbuettel D (2013). "Fast and Elegant Numerical Linear Algebra Using the RcppEigen Package." Journal of Statistical Software, 52(5), 1-24. \doi{10.18637/jss.v052.i05}.
  
  Aggarwal, C. C. (2020). Linear Algebra and Optimization for Machine Learning: A Textbook. Springer Cham. \doi{10.1007/978-3-030-40344-7}.
  
  Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.). Springer New York, NY. \doi{10.1007/978-0-387-84858-7}.
}
\keyword{package}
\seealso{
  \code{\link{formula}}, \code{\link{lm}}, \code{\link{glm}}
}
\examples{
data(mtcars)
n <- nrow(mtcars)

# Formula method
cvLM(
  mpg ~ .,
  data = mtcars,
  K.vals = n,    # Leave-one-out CV
  lambda = 10    # Shrinkage parameter of 10
)

# lm method
my.lm <- lm(mpg ~ ., data = mtcars)
cvLM(
  my.lm,
  data = mtcars,
  K.vals = c(5L, 8L), # Perform both 5- and 8-fold CV
  n.threads = 8L,     # Allow up to 8 threads for computation
  seed = 1234L
)

# glm method
my.glm <- glm(mpg ~ ., data = mtcars)
cvLM(
  my.glm,
  data = mtcars,
  K.vals = n, generalized = TRUE # Use generalized CV
)
}
