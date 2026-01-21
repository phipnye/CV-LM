## grid.search.R: Quickly search for the optimal value of the regularization parameter in ridge regression
##
## This file is part of the cvLM package.

grid.search <- function(
  formula,
  data,
  subset,
  na.action,
  K = 10L,
  generalized = FALSE,
  seed = 1L,
  n.threads = 1L,
  tol = 1e-7,
  max.lambda = 10000,
  precision = 0.1,
  center = TRUE
) {
  # --- Extract data

  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "na.action"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")

  if (is.empty.model(mt)) {
    stop("Empty model specified.", call. = FALSE)
  }

  X <- model.matrix(mt, mf)
  y <- model.response(mf, "double")

  # --- Confirm validity of arguments

  # Number of folds
  K <- .assert_integer_scalar(K, "K", nonneg = TRUE)

  # Generalized boolean
  .assert_logical_scalar(generalized, "generalized")

  # Seed
  seed <- .assert_integer_scalar(seed, "seed", nonneg = TRUE)

  # Number of threads (-1 -> defaultNumThreads)
  n.threads <- .assert_valid_threads(n.threads)

  # Threshold for SVD decomposition
  tol <- .assert_double_scalar(tol, "tol", nonneg = TRUE)

  # Maximum lambda to check
  max.lambda <- .assert_double_scalar(max.lambda, "max.lambda", nonneg = TRUE)

  # Precision / step size
  precision <- .assert_double_scalar(precision, "precision", nonneg = TRUE)

  # Whether to center the data - affecting whether the intercept term is penalized or not in the case of 
  # ridge regression (can also provide different numbers for undetermined OLS cases)
  .assert_logical_scalar(center, "center")

  # Drop the intercept term if we're centering the data
  if (center && attr(mt, "intercept") == 1L) {
    X <- .drop_intercept(X)
  }

  # Check for valid regression data before passing to C++
  .assert_valid_data(y, X)
  grid.search.rcpp(
    y = y,
    x = X,
    k0 = K,
    maxLambda = max.lambda,
    precision = precision,
    generalized = generalized,
    seed = seed,
    nThreads = n.threads,
    threshold = tol,
    center = center
  )
}
