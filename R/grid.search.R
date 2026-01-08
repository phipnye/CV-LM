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
  penalize.intercept = FALSE
) {
  # --- Extract data

  mf <- match.call(expand.dots = FALSE)
  m <- match(c("object", "data", "subset", "na.action"), names(mf), 0L)
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
  
  # Threshold for SVD decomposition (and tolerance at which we force inclusion of max.lambda)
  tol <- .assert_double_scalar(tol, "tol", nonneg = TRUE)

  # Maximum lambda to check
  max.lambda <- .assert_double_scalar(max.lambda, "max.lambda", nonneg = TRUE)

  # Precision / step size
  precision <- .assert_double_scalar(precision, "precision", nonneg = TRUE)

  # Whether to penalize the intercept coefficient in the case of ridge regression
  .assert_logical_scalar(penalize.intercept, "penalize.intercept")

  # We only center if it's a ridge regression model with an intercept
  centered <- !penalize.intercept && attr(mt, "intercept") == 1L

  # Center the data and drop the intercept column
  if (centered) {
    tmp <- .center_data(y, X, mt)
    y <- tmp$y
    X <- tmp$X
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
    centered = centered
  )
}
