## cvLM.R: Fast cross-validation for linear and ridge regression models using RcppEigen
##
## This file is part of the cvLM package.

# Internal function that accepts prepared data and parameters
.cvLM_fit <- function(
  y,
  X,
  K.vals,
  lambda,
  generalized,
  seed,
  n.threads,
  tol,
  center,
  mt
) {
  # --- Confirm validity of arguments

  # Number of folds
  K.vals <- .assert_valid_kvals(K.vals, nrow(X))

  # Shrinkage parameter
  lambda <- .assert_double_scalar(lambda, "lambda", nonneg = TRUE)

  # Generalized boolean
  .assert_logical_scalar(generalized, "generalized")

  # Seed
  seed <- .assert_integer_scalar(seed, "seed", nonneg = FALSE)

  # Number of threads (-1 -> defaultNumThreads)
  n.threads <- .assert_valid_threads(n.threads)

  # Threshold for complete orthogonal decomposition
  tol <- .assert_double_scalar(tol, "tol", nonneg = TRUE)

  # Whether to center the data - affecting whether the intercept term is penalized or not in the case of
  # ridge regression (can also provide different numbers for undetermined OLS cases)
  .assert_logical_scalar(center, "center")

  # Drop the intercept term if we're centering the data
  if (center && attr(mt, "intercept") == 1L) {
    X <- .drop_intercept(X)
  }

  # Check for valid regression data before passing to C++
  .assert_valid_data(y, X)
  
  # If generalized, K doesn't matter so just set it to look like LOOCV since it's an LOOCV shortcut
  if (generalized) {
    K.vals <- nrow(X)
  }

  # Pass off to C++
  cvs <- vapply(
    K.vals,
    function(K) {
      cv.lm.rcpp(
        y = y,
        x = X,
        k0 = K,
        lambda = lambda,
        generalized = generalized,
        seed = seed,
        nThreads = min(K, n.threads),
        threshold = tol,
        center = center
      )
    },
    numeric(1L),
    USE.NAMES = FALSE
  )

  data.frame(K = K.vals, CV = cvs, seed = seed)
}

cvLM <- function(object, ...) UseMethod("cvLM")

# Formula method
cvLM.formula <- function(
  object,
  data,
  subset,
  na.action,
  K.vals = 10L,
  lambda = 0,
  generalized = FALSE,
  seed = 1L,
  n.threads = 1L,
  tol = 1e-7,
  center = TRUE,
  ...
) {
  # --- Extract data (mimic lm() behavior)

  mf <- match.call(expand.dots = FALSE)
  m <- match(c("object", "data", "subset", "na.action"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  names(mf)[names(mf) == "object"] <- "formula"
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")

  if (stats::is.empty.model(mt)) {
    stop("Empty model specified.", call. = FALSE)
  }

  X <- stats::model.matrix(mt, mf)
  y <- stats::model.response(mf, "double")

  .cvLM_fit(
    y = y,
    X = X,
    K.vals = K.vals,
    lambda = lambda,
    generalized = generalized,
    seed = seed,
    n.threads = n.threads,
    tol = tol,
    center = center,
    mt = mt
  )
}

# lm method
cvLM.lm <- function(
  object,
  data,
  K.vals = 10L,
  lambda = 0,
  generalized = FALSE,
  seed = 1L,
  n.threads = 1L,
  tol = 1e-7,
  center = TRUE,
  ...
) {
  # Raise warning for unsupported lm features (weights and offset)
  if (!is.null(object$weights) && length(unique(object$weights)) > 1L) {
    warning(
      "cvLM does not currently support weighted least squares. Weights will be ignored.",
      call. = FALSE
    )
  }

  if (!is.null(object$offset)) {
    warning(
      "cvLM does not currently support offsets. Offset will be ignored.",
      call. = FALSE
    )
  }

  # --- Extract data

  mf <- stats::model.frame(object, data = data)
  mt <- attr(mf, "terms")
  X <- stats::model.matrix(mt, mf)
  y <- stats::model.response(mf, "double")

  .cvLM_fit(
    y = y,
    X = X,
    K.vals = K.vals,
    lambda = lambda,
    generalized = generalized,
    seed = seed,
    n.threads = n.threads,
    tol = tol,
    center = center,
    mt = mt
  )
}

cvLM.glm <- function(
  object,
  data,
  K.vals = 10L,
  lambda = 0,
  generalized = FALSE,
  seed = 1L,
  n.threads = 1L,
  tol = 1e-7,
  center = TRUE,
  ...
) {
  if (!.is_lm(object)) {
    stop(
      "cvLM only performs cross-validation for linear and ridge regression models.",
      call. = FALSE
    )
  }

  # Use NextMethod to dispatch to cvLM.lm
  NextMethod("cvLM")
}
