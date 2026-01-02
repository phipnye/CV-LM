## cvLM.R: Fast cross-validation for linear and ridge regression models using RcppEigen
##
## This file is part of the cvLM package.

is.linear.reg.model <- function(model) inherits(model, "lm") && all.equal(family(model), gaussian())

prepare.data <- function(y, X, mt, lambda) {
  # Identify the intercept column
  intercept.col <- which(colnames(X) == "(Intercept)")

  if (length(intercept.col) > 0L) {
    # Center y
    y <- y - mean(y)
    # Drop intercept and center X columns
    # When you center y and X, the regression line is forced to pass through
    # the origin of the new coordinate system, where the intercept is
    # mathematically guaranteed to be zero, hence a column of ones is not needed anymore
    X <- scale(X[, -intercept.col, drop = FALSE], scale = FALSE)
  }
  return(list(y = y, X = X))
}

validate.args <- function(K.vals, lambda, generalized, data, seed, n.threads) {
  if (any(is.na(K.vals)) || !is.integer(K.vals) || length(K.vals) < 1L) {
    stop("Argument 'K.vals' must be a non-empty integer vector.", call. = FALSE)
  }

  if (any(K.vals < 2L) || any(K.vals > nrow(data))) {
    stop("Invalid number of folds specified that lies outside the valid range: ", 2L, "-", nrow(data), ".", call. = FALSE)
  }

  if (length(lambda) != 1L || is.na(lambda) || !is.numeric(lambda) || lambda < 0) {
    stop("Argument 'lambda' must be a non-negative numeric scalar.", call. = FALSE)
  }

  if (!(isTRUE(generalized) || isFALSE(generalized))) {
    stop("Argument 'generalized' should be TRUE or FALSE.", call. = FALSE)
  }

  if (isTRUE(generalized) && any(K.vals != nrow(data))) {
    stop("Argument 'K.vals' should be equivalent to the number of observations when computing generalized CV.", call. = FALSE)
  }

  if (length(seed) != 1L || is.na(seed) || !is.integer(seed)) {
    stop("Argument 'seed' must be a single integer value.", call. = FALSE)
  }

  if (length(n.threads) != 1L || is.na(n.threads) || !is.integer(n.threads) || n.threads < 1L) {
    stop("Argument 'n.threads' must be a single positive integer value.", call. = FALSE)
  }

  return(invisible())
}

eval.cvLM <- function(y, X, K.vals, lambda, generalized, seed, n.threads, centered = FALSE) {
  cvs <- vapply(
    K.vals,
    function(K) {
      n.threads <- min(K, n.threads)
      return(cv.lm.rcpp(y, X, K, lambda, generalized, seed, n.threads, centered))
    },
    numeric(1L),
    USE.NAMES = FALSE
  )

  return(data.frame(
    K = K.vals,
    CV = cvs,
    seed = seed
  ))
}

cvLM <- function(object, ...) UseMethod("cvLM")

cvLM.formula <- function(object, data, subset, na.action, K.vals = 10L, lambda = 0, generalized = FALSE,
                         seed = 1L, n.threads = 1L, ...) {
  validate.args(K.vals, lambda, generalized, data, seed, n.threads)

  if (is.integer(lambda)) {
    lambda <- as.double(lambda)
  }

  mf <- match.call(expand.dots = FALSE)
  m <- match(c("object", "data", "subset", "na.action"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  names(mf)[names(mf) == "object"] <- "formula"
  mf[["drop.unused.levels"]] <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())

  if (is.empty.model(mt <- attr(mf, "terms"))) {
    stop("Empty model specified.", call. = FALSE)
  }

  X <- model.matrix(mt, mf)
  y <- model.response(mf, "double")

  # We only center if it's a Ridge model (lambda > 0)
  if (lambda > 0 && attr(mt, "intercept") == 1L) {
    prep <- prepare.data(y, X, mt, lambda) # center the data and drop the intercept column
    return(eval.cvLM(prep[["y"]], prep[["X"]], K.vals, lambda, generalized, seed, n.threads, centered = TRUE))
  }

  return(eval.cvLM(y, X, K.vals, lambda, generalized, seed, n.threads))
}

cvLM.lm <- function(object, data, K.vals = 10L, lambda = 0, generalized = FALSE, seed = 1L, n.threads = 1L,
                    ...) {
  validate.args(K.vals, lambda, generalized, data, seed, n.threads)

  if (is.integer(lambda)) {
    lambda <- as.double(lambda)
  }

  formula <- formula(object)
  mf <- model.frame(formula, data)
  mt <- attr(mf, "terms")
  X <- model.matrix(mt, mf)
  y <- model.response(mf, "double")

  # We only center if it's a Ridge model (lambda > 0)
  if (lambda > 0 && attr(mt, "intercept") == 1L) {
    prep <- prepare.data(y, X, mt, lambda) # center the data and drop the intercept column
    return(eval.cvLM(prep[["y"]], prep[["X"]], K.vals, lambda, generalized, seed, n.threads, centered = TRUE))
  }

  return(eval.cvLM(y, X, K.vals, lambda, generalized, seed, n.threads))
}

cvLM.glm <- function(object, data, K.vals = 10L, lambda = 0, generalized = FALSE, seed = 1L, n.threads = 1L,
                     ...) {
  if (!is.linear.reg.model(object)) {
    stop("cvLM only performs cross-validation for linear and ridge regression models.", call. = FALSE)
  }

  class(object) <- c("lm", setdiff(class(object), "lm"))
  NextMethod("cvLM")
}
