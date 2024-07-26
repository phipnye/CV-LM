## cvLM.R: Fast cross-validation for linear and ridge regression models using RcppEigen
##
## This file is part of the cvLM package.

is.linear.reg.model <- function(model) inherits(model, "lm") && all.equal(family(model), gaussian())

validate.args <- function(K.vals, lambda, generalized, data, seed, n.threads) {
  if (any(is.na(K.vals)) || !is.integer(K.vals) || length(K.vals) < 1L)
    stop("Argument 'K.vals' must be a non-empty integer vector.", call. = FALSE)
  
  if (any(K.vals < 2L) || any(K.vals > nrow(data)))
    stop("Invalid number of folds specified that lies outside the valid range: ", 2L, "-", nrow(data), ".", call. = FALSE)
  
  if (length(lambda) != 1L || is.na(lambda) || !is.numeric(lambda) || lambda < 0)
    stop("Argument 'lambda' must be a non-negative numeric scalar.", call. = FALSE)
  
  if (!(isTRUE(generalized) || isFALSE(generalized)))
    stop("Argument 'generalized' should be TRUE or FALSE.", call. = FALSE)
  
  if (isTRUE(generalized) && any(K.vals != nrow(data)))
    stop("Argument 'K.vals' should be equivalent to the number of observations when computing generalized CV.", call. = FALSE)
  
  if (length(seed) != 1L || is.na(seed) || !is.integer(seed))
    stop("Argument 'seed' must be a single integer value.", call. = FALSE)
  
  if (length(n.threads) != 1L || is.na(n.threads) || !is.integer(n.threads) || n.threads < 1L)
    stop("Argument 'n.threads' must be a single positive integer value.", call. = FALSE)
  
  return(invisible())
}

eval.cvLM <- function(y, X, K.vals, lambda, generalized, seed, n.threads) {
  CV.RES <- mapply(
    function(K, y, X, lambda, generalized, seed, n.threads) {
      n.threads <- min(K, n.threads)
      cv.k <- cv.lm.rcpp(y, X, K, lambda, generalized, seed, n.threads)
      return(cv.k)
    },
    K.vals,
    MoreArgs = list(y = y, X = X, lambda = lambda, generalized = generalized, seed = seed,
                    n.threads = n.threads),
    SIMPLIFY = FALSE,
    USE.NAMES = FALSE
  )
  
  CV.RES <- do.call(rbind, CV.RES)
  
  return(CV.RES)
}

cvLM <- function(object, ...) UseMethod("cvLM")

cvLM.formula <- function(object, data, subset, na.action, K.vals = 10L, lambda = 0, generalized = FALSE,
                         seed = 1L, n.threads = 1L, ...) {
  validate.args(K.vals, lambda, generalized, data, seed, n.threads)
  
  if (is.integer(lambda))
    lambda <- as.double(lambda)
  
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("object", "data", "subset", "na.action"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  names(mf)[names(mf) == "object"] <- "formula"
  mf[["drop.unused.levels"]] <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())
  
  if (is.empty.model(mt <- attr(mf, "terms")))
    stop("Empty model specified.", call. = FALSE)
  
  X <- model.matrix(mt, mf)
  y <- model.response(mf, "double")
  
  CV.RES <- eval.cvLM(y, X, K.vals, lambda, generalized, seed, n.threads)
  return(CV.RES)
}

cvLM.lm <- function(object, data, K.vals = 10L, lambda = 0, generalized = FALSE, seed = 1L, n.threads = 1L,
                    ...) {
  validate.args(K.vals, lambda, generalized, data, seed, n.threads)
  
  if (is.integer(lambda))
    lambda <- as.double(lambda)

  formula <- formula(object)
  mf <- model.frame(formula, data)
  mt <- attr(mf, "terms")
  X <- model.matrix(mt, mf)
  y <- model.response(mf, "double")
  
  CV.RES <- eval.cvLM(y, X, K.vals, lambda, generalized, seed, n.threads)
  return(CV.RES)
}

cvLM.glm <- function(object, data, K.vals = 10L, lambda = 0, generalized = FALSE, seed = 1L, n.threads = 1L,
                     ...) {
  if (!is.linear.reg.model(object))
    stop("cvLM only performs cross-validation for linear and ridge regression models.", call. = FALSE)
  class(object) <- c("lm", setdiff(class(object), "lm"))
  NextMethod("cvLM")
}
