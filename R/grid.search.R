## grid.search.R: Quickly search for the optimal value of the regularization parameter in ridge regression
##
## This file is part of the cvLM package.

grid.search <- function(formula, data, subset, na.action, K = 10L, generalized = FALSE, seed = 1L,
                        n.threads = 1L, max.lambda = 10000, precision = 0.1, verbose = TRUE) {
  if (length(K) != 1L || is.na(K) || !is.integer(K) || K < 2L || K > nrow(data)) {
    stop("Argument 'K' must be a single non-missing integer value between ", 2L, " and ", nrow(data), ".")
  }
  if (!(isTRUE(generalized) || isFALSE(generalized))) {
    stop("Argument 'generalized' should be TRUE or FALSE.")
  }
  if (isTRUE(generalized) && K != nrow(data)) {
    stop("Argument 'K' should be equivalent to the number of observations when computing generalized CV.")
  }
  if (length(seed) != 1L || is.na(seed) || !is.integer(seed)) {
    stop("Argument 'seed' must be a single integer value.")
  }
  if (length(n.threads) != 1L || is.na(n.threads) || !is.integer(n.threads) || n.threads < 1L) {
    stop("Argument 'n.threads' must be a single positive integer value.")
  }
  if (!(isTRUE(verbose) || isFALSE(verbose))) {
    stop("Argument 'verbose' should be TRUE or FALSE.")
  }

  n.threads <- min(K, n.threads)
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "na.action"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf[["drop.unused.levels"]] <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())

  if (is.empty.model(mt <- attr(mf, "terms"))) {
    stop("Empty model specified.")
  }

  X <- model.matrix(mt, mf)
  y <- model.response(mf, "double")
  
  return(list(CV = opt.cv, lambda = opt.lambda))
}
