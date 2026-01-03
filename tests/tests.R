library(cvLM)
library(boot)
library(bench)
library(parallel)

n.threads <- detectCores(logical = TRUE)

# OLS ---------------------------------------------------------------------------------------------------

## ---------------------------
## 1. Standard Matrix
## ---------------------------

set.seed(42)
n <- 1000L
p <- 10L
X <- matrix(rnorm(n * p), n, p)
y <- drop(X %*% rnorm(p) + rnorm(n))
df <- data.frame(y, X)
glm.fit <- glm(y ~ ., data = df)

# 1a. K-Fold CV
bench::mark(
  boot = {
    set.seed(42)
    cv.glm(df, glm.fit, K = 10L)$delta[1L]
  },
  cvLM = cvLM(glm.fit, data = df, K.vals = 10L, n.threads = n.threads, seed = 42L)$CV,
  iterations = 10L
)

# 1b. LOOCV (fewer iterations since boot doesn't use close-formed solution)
bench::mark(
  boot = {
    set.seed(42)
    cv.glm(df, glm.fit)$delta[1L]
  },
  cvLM = cvLM(glm.fit, data = df, K.vals = n, n.threads = n.threads, seed = 42L)$CV,
  iterations = 3L
)

# 1c. LOOCV vs GCV
bench::mark(
  cvLM.loocv = cvLM(glm.fit, data = df, K.vals = n, n.threads = n.threads, seed = 42L)$CV,
  cvLM.gcv = cvLM(glm.fit, data = df, K.vals = n, n.threads = n.threads, seed = 42L, generalized = TRUE)$CV,
  check = FALSE, # not identical results
  iterations = 100L
)

## ---------------------------
## 2. Rank-Deficient Matrix
## ---------------------------

set.seed(42)
n.rd <- 1000L
p.rd <- 10L
X.rd <- matrix(rnorm(n.rd * p.rd), n.rd, p.rd)
X.rd[, 3] <- X.rd[, 1] + X.rd[, 2]
y.rd <- drop(X.rd %*% rnorm(p.rd) + rnorm(n.rd))
df.rd <- data.frame(y = y.rd, X.rd)
glm.fit.rd <- glm(y ~ ., data = df.rd)

# 2a. K-Fold CV
bench::mark(
  boot = {
    set.seed(42)
    cv.glm(df.rd, glm.fit.rd, K = 10L)$delta[1L]
  },
  cvLM = cvLM(glm.fit.rd, data = df.rd, K.vals = 10L, n.threads = n.threads, seed = 42L)$CV,
  iterations = 10L
)

# 2b. LOOCV (fewer iterations since boot doesn't use close-formed solution)
bench::mark(
  boot = {
    set.seed(42)
    cv.glm(df.rd, glm.fit.rd)$delta[1L]
  },
  cvLM = cvLM(glm.fit.rd, data = df.rd, K.vals = n.rd, n.threads = n.threads, seed = 42L)$CV,
  iterations = 3L
)

# 2c. LOOCV vs GCV
bench::mark(
  cvLM.loocv = cvLM(glm.fit.rd, data = df.rd, K.vals = n.rd, n.threads = n.threads, seed = 42L)$CV,
  cvLM.gcv = cvLM(glm.fit.rd, data = df.rd, K.vals = n.rd, n.threads = n.threads, seed = 42L, generalized = TRUE)$CV,
  check = FALSE, # not identical results
  iterations = 100L
)

## ---------------------------
## 3. Ill-Conditioned Matrix
## ---------------------------

set.seed(42)
n.ill <- 1000L
p.ill <- 10L
X.ill <- matrix(rnorm(n.ill * p.ill), n.ill, p.ill)

# Create near-perfect collinearity (X3 is almost X1 + X2)
# The noise level (1e-8) will square to 1e-16 in X'X
X.ill[, 3] <- X.ill[, 1] + X.ill[, 2] + rnorm(n.ill, sd = 1e-8)
y.ill <- drop(X.ill %*% rnorm(p.ill) + rnorm(n.ill))
df.ill <- data.frame(y = y.ill, X.ill)

# R's glm/lm uses QR and will likely keep all columns here
glm.fit.ill <- glm(y ~ ., data = df.ill)

# 3a. K-Fold CV
bench::mark(
  boot = {
    set.seed(42)
    cv.glm(df.ill, glm.fit.ill, K = 10L)$delta[1L]
  },
  cvLM = cvLM(glm.fit.ill, data = df.ill, K.vals = 10L, n.threads = n.threads, seed = 42L)$CV,
  iterations = 10L
)

# 3b. LOOCV
bench::mark(
  boot = {
    set.seed(42)
    cv.glm(df.ill, glm.fit.ill)$delta[1L]
  },
  cvLM = cvLM(glm.fit.ill, data = df.ill, K.vals = n.ill, n.threads = n.threads, seed = 42L)$CV,
  iterations = 3L
)

## ---------------------------
## 4. Wide Matrix (n < p)
## ---------------------------

set.seed(42)
n.wide <- 50L
p.wide <- 100L 
X.wide <- matrix(rnorm(n.wide * p.wide), n.wide, p.wide)
y.wide <- drop(X.wide %*% rnorm(p.wide) + rnorm(n.wide))
df.wide <- data.frame(y = y.wide, X.wide)

# OLS Case: glm/lm will zero out p - n coefficients
glm.fit.wide <- glm(y ~ ., data = df.wide)

# 4a. K-Fold CV
bench::mark(
  boot = {
    set.seed(42)
    cv.glm(df.wide, glm.fit.wide, K = 10L)$delta[1L]
  },
  cvLM = cvLM(glm.fit.wide, data = df.wide, K.vals = 10L, n.threads = n.threads, seed = 42L)$CV,
  iterations = 10L
)

# 4b. LOOCV
bench::mark(
  boot = {
    set.seed(42)
    cv.glm(df.wide, glm.fit.wide)$delta[1L]
  },
  cvLM = cvLM(glm.fit.wide, data = df.wide, K.vals = n.wide, n.threads = n.threads, seed = 42L)$CV,
  iterations = 3L
)

# 4c. LOOCV vs GCV (OLS)
bench::mark(
  cvLM.loocv = cvLM(glm.fit.wide, data = df.wide, K.vals = n.wide, n.threads = n.threads, seed = 42L)$CV,
  cvLM.gcv   = cvLM(glm.fit.wide, data = df.wide, K.vals = n.wide, n.threads = n.threads, seed = 42L, generalized = TRUE)$CV,
  check = FALSE, 
  iterations = 100L
)
