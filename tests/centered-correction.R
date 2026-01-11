set.seed(42)
n <- 50
p <- 100
lambda <- 2.0

# Prepare centered data
X_raw <- matrix(rnorm(n * (p - 1)), n, p - 1)
X <- cbind(1, X_raw)
X_centered <- scale(X_raw, scale = FALSE) # X is now centered

# Compute Hat Matrix the "LONG" way (Primal Definition)
# H = X (X'X + lambda*I)^-1 X'
XtX_L <- t(X) %*% X + lambda * diag(p)
H_long <- X %*% solve(XtX_L) %*% t(X)
trace_long <- sum(diag(H_long))

# Compute using your Dual Shortcut
# trace = n - lambda * trace((XX' + lambda*I)^-1)
XXt_L <- X_centered %*% t(X_centered) + lambda * diag(n)
trace_shortcut <- n - lambda * sum(diag(solve(XXt_L))) + 1

# Compute using primal shortcut
XtX_L_centerd <- t(X_centered) %*% X_centered + lambda * diag(p - 1)
primal_trace_shortcut <- p - lambda * sum(diag(solve(XtX_L_centerd))) + 1

# Results
cat("Trace (Long Primal): ", trace_long, "\n")
cat("Trace (Dual Shortcut):", trace_shortcut, "\n")
cat("Trace (Primal Shortcut):", trace_shortcut, "\n")

# -------------------------------------------------------------------------------------------------------


set.seed(42)
n <- 50
p <- 100
lambda <- 2.0

# Prepare data
X_raw <- matrix(rnorm(n * (p - 1)), n, p - 1)
X_full <- cbind(1, X_raw)               # Model with intercept
X_centered <- scale(X_raw, scale = FALSE) # Centered features only

# Compute "True" Diagonal the LONG way (Full model)
# H = X_full (X_full' X_full + lambda * I_p)^-1 X_full'
XtX_full_L <- t(X_full) %*% X_full + lambda * diag(p)
H_full <- X_full %*% solve(XtX_full_L) %*% t(X_full)
diag_true <- diag(H_full)

# Compute DUAL shortcut + 1/n
# diag(H) = diag(I - lambda * (X_c X_c' + lambda * I_n)^-1) + 1/n
XXt_c_L <- X_centered %*% t(X_centered) + lambda * diag(n)
inv_dual <- solve(XXt_c_L)
diag_dual_shortcut <- (1 - lambda * diag(inv_dual)) + (1/n)

# Compute PRIMAL shortcut + 1/n
# diag(H) = diag(X_c (X_c' X_c + lambda * I_{p-1})^-1 X_c') + 1/n
XtX_c_L <- t(X_centered) %*% X_centered + lambda * diag(p - 1)
diag_primal_shortcut <- diag(X_centered %*% solve(XtX_c_L) %*% t(X_centered)) + (1/n)

# Results
cat("Max difference (True vs Dual Shortcut):  ", max(abs(diag_true - diag_dual_shortcut)), "\n")
cat("Max difference (True vs Primal Shortcut): ", max(abs(diag_true - diag_primal_shortcut)), "\n")
cat("Mean of diagonal (True):                 ", mean(diag_true), "\n")
cat("TraceH / n (from previous logic):        ", (sum(diag_true)/n), "\n")
