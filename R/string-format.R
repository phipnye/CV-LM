## string-format.R: String formatting helpers
##
## This file is part of the cvLM package.

.p_a <- function(...) paste0(..., collapse = " & ")
.p_n <- function(...) paste0(..., collapse = "")

.fmt <- function(x, n.digits, big.mark) {
  format(
    round(x, n.digits),
    nsmall = n.digits,
    big.mark = big.mark,
    trim = TRUE
  )
}
