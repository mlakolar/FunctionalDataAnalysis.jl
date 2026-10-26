# Run from the repository root: Rscript test/reference/regression_m7e.R
# References: R 4.5.1; refund 0.1.40; mgcv 1.9-3; fda 6.3.0.
# Sources audited (printed from the installed namespaces):
#   refund pffr, pffr_prepare, apply_sandwich_correction, gam_sandwich_cluster,
#   assemble_cluster_sandwich, build_cluster_id, pffr_gls (removed), fosr, amc,
#   lofocv, lw.test; mgcv smooth.construct.bs.smooth.spec, gam (paraPen).
# Penalized function-on-scalar regression on a dense grid with a numeric
# covariate and a three-level factor (treatment coded). CI reads these small
# CSVs and does not need R.
suppressMessages({
    library(refund)
    library(mgcv)
    library(fda)
})
stopifnot(packageVersion("refund") == "0.1.40", packageVersion("mgcv") == "1.9.3",
          packageVersion("fda") == "6.3.0")
options(digits = 17)
header <- paste("#", R.version.string, "; refund", packageVersion("refund"),
                "; mgcv", packageVersion("mgcv"), "; fda", packageVersion("fda"),
                "; regression_m7e.R")
write_ref <- function(x, name, description) {
    path <- paste0("test/data/m7e_", name, ".csv")
    writeLines(c(header, paste("#", description)), path)
    x <- as.data.frame(x)
    x[] <- lapply(x, function(column) if (is.numeric(column)) sprintf("%.17g", column) else column)
    write.table(x, path, append = TRUE, sep = ",", row.names = FALSE,
                col.names = FALSE, quote = FALSE)
}
rows <- function(quantity, value, component = 0L) {
    data.frame(quantity, index = seq_along(value), component, value = as.vector(value))
}
matrix_rows <- function(quantity, value) {
    value <- as.matrix(value)
    do.call(rbind, lapply(seq_len(ncol(value)), function(j) rows(quantity, value[, j], j)))
}

## Inputs: 40 curves on 31 equally spaced points of [0, 1]. Residual curves are
## correlated within subjects (two smooth random modes plus rough noise).
n <- 40L; J <- 31L; K <- 10L
t <- seq(0, 1, length = J)
i <- seq_len(n)
z1 <- sin(2.3 * i)
level <- c("a", "b", "c")[(i * 7L) %% 3L + 1L]
g <- factor(level)
X <- model.matrix(~ z1 + g)
gb <- X[, "gb"]; gc <- X[, "gc"]
noise <- outer(cos(1.7 * i), sin(pi * t)) * .4 + outer(sin(3.1 * i), cos(2 * pi * t)) * .3 +
    .15 * sin(outer(i, seq_len(J), function(a, b) a * 1.3 + b * 2.9 + a * b * .37))
Y <- outer(rep(1, n), 1 + sin(2 * pi * t)) + outer(z1, 2 * t^2 - t) + outer(gb, cos(pi * t)) +
    outer(gc, .8 * sin(3 * t)) + noise
write_ref(cbind(z1, match(level, c("a", "b", "c")), Y), "input",
          "columns: z1, factor level code (1=a, 2=b, 3=c), then the 31 response values")
znew <- cbind(c(.3, -1.2, .8), c(1, 2, 3))   # z1 and level codes for prediction

## pffr with scalar terms only. pffr's own factor handling (by = factor) adds
## a centered smooth per level; the package uses treatment-coded dummies as
## fosr (model.matrix) does, so pffr receives the numeric dummy columns.
## The package's default coefficient basis is fda's cubic B-spline basis with
## equally spaced knots and an integrated squared second derivative penalty.
## mgcv's "bs" basis with m = c(3, 2) and explicit knots spans the same space
## with the same exact penalty: knots extend three intervals beyond [0, 1]
## (without them mgcv widens the range by 0.1%).
nk <- K - 3L + 1L; dx <- 1 / (nk - 1)
kv <- c(-(3:1) * dx, seq(0, 1, length = nk), 1 + (1:3) * dx)
bs <- list(bs = "bs", k = K, m = c(3, 2))
d <- data.frame(z1 = z1, gb = gb, gc = gc)
d$Y <- Y
fit_pffr <- function(sandwich) pffr(Y ~ z1 + gb + gc, yind = t, data = d, bs.yindex = bs,
    bs.int = bs, knots = list(yindex.vec = kv), sandwich = sandwich)
m <- fit_pffr("none")
mc <- fit_pffr("cluster")
stopifnot(isTRUE(all.equal(m$coefficients, mc$coefficients)))
# Coefficient functions beta_j(t) = L_j(t) theta: the intercept function is the
# parametric constant plus the centered smooth; by-variable smooths are
# evaluated with the covariate set to one.
nterm <- length(m$smooth)
L <- lapply(seq_len(nterm), function(j) {
    s <- m$smooth[[j]]
    nd <- data.frame(yindex.vec = t, z1 = 1, gb = 1, gc = 1)
    out <- matrix(0, J, length(m$coefficients))
    out[, s$first.para:s$last.para] <- PredictMat(s, nd)
    if (j == 1) out[, 1] <- 1
    out
})
beta <- sapply(L, function(Lj) Lj %*% m$coefficients)
se_of <- function(V) sapply(L, function(Lj) sqrt(rowSums((Lj %*% V) * Lj)))
term_edf <- sapply(seq_len(nterm), function(j) {
    s <- m$smooth[[j]]
    sum(m$edf[s$first.para:s$last.para]) + (j == 1) * m$edf[1]
})
# Smoothing parameters in the units of the raw (exact) penalty: mgcv scales
# each penalty by S.scale before applying sp.
scale <- sapply(m$smooth, function(s) s$S.scale)
newdata <- data.frame(z1 = znew[, 1], gb = as.numeric(znew[, 2] == 2), gc = as.numeric(znew[, 2] == 3))
pred <- predict(m, newdata = newdata)
Xnew <- cbind(1, newdata$z1, newdata$gb, newdata$gc)
stopifnot(max(abs(pred - Xnew %*% t(beta))) < 1e-10)
write_ref(rbind(
    rows("lambda", m$sp / scale),
    rows("sp", m$sp),
    rows("S_scale", scale),
    rows("reml", m$gcv.ubre),
    rows("sigma2", m$sig2),
    rows("edf", term_edf),
    rows("edf_total", sum(m$edf)),
    matrix_rows("beta", beta),
    matrix_rows("se_bayesian", se_of(m$Vp)),
    matrix_rows("se_frequentist", se_of(m$Ve)),
    matrix_rows("se_cluster", se_of(mc$Vp)),
    matrix_rows("fitted", t(fitted(m))),
    matrix_rows("new_covariates", znew),
    matrix_rows("prediction", t(pred))),
    "pffr", "pffr(Y ~ z1 + gb + gc), REML, bs basis k=10 m=c(3,2); sandwich none and cluster")

## pffr with refund's default bases: bs.int = list(bs="ps", k=20, m=c(2,1)) for
## the functional intercept and bs.yindex = list(bs="ps", k=5, m=c(2,1)) for the
## scalar terms, knots from the range of yind widened by 0.1% (mgcv "ps").
## The package reproduces these with basis=:pffr (PSplineBasis).
fit_default <- function(sandwich) pffr(Y ~ z1 + gb + gc, yind = t, data = d, sandwich = sandwich)
p0 <- fit_default("none")
pc <- fit_default("cluster")
stopifnot(isTRUE(all.equal(p0$coefficients, pc$coefficients)),
          sapply(p0$smooth, function(s) s$bs.dim) == c(20, 5, 5, 5))
coefficient_maps <- function(m) lapply(seq_along(m$smooth), function(j) {
    s <- m$smooth[[j]]
    nd <- data.frame(yindex.vec = t, z1 = 1, gb = 1, gc = 1)
    out <- matrix(0, J, length(m$coefficients))
    out[, s$first.para:s$last.para] <- PredictMat(s, nd)
    if (j == 1) out[, 1] <- 1
    out
})
L0 <- coefficient_maps(p0)
beta0 <- sapply(L0, function(Lj) Lj %*% p0$coefficients)
se0 <- function(V) sapply(L0, function(Lj) sqrt(rowSums((Lj %*% V) * Lj)))
edf0 <- sapply(seq_along(p0$smooth), function(j) {
    s <- p0$smooth[[j]]
    sum(p0$edf[s$first.para:s$last.para]) + (j == 1) * p0$edf[1]
})
scale0 <- sapply(p0$smooth, function(s) s$S.scale)
pred0 <- predict(p0, newdata = newdata)
stopifnot(max(abs(pred0 - Xnew %*% t(beta0))) < 1e-10)
write_ref(rbind(
    rows("lambda", p0$sp / scale0),
    rows("sp", p0$sp),
    rows("S_scale", scale0),
    rows("reml", p0$gcv.ubre),
    rows("sigma2", p0$sig2),
    rows("edf", edf0),
    rows("edf_total", sum(p0$edf)),
    matrix_rows("beta", beta0),
    matrix_rows("se_bayesian", se0(p0$Vp)),
    matrix_rows("se_cluster", se0(pc$Vp)),
    matrix_rows("fitted", t(fitted(p0))),
    matrix_rows("prediction", t(pred0))),
    "pffr_default", "pffr(Y ~ z1 + gb + gc) with default ps bases (k=20 intercept, k=5 terms, m=c(2,1)), REML; sandwich none and cluster")

## fosr. Its default method="OLS" picks one shared lambda by leave-one-function-
## out CV; with multi.sp = TRUE (the GLS/mix default) its helper amc passes only
## the first penalty to gam, so only the intercept is penalized. amc also never
## forwards gam.method to gam, so every lambda fosr estimates with gam uses
## mgcv's default GCV.Cp, whatever gam.method says. A single shared penalty
## (multi.sp = FALSE) with max.iter = 0 is fosr's penalized OLS fit with lambda
## chosen by GCV and the residual-covariance sandwich GinvXT (I %x% Sigma) GinvXT'.
f_raw <- fosr(Y = Y, X = X, argvals = t, nbasis = K, method = "GLS", multi.sp = FALSE,
              max.iter = 0)
stopifnot(f_raw$fit$gam$method == "GCV")
f_fixed <- fosr(Y = Y, X = X, argvals = t, nbasis = K, method = "OLS", lambda = 0.05)
write_ref(rbind(
    rows("lambda", f_raw$lambda),
    rows("gcv", f_raw$fit$gam$gcv.ubre),
    rows("edf_total", f_raw$edf),
    matrix_rows("beta", f_raw$est.func),
    matrix_rows("se_pooled", f_raw$se.func),
    matrix_rows("fitted", t(f_raw$yhat)),
    matrix_rows("fixed_beta", f_fixed$est.func),
    matrix_rows("fixed_se_pooled", f_fixed$se.func)),
    "fosr_grid", "fosr raw response: GLS/multi.sp=FALSE/max.iter=0 (GCV.Cp) and OLS lambda=0.05")

## fosr with a basis response: least squares in L2 through the exact Gram
## matrix, J^(1/2) from svd as in fosr. The response basis is the cubic
## B-spline basis with 12 functions; its coefficients come from least squares
## on the grid.
rb <- create.bspline.basis(c(0, 1), nbasis = 12, norder = 4)
yfd <- Data2fd(t, t(Y), rb, lambda = 0)
argvals <- seq(0, 1, length = 41)
f_fd <- fosr(fdobj = yfd, X = X, argvals = argvals, method = "GLS", multi.sp = FALSE,
             max.iter = 0)
stopifnot(f_fd$fit$gam$method == "GCV")
write_ref(rbind(
    matrix_rows("coefs", t(yfd$coefs)),
    rows("grid", argvals),
    rows("lambda", f_fd$lambda),
    rows("gcv", f_fd$fit$gam$gcv.ubre),
    rows("edf_total", f_fd$edf),
    matrix_rows("beta", f_fd$est.func),
    matrix_rows("se_pooled", f_fd$se.func),
    matrix_rows("fitted_coefs", t(f_fd$yhat$coefs))),
    "fosr_basis", "fosr fd response (12 cubic B-splines): GLS/multi.sp=FALSE/max.iter=0 (GCV.Cp)")
